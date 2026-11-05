"""Feasibility check: does the point-world CVS assessment give false reassurance, and can
ours bound it at useful sensitivity?

Splits (official, by video):
  train (120 videos)       fit the heads
  val  (41 videos)         10 videos -> temperature; 31 videos -> conformal calibration
  test (40 videos)         evaluation only

Reference standards for "CVS achieved": majority of the 3 surgeons (main), unanimous,
and each surgeon on their own. Two calibration targets for ours:
  ours-maj    score of the majority configuration
  ours-all    max over the three raters' configurations -> Gamma covers every surgeon's
              reading at once, so a confirmation is wrong for *no* surgeon w.p. >= 1-alpha

    CUDA_VISIBLE_DEVICES=5 python scripts/feasibility.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdss import decide                                          # noqa: E402
from cdss.data import keyframes                                  # noqa: E402
from cdss.models import Head, load_features, temporal_inputs     # noqa: E402

FEAT = ROOT / "experiments" / "features"
OUT = ROOT / "experiments" / "feasibility"
ALPHAS = (0.01, 0.02, 0.05, 0.10)
REFS = ("majority", "unanimous", "any", "rater1", "rater2", "rater3")


def subset(kf, mask):
    from cdss.data import Keyframes
    return Keyframes(kf.vid[mask], kf.frame[mask], kf.raters[mask], [f for f, k in zip(kf.files, mask) if k])


def main(window=5, seed=0):
    from sklearn.metrics import average_precision_score
    data = {}
    for s in ("train", "val", "test"):
        idx, F = load_features(FEAT / f"dinov2l_{s}.npz")
        kf = keyframes(s)
        data[s] = (kf, temporal_inputs(kf, idx, F, window))
    kf_tr, X_tr = data["train"]; kf_va, X_va = data["val"]; kf_te, X_te = data["test"]
    vids = np.unique(kf_va.vid)
    temp_vids = np.random.default_rng(seed).choice(vids, 10, replace=False)
    t_mask = np.isin(kf_va.vid, temp_vids)
    kf_t, X_t = subset(kf_va, t_mask), X_va[t_mask]
    kf_c, X_c = subset(kf_va, ~t_mask), X_va[~t_mask]

    out = {"n": {"train": len(kf_tr.vid), "temp": len(kf_t.vid), "cal": len(kf_c.vid),
                 "test": len(kf_te.vid)},
           "cvs_prevalence_test": {r: float(kf_te.cvs(r).mean()) for r in REFS}, "heads": {}}
    rows = []
    for kind in ("marginal", "joint"):
        head = Head(X_tr.shape[1], kind).fit(X_tr, kf_tr.raters, seed=seed)
        T = head.fit_temperature(X_t, kf_t.raters)
        P = {n: head.predict(X) for n, X in (("cal", X_c), ("test", X_te))}
        P8 = {n: (decide.joint_from_marginals(p) if kind == "marginal" else p) for n, p in P.items()}
        maj_te = kf_te.reference("majority")
        perc = {"T": T, "AP_cvs_majority": float(average_precision_score(kf_te.cvs(), P8["test"][:, 7]))}
        if kind == "marginal":
            for j, c in enumerate(("C1", "C2", "C3")):
                perc[f"AP_{c}"] = float(average_precision_score(maj_te[:, j], P["test"][:, j]))
        res = {"perception": perc, "alphas": {}}

        # calibration scores
        c_maj = decide.codes(kf_c.reference("majority"))
        s_maj = -np.log(np.clip(P8["cal"][np.arange(len(c_maj)), c_maj], decide.EPS, 1))
        c_r = np.stack([decide.codes(kf_c.raters[:, r].astype(bool)) for r in range(3)], 1)
        s_all = -np.log(np.clip(np.take_along_axis(P8["cal"], c_r, 1), decide.EPS, 1)).max(1)

        def evaluate(name, confirm, alpha, extra=None):
            r = {"confirm_rate": float(confirm.mean()), "n_confirm": int(confirm.sum())}
            for ref in REFS:
                cvs = kf_te.cvs(ref)
                r[f"false_reassurance_{ref}"] = int((confirm & ~cvs).sum())
                r[f"fr_rate_{ref}"] = float((confirm & ~cvs).mean())
                r[f"fdr_{ref}"] = float((confirm & ~cvs).sum() / max(confirm.sum(), 1))
                r[f"sensitivity_{ref}"] = float((confirm & cvs).sum() / max(cvs.sum(), 1))
            r.update(extra or {})
            rows.append({"head": kind, "method": name, "alpha": alpha, **r})
            return r

        evaluate("MAP (point world)", decide.map_confirm(P8["test"]), None)
        cvs_cal = kf_c.cvs("majority")
        for alpha in ALPHAS:
            q_maj = decide.conformal_quantile(s_maj, alpha)
            q_all = decide.conformal_quantile(s_all, alpha)
            c_te_maj = decide.codes(maj_te)
            cov_maj = decide.world_sets(P8["test"], q_maj)[np.arange(len(c_te_maj)), c_te_maj]
            G_all = decide.world_sets(P8["test"], q_all)
            cov_all = np.stack([G_all[np.arange(len(c_te_maj)), decide.codes(kf_te.raters[:, r].astype(bool))]
                                for r in range(3)], 1).all(1)
            evaluate("ours-maj", decide.world_confirm(P8["test"], q_maj), alpha,
                     {"q_hat": q_maj, "coverage": float(cov_maj.mean()),
                      "mean_set_size": float(decide.world_sets(P8["test"], q_maj).sum(1).mean())})
            evaluate("ours-all", decide.world_confirm(P8["test"], q_all), alpha,
                     {"q_hat": q_all, "coverage_all_raters": float(cov_all.mean()),
                      "mean_set_size": float(G_all.sum(1).mean())})
            evaluate("Conformal selection (BH)", decide.conformal_selection(P8["cal"], cvs_cal, P8["test"], alpha), alpha)
            # threshold baseline: smallest tau whose calibration false-reassurance rate <= alpha
            taus = np.unique(P8["cal"][:, 7])
            ok = [t for t in taus if (decide.threshold_confirm(P8["cal"], t) & ~cvs_cal).mean() <= alpha]
            tau = min(ok) if ok else 1.1
            evaluate("Tuned threshold", decide.threshold_confirm(P8["test"], tau), alpha, {"tau": float(tau)})
        out["heads"][kind] = res
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "feasibility.csv", index=False)
    out["perception"] = {k: v["perception"] for k, v in out["heads"].items()}
    (OUT / "feasibility_summary.json").write_text(json.dumps(out, indent=2))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = ["head", "method", "alpha", "confirm_rate", "n_confirm", "false_reassurance_majority",
            "fdr_majority", "sensitivity_majority", "false_reassurance_any", "fdr_unanimous",
            "coverage", "coverage_all_raters", "mean_set_size"]
    print(json.dumps(out["n"]), json.dumps(out["cvs_prevalence_test"]))
    print(json.dumps(out["perception"], indent=1))
    print(df[[c for c in cols if c in df]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
