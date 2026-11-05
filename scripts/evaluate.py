"""Decision-layer evaluation on a fine-tuned CVS model's logits.

Same protocol as feasibility.py: the 10 val videos used for checkpoint selection also fit
the temperature; the other 31 val videos are the conformal calibration set; the 40 test
videos are evaluation only. Temporal evidence = mean of the per-frame logits over the
last `w` seconds (1 fps, past frames only, so it is causal / usable live).

    python scripts/evaluate.py --run experiments/finetune/dinov2l_b8_s0 --window 1 5 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdss import decide                           # noqa: E402
from cdss.data import keyframes                   # noqa: E402
from cdss import riskcontrol as RC                # noqa: E402

ALPHAS = (0.01, 0.02, 0.05, 0.10)
REFS = ("majority", "unanimous", "any", "rater1", "rater2", "rater3")


def causal_logits(npz, kf, window: int) -> np.ndarray:
    """(N, 3) mean logits over the last `window` 1-fps frames ending at each keyframe."""
    z = np.load(npz)
    vid, frame, L = z["vid"], z["frame"], z["logits"]
    out = []
    by_vid = {}
    for i, v in enumerate(vid):
        by_vid.setdefault(int(v), []).append(i)
    for v, f in zip(kf.vid, kf.frame):
        rows = by_vid[int(v)]
        fr = frame[rows]
        pos = int(np.searchsorted(fr, f))
        assert fr[pos] == f
        sel = rows[max(0, pos - window + 1):pos + 1]
        out.append(L[sel].mean(0))
    return np.stack(out)


def fit_temperature(L, Y):
    L, Y = torch.tensor(L), torch.tensor(Y, dtype=torch.float32)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(L / logT.exp(), Y)
        l.backward()
        return l
    opt.step(closure)
    return float(logT.exp())


def evaluate(run: Path, window: int, seed: int = 0):
    from sklearn.metrics import average_precision_score
    va, te = keyframes("val"), keyframes("test")
    sel_vids = np.random.default_rng(seed).choice(np.unique(va.vid), 10, replace=False)
    sel = np.isin(va.vid, sel_vids)
    Lva = causal_logits(run / "logits_val.npz", va, window)
    Lte = causal_logits(run / "logits_test.npz", te, window)
    T = fit_temperature(Lva[sel], va.soft[sel])
    P3c, P3t = 1 / (1 + np.exp(-Lva[~sel] / T)), 1 / (1 + np.exp(-Lte / T))
    P8c, P8t = decide.joint_from_marginals(P3c), decide.joint_from_marginals(P3t)
    Rc = va.raters[~sel]
    maj_c = Rc.sum(1) >= 2
    maj_t = te.reference("majority"); cvs = maj_t.all(1)
    aps = [float(average_precision_score(maj_t[:, j], P3t[:, j])) for j in range(3)]
    summary = {"window_s": window, "T": T, "AP": aps, "mAP": float(np.mean(aps)),
               "AP_cvs": float(average_precision_score(cvs, P8t[:, 7]))}

    c_c = decide.codes(maj_c)
    s_c = -np.log(np.clip(P8c[np.arange(len(c_c)), c_c], decide.EPS, 1))
    cvs_c = maj_c.all(1)
    rows = []

    def rec(method, alpha, confirm, alarm=None, extra=None):
        r = {"window_s": window, "method": method, "alpha": alpha,
             "confirm_rate": float(confirm.mean()), "n_confirm": int(confirm.sum())}
        for ref in REFS:
            c = te.cvs(ref)
            r[f"false_reassure_{ref}"] = int((confirm & ~c).sum())
            r[f"fdr_{ref}"] = float((confirm & ~c).sum() / max(confirm.sum(), 1))
            r[f"sens_{ref}"] = float((confirm & c).sum() / max(c.sum(), 1))
        if alarm is not None:
            r["alarm_rate_nonCVS"] = float(alarm[~cvs].mean())
            r["false_alarm"] = int((alarm & cvs).sum())
            r["false_alarm_any"] = int((alarm & te.cvs("any")).sum())
        r["fr_rate_given_notCVS"] = float((confirm & ~cvs).sum() / max((~cvs).sum(), 1))
        r.update(extra or {})
        rows.append(r)

    rec("MAP (point world)", None, decide.map_confirm(P8t))
    for alpha in ALPHAS:
        q = decide.conformal_quantile(s_c, alpha)
        G = decide.world_sets(P8t, q)
        ct = decide.codes(maj_t)
        rec("ours", alpha, decide.world_confirm(P8t, q), G.any(1) & ~G[:, 7],
            {"q_hat": q, "coverage": float(G[np.arange(len(ct)), ct].mean()),
             "mean_set_size": float(G.sum(1).mean()),
             **{f"C{j+1}_certified_met": float((G.any(1) & np.all(decide.CODES[None, :, j] | ~G, 1)).mean())
                for j in range(3)},
             **{f"C{j+1}_certified_not_met": float((G.any(1) & np.all(~decide.CODES[None, :, j] | ~G, 1)).mean())
                for j in range(3)}})
        qb = decide.conformal_quantile(decide.projected_scores(P8c, cvs_c), alpha)
        rec("projected", alpha, decide.projected_confirm(P8t, qb), decide.projected_alarm(P8t, qb),
            {"q_hat": qb, "coverage": float((decide.projected_scores(P8t, cvs) <= qb).mean())})
        rec("CRC-Mondrian", alpha, decide.mondrian_confirm(P8c, cvs_c, P8t, alpha),
            decide.mondrian_alarm(P8c, cvs_c, P8t, alpha))
        rec("Conformal selection (BH)", alpha, decide.conformal_selection(P8c, cvs_c, P8t, alpha))
        # risk-control baselines: confirm iff P(CVS) > lam
        grid = RC.threshold_grid(P8c[:, 7])
        conf_c = P8c[:, 7][:, None] > grid[None, :]
        L_joint = conf_c & ~cvs_c[:, None]                    # P(confirm and not CVS)
        L_cond = conf_c[~cvs_c]                               # P(confirm | not CVS)
        for name, L, fn in (("CRC (joint)", L_joint, RC.crc_index), ("LTT (joint)", L_joint, RC.ltt_index),
                            ("CRC (given not CVS)", L_cond, RC.crc_index), ("LTT (given not CVS)", L_cond, RC.ltt_index)):
            g = fn(L, alpha)
            lam = np.inf if g is None else grid[g]
            rec(name, alpha, P8t[:, 7] > lam, extra={"lambda": float(lam)})
        taus = np.unique(P8c[:, 7])
        ok = [t for t in taus if (decide.threshold_confirm(P8c, t) & ~cvs_c).mean() <= alpha]
        tau = min(ok) if ok else 1.1
        rec("Tuned threshold", alpha, decide.threshold_confirm(P8t, tau), extra={"tau": float(tau)})
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--window", type=int, nargs="+", default=[1, 5, 10])
    a = ap.parse_args()
    run = ROOT / a.run
    summaries, rows = [], []
    for w in a.window:
        s, r = evaluate(run, w)
        summaries.append(s); rows += r
    df = pd.DataFrame(rows)
    df.to_csv(run / "decisions.csv", index=False)
    (run / "decisions_summary.json").write_text(json.dumps(summaries, indent=1))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    for s in summaries:
        print(f"window {s['window_s']:>2}s  T={s['T']:.2f}  AP C1/C2/C3 = "
              f"{', '.join(f'{100*x:.1f}' for x in s['AP'])}  mAP {100*s['mAP']:.1f}  AP(all three) {100*s['AP_cvs']:.1f}")
    cols = ["window_s", "method", "alpha", "n_confirm", "false_reassure_majority", "fdr_majority",
            "sens_majority", "fr_rate_given_notCVS", "false_reassure_any", "alarm_rate_nonCVS", "false_alarm", "coverage",
            "mean_set_size"]
    print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
