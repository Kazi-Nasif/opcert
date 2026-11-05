"""External validation on Cholec80-CVS (74 videos with no Endoscapes overlap).

Scenario A  (deploy as-is)  temperature + calibration from Endoscapes, test on Cholec80.
Scenario B  (local recalibration)  calibrate on k Cholec80 videos, test on the rest;
            20 random video splits per k. Temperature still from Endoscapes.
Reference standards: CVS strict (every criterion scored 2) and the dataset's own
definition (total >= 5). Confidence intervals: 2,000-sample bootstrap over test *videos*.

    python scripts/evaluate_cholec80.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cdss import decide                            # noqa: E402
from cdss import riskcontrol as RC                 # noqa: E402
from cdss.data import keyframes                    # noqa: E402
from evaluate import causal_logits, fit_temperature  # noqa: E402

C80 = ROOT / "data" / "cholec80"
ALPHAS = (0.01, 0.02, 0.05, 0.10)


def methods(P8c, cvs_c, P8t, alpha):
    """name -> (confirm, alarm or None)"""
    out = {"CRC-Mondrian": (decide.mondrian_confirm(P8c, cvs_c, P8t, alpha),
                             decide.mondrian_alarm(P8c, cvs_c, P8t, alpha))}
    grid = RC.threshold_grid(P8c[:, 7])
    conf_c = P8c[:, 7][:, None] > grid[None, :]
    for name, L, fn in (("CRC (joint)", conf_c & ~cvs_c[:, None], RC.crc_index),
                        ("LTT (joint)", conf_c & ~cvs_c[:, None], RC.ltt_index),
                        ("LTT (given not CVS)", conf_c[~cvs_c], RC.ltt_index)):
        g = fn(L, alpha); lam = np.inf if g is None else grid[g]
        out[name] = (P8t[:, 7] > lam, None)
    out["Conformal selection (BH)"] = (decide.conformal_selection(P8c, cvs_c, P8t, alpha), None)
    return out


def metrics(confirm, alarm, cvs, vids, n_boot=2000, seed=0):
    def m(idx):
        c, y = confirm[idx], cvs[idx]
        r = {"confirm_rate": c.mean(), "fr_given_notCVS": (c & ~y).sum() / max((~y).sum(), 1),
             "fdr": (c & ~y).sum() / max(c.sum(), 1), "sens": (c & y).sum() / max(y.sum(), 1)}
        if alarm is not None:
            a = alarm[idx]
            r["alarm_on_notCVS"] = (a & ~y).sum() / max((~y).sum(), 1)
            r["false_alarm_given_CVS"] = (a & y).sum() / max(y.sum(), 1)
        return r
    point = m(np.arange(len(cvs)))
    rng = np.random.default_rng(seed)
    uv = np.unique(vids); rows = {v: np.flatnonzero(vids == v) for v in uv}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[v] for v in rng.choice(uv, len(uv), replace=True)])
        boots.append(m(idx))
    B = pd.DataFrame(boots)
    return {k: float(v) for k, v in point.items()} | {f"{k}_lo": float(B[k].quantile(0.025)) for k in B} | \
           {f"{k}_hi": float(B[k].quantile(0.975)) for k in B}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    from sklearn.metrics import average_precision_score
    df = pd.read_csv(C80 / "cholec80_cvs_frames.csv")
    vids = df.video.to_numpy()
    crit = (df[["C1", "C2", "C3"]] == 2).to_numpy()
    rows, info = [], {"frames": len(df), "videos": int(df.video.nunique())}
    for s in a.seeds:
        run = ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{s}"
        va = keyframes("val")
        sel_vids = np.random.default_rng(0).choice(np.unique(va.vid), 10, replace=False)
        sel = np.isin(va.vid, sel_vids)
        Lva = causal_logits(run / "logits_val.npz", va, 1)
        T = fit_temperature(Lva[sel], va.soft[sel])
        P8c = decide.joint_from_marginals(1 / (1 + np.exp(-Lva[~sel] / T)))
        cvs_c = (va.raters[~sel].sum(1) >= 2).all(1)
        z = np.load(run / "logits_cholec80.npz")
        assert (z["video"] == df.video.to_numpy()).all() and (z["frame"] == df.frame.to_numpy()).all()
        P3 = 1 / (1 + np.exp(-z["logits"] / T)); P8 = decide.joint_from_marginals(P3)
        info[f"seed{s}_AP"] = {c: float(average_precision_score(crit[:, j], P3[:, j]))
                               for j, c in enumerate(("C1", "C2", "C3"))}
        for ref in ("cvs_strict", "cvs_authors"):
            cvs = df[ref].to_numpy()
            info[f"seed{s}_AP_cvs_{ref}"] = float(average_precision_score(cvs, P8[:, 7]))
            r = metrics(decide.map_confirm(P8), None, cvs, vids)
            rows.append({"seed": s, "ref": ref, "scenario": "A: Endoscapes calibration", "method": "MAP (point world)", "alpha": None, **r})
            for alpha in ALPHAS:
                for name, (conf, alarm) in methods(P8c, cvs_c, P8, alpha).items():
                    rows.append({"seed": s, "ref": ref, "scenario": "A: Endoscapes calibration", "method": name,
                                 "alpha": alpha, **metrics(conf, alarm, cvs, vids, n_boot=500)})
            # Scenario B: recalibrate on k Cholec80 videos
            uv = np.unique(vids)
            for k in (10, 20):
                for rep in range(20):
                    rng = np.random.default_rng(1000 * k + rep)
                    calv = rng.choice(uv, k, replace=False)
                    cm = np.isin(vids, calv)
                    for alpha in (0.02, 0.05, 0.10):
                        for name, (conf, alarm) in methods(P8[cm], cvs[cm], P8[~cm], alpha).items():
                            c, y = conf, cvs[~cm]
                            rows.append({"seed": s, "ref": ref, "scenario": f"B: recalibrate on {k} Cholec80 videos",
                                         "rep": rep, "method": name, "alpha": alpha, "confirm_rate": c.mean(),
                                         "fr_given_notCVS": (c & ~y).sum() / max((~y).sum(), 1),
                                         "fdr": (c & ~y).sum() / max(c.sum(), 1), "sens": (c & y).sum() / max(y.sum(), 1),
                                         **({"alarm_on_notCVS": (alarm & ~y).sum() / max((~y).sum(), 1),
                                             "false_alarm_given_CVS": (alarm & y).sum() / max(y.sum(), 1)}
                                            if alarm is not None else {})})
        print(f"seed {s} done", flush=True)
    out = ROOT / "experiments" / "cholec80"; out.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(rows); R.to_csv(out / "transfer.csv", index=False)
    (out / "info.json").write_text(json.dumps(info, indent=1))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(json.dumps(info, indent=1))
    R["alpha"] = R.alpha.fillna(-1)
    cols = ["confirm_rate", "fr_given_notCVS", "fdr", "sens", "alarm_on_notCVS", "false_alarm_given_CVS"]
    print(R.groupby(["ref", "scenario", "alpha", "method"])[cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
