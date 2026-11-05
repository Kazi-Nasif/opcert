"""Surgery-level CVS certification under external transfer (Endoscapes -> Cholec80).

A  Endoscapes calibration: all 71 non-selection Endoscapes videos (val + test), one
   score per video (max P(CVS) over its non-CVS keyframes); deployed on the 74 clean
   Cholec80 videos.
B  Local recalibration on k Cholec80 videos (k = 20, 37), evaluated on the rest;
   100 random video splits.
Frame-pooled comparators (CRC-Mondrian = CRC, LTT) use the same calibration frames.
Reference standards: Cholec80 total >= 5 (dataset's definition) and strict (all 2).

    python scripts/cholec80_surgery_level.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cdss import decide                               # noqa: E402
from cdss import riskcontrol as RC                    # noqa: E402
from cdss.data import keyframes                       # noqa: E402
from evaluate import causal_logits, fit_temperature   # noqa: E402

ALPHAS = (0.05, 0.10, 0.20)


def evaluate(conf, cvs, vids):
    bad = [((conf & ~cvs)[vids == v]).any() for v in np.unique(vids)]
    return {"surgeries_with_false_confirm": float(np.mean(bad)),
            "frame_fr_given_notCVS": float((conf & ~cvs).sum() / max((~cvs).sum(), 1)),
            "sens": float((conf & cvs).sum() / max(cvs.sum(), 1)),
            "confirm_rate": float(conf.mean())}


def methods(pc, yc, vc, pt, alpha):
    grid = RC.threshold_grid(pc)
    Lc = (pc[:, None] > grid[None, :])[~yc]
    g = RC.ltt_index(Lc, alpha)
    P8c = np.zeros((len(pc), 8)); P8c[:, 7] = pc
    P8t = np.zeros((len(pt), 8)); P8t[:, 7] = pt
    return {"Surgery-level": decide.video_max_confirm(pc, yc, vc, pt, alpha),
            "CRC-Mondrian (frames pooled)": decide.mondrian_confirm(P8c, yc, P8t, alpha),
            "LTT (frames pooled)": pt > (grid[g] if g is not None else np.inf)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    c80 = pd.read_csv(ROOT / "data" / "cholec80" / "cholec80_cvs_frames.csv")
    va, te = keyframes("val"), keyframes("test")
    sel_vids = np.random.default_rng(0).choice(np.unique(va.vid), 10, replace=False)
    rows = []
    for s in a.seeds:
        run = ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{s}"
        Lva, Lte = causal_logits(run / "logits_val.npz", va, 1), causal_logits(run / "logits_test.npz", te, 1)
        sel = np.isin(va.vid, sel_vids)
        T = fit_temperature(Lva[sel], va.soft[sel])
        pE = np.prod(1 / (1 + np.exp(-np.concatenate([Lva[~sel], Lte]) / T)), 1)
        yE = (np.concatenate([va.raters[~sel], te.raters]).sum(1) >= 2).all(1)
        vE = np.concatenate([va.vid[~sel], te.vid])
        z = np.load(run / "logits_cholec80.npz")
        pC = np.prod(1 / (1 + np.exp(-z["logits"] / T)), 1); vC = c80.video.to_numpy()
        for ref in ("cvs_authors", "cvs_strict"):
            yC = c80[ref].to_numpy()
            P8 = np.zeros((len(pC), 8)); P8[:, 7] = pC
            mapc = (1 / (1 + np.exp(-z["logits"] / T)) > 0.5).all(1)
            rows.append({"seed": s, "ref": ref, "scenario": "A: Endoscapes calibration", "alpha": None,
                         "method": "MAP (point world)", **evaluate(mapc, yC, vC)})
            for alpha in ALPHAS:
                for name, conf in methods(pE, yE, vE, pC, alpha).items():
                    rows.append({"seed": s, "ref": ref, "scenario": "A: Endoscapes calibration", "alpha": alpha,
                                 "method": name, **evaluate(conf, yC, vC)})
            uv = np.unique(vC)
            for k in (20, 37):
                for rep in range(100):
                    rng = np.random.default_rng(10_000 * k + rep)
                    cm = np.isin(vC, rng.choice(uv, k, replace=False))
                    for alpha in ALPHAS:
                        for name, conf in methods(pC[cm], yC[cm], vC[cm], pC[~cm], alpha).items():
                            rows.append({"seed": s, "ref": ref, "scenario": f"B: recalibrate on {k} Cholec80 videos",
                                         "rep": rep, "alpha": alpha, "method": name,
                                         **evaluate(conf, yC[~cm], vC[~cm])})
        print(f"seed {s} done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "experiments" / "surgery_level"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cholec80_surgery_level.csv", index=False)
    df["alpha"] = df.alpha.fillna(-1)
    g = df.groupby(["ref", "scenario", "alpha", "method"])[
        ["surgeries_with_false_confirm", "frame_fr_given_notCVS", "sens", "confirm_rate"]].mean().round(4)
    pd.set_option("display.width", 220); print(g.to_string())


if __name__ == "__main__":
    main()
