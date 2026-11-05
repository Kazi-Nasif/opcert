"""Is the small bound excess on the official test split a method failure, or split shift?

Pool the CVS keyframes of the official val videos (minus the 10 used for checkpoint
selection / temperature) and the official test videos, then draw 200 random
video-level splits into calibration and evaluation halves. Under exchangeability of
videos, split conformal guarantees the false-reassurance rate *on average over splits*;
LTT guarantees it with probability >= 1 - delta (delta = 0.1) per split.

    python scripts/resplit_check.py --runs dinov2l_b8_s0 resnet50_s0 ...
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
from cdss import decide                               # noqa: E402
from cdss import riskcontrol as RC                    # noqa: E402
from cdss.data import keyframes                       # noqa: E402
from evaluate import causal_logits, fit_temperature   # noqa: E402

ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--splits", type=int, default=200)
    a = ap.parse_args()
    va, te = keyframes("val"), keyframes("test")
    sel_vids = np.random.default_rng(0).choice(np.unique(va.vid), 10, replace=False)
    rows = []
    for run in a.runs:
        d = ROOT / "experiments" / "finetune" / run
        Lva, Lte = causal_logits(d / "logits_val.npz", va, 1), causal_logits(d / "logits_test.npz", te, 1)
        sel = np.isin(va.vid, sel_vids)
        T = fit_temperature(Lva[sel], va.soft[sel])
        L = np.concatenate([Lva[~sel], Lte]); vids = np.concatenate([va.vid[~sel], te.vid])
        R = np.concatenate([va.raters[~sel], te.raters])
        p = decide.joint_from_marginals(1 / (1 + np.exp(-L / T)))
        cvs = (R.sum(1) >= 2).all(1)
        uv = np.unique(vids)
        for rep in range(a.splits):
            rng = np.random.default_rng(rep)
            calv = rng.choice(uv, len(uv) // 2, replace=False)
            c = np.isin(vids, calv)
            for alpha in ALPHAS:
                grid = RC.threshold_grid(p[c, 7])
                Lc = (p[c, 7][:, None] > grid[None, :])[~cvs[c]]
                g = RC.ltt_index(Lc, alpha)
                for name, conf in (
                        ("MAP (point world)", decide.map_confirm(p[~c])),
                        ("CRC-Mondrian (frames pooled)", decide.mondrian_confirm(p[c], cvs[c], p[~c], alpha)),
                        ("LTT (frames pooled)", p[~c, 7] > (grid[g] if g is not None else np.inf)),
                        ("One frame per video (Dunn et al.)",
                         decide.one_per_video_confirm(p[c, 7], cvs[c], vids[c], p[~c, 7], alpha, rng)),
                        ("surgery-level (video max)",
                         decide.video_max_confirm(p[c, 7], cvs[c], vids[c], p[~c, 7], alpha))):
                    y, tv = cvs[~c], vids[~c]
                    bad_video = [((conf & ~y)[tv == v]).any() for v in np.unique(tv)]
                    rows.append({"run": run, "rep": rep, "alpha": alpha, "method": name,
                                 "fr_given_notCVS": (conf & ~y).sum() / max((~y).sum(), 1),
                                 "video_false_confirm": float(np.mean(bad_video)),
                                 "sens": (conf & y).sum() / max(y.sum(), 1),
                                 "fdr": (conf & ~y).sum() / max(conf.sum(), 1)})
        print(run, "done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "experiments" / "resplit"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "resplit.csv", index=False)
    df["arch"] = df.run.str.replace(r"_s\d+$", "", regex=True)
    df["exceeds"] = df.fr_given_notCVS > df.alpha
    g = df.groupby(["arch", "alpha", "method"]).agg(frame_fr=("fr_given_notCVS", "mean"),
                                                     frac_splits_exceeding=("exceeds", "mean"),
                                                     surgeries_with_false_confirm=("video_false_confirm", "mean"),
                                                     sens=("sens", "mean"), fdr=("fdr", "mean")).round(4)
    pd.set_option("display.width", 200)
    print(g.to_string())
    (out / "summary.json").write_text(g.reset_index().to_json(orient="records", indent=1))


if __name__ == "__main__":
    main()
