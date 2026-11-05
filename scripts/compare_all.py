"""Full comparison: every certification method x every CVS backbone x every metric,
on Endoscapes, 200 random video-level halvings of the 71 non-selection val+test videos.

Methods (frame-level unless stated)
  Point world (MAP)                   P(C1)P(C2)P(C3) > 0.5 on each criterion
  CRC (= frame-level)            Prop. 2, class-conditional, frames pooled
  LTT                                 Learn-then-Test, frames pooled, delta = 0.1
  CRC + smoothing (10 s)              causal mean of logits over 10 s, then CRC
  ACI (label feedback)                adaptive conformal inference along each test stream
  SGR (selective risk <= a)           selection with guaranteed risk, delta = 0.1
  Conformal selection (BH, FDR a)     Jin & Candes
  One frame / video (Dunn)            exact frame-level, single subsampling
  Repeated subsampling (Dunn, a/2)    level-a frame guarantee
  Surgery-level (ours)           P(any false confirmation per operation) <= a
  Surgery-level, k = 1 (ours)    P(> 1 false confirmation per operation) <= a
  Surgery-level + smoothing      ours on 10 s smoothed confidences

Metrics: per-operation false-confirmation rate (>= 1, and > 1), frame-level false
reassurance P(confirm | not CVS), sensitivity P(confirm | CVS), FDR, confirm rate.

    python scripts/compare_all.py --archs dinov2l_b8 resnet50 surgenetxl endovit
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
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
SPLITS = 200


def load_run(run):
    va, te = keyframes("val"), keyframes("test")
    sel_vids = np.random.default_rng(0).choice(np.unique(va.vid), 10, replace=False)
    sel = np.isin(va.vid, sel_vids)
    d = ROOT / "experiments" / "finetune" / run
    T = fit_temperature(causal_logits(d / "logits_val.npz", va, 1)[sel], va.soft[sel])
    out = {}
    for w in (1, 10):
        L = np.concatenate([causal_logits(d / "logits_val.npz", va, w)[~sel], causal_logits(d / "logits_test.npz", te, w)])
        out[w] = np.prod(1 / (1 + np.exp(-L / T)), 1)
    vids = np.concatenate([va.vid[~sel], te.vid]); frames = np.concatenate([va.frame[~sel], te.frame])
    cvs = (np.concatenate([va.raters[~sel], te.raters]).sum(1) >= 2).all(1)
    o = np.lexsort((frames, vids))                       # (video, time) order for ACI
    return {w: p[o] for w, p in out.items()}, vids[o], cvs[o]


def job(args):
    run, rep, P, vids, cvs = args
    rng = np.random.default_rng(rep)
    uv = np.unique(vids)
    c = np.isin(vids, rng.choice(uv, len(uv) // 2, replace=False)); t = ~c
    y, tv = cvs[t], vids[t]
    rows = []

    def add(method, alpha, conf):
        per_op = np.array([((conf & ~y)[tv == v]).sum() for v in np.unique(tv)])
        rows.append({"run": run, "rep": rep, "alpha": alpha, "method": method,
                     "ops_false_ge1": float((per_op >= 1).mean()), "ops_false_gt1": float((per_op > 1).mean()),
                     "frame_fr": float((conf & ~y).sum() / max((~y).sum(), 1)),
                     "sens": float((conf & y).sum() / max(y.sum(), 1)),
                     "fdr": float((conf & ~y).sum() / max(conf.sum(), 1)), "confirm_rate": float(conf.mean())})

    p1, p10 = P[1], P[10]
    # very small frame-level levels: does a frame-level method reach ours's low-risk region,
    # and if so with what sensitivity? (it carries no per-operation guarantee there)
    for alpha in (0.002, 0.005, 0.01, 0.02):
        for name, pp in (("CRC (= frame-level)", p1), ("CRC + smoothing (10 s)", p10)):
            P8c = np.zeros((c.sum(), 8)); P8c[:, 7] = pp[c]
            P8t = np.zeros((t.sum(), 8)); P8t[:, 7] = pp[t]
            add(name, alpha, decide.mondrian_confirm(P8c, cvs[c], P8t, alpha))
    for alpha in ALPHAS:
        P8c = np.zeros((c.sum(), 8)); P8c[:, 7] = p1[c]
        P8t = np.zeros((t.sum(), 8)); P8t[:, 7] = p1[t]
        add("CRC (= frame-level)", alpha, decide.mondrian_confirm(P8c, cvs[c], P8t, alpha))
        grid = RC.threshold_grid(p1[c]); Lc = (p1[c][:, None] > grid[None])[~cvs[c]]
        g = RC.ltt_index(Lc, alpha); add("LTT", alpha, p1[t] > (grid[g] if g is not None else np.inf))
        P8c[:, 7] = p10[c]; P8t[:, 7] = p10[t]
        add("CRC + smoothing (10 s)", alpha, decide.mondrian_confirm(P8c, cvs[c], P8t, alpha))
        add("ACI (label feedback)", alpha, decide.aci_confirm(np.sort(p1[c][~cvs[c]]), p1[t], y, alpha))
        add("SGR (selective risk)", alpha, decide.sgr_confirm(p1[c], cvs[c], p1[t], risk=alpha))
        P8c[:, 7] = p1[c]; P8t[:, 7] = p1[t]
        add("Conformal selection (BH)", alpha, decide.conformal_selection(P8c, cvs[c], P8t, alpha))
        add("One frame / video (Dunn)", alpha, decide.one_per_video_confirm(p1[c], cvs[c], vids[c], p1[t], alpha, rng))
        add("Repeated subsampling (Dunn, a/2)", alpha,
            decide.repeated_subsampling_confirm(p1[c], cvs[c], vids[c], p1[t], alpha / 2, rng, B=30))
        add("Surgery-level (ours)", alpha, decide.video_max_confirm(p1[c], cvs[c], vids[c], p1[t], alpha))
        add("Surgery-level, k = 1 (ours)", alpha, decide.video_kth_confirm(p1[c], cvs[c], vids[c], p1[t], alpha, 1))
        add("Surgery-level + smoothing (ours)", alpha, decide.video_max_confirm(p10[c], cvs[c], vids[c], p10[t], alpha))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["dinov2l_b8", "resnet50", "surgenetxl", "endovit"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="compare_all.csv")
    a = ap.parse_args()
    rows = []
    for arch in a.archs:
        for s in (0, 1, 2):
            run = f"{arch}_s{s}"
            if not (ROOT / "experiments" / "finetune" / run / "logits_test.npz").exists():
                print("missing", run); continue
            P, vids, cvs = load_run(run)
            with Pool(a.workers) as pool:
                for r in pool.imap_unordered(job, [(run, rep, P, vids, cvs) for rep in range(SPLITS)], chunksize=4):
                    rows += r
            # point world: needs per-criterion probabilities -> recompute once per run
            d = ROOT / "experiments" / "finetune" / run
            va, te = keyframes("val"), keyframes("test")
            sel = np.isin(va.vid, np.random.default_rng(0).choice(np.unique(va.vid), 10, replace=False))
            T = fit_temperature(causal_logits(d / "logits_val.npz", va, 1)[sel], va.soft[sel])
            L = np.concatenate([causal_logits(d / "logits_val.npz", va, 1)[~sel], causal_logits(d / "logits_test.npz", te, 1)])
            vv = np.concatenate([va.vid[~sel], te.vid]); ff = np.concatenate([va.frame[~sel], te.frame])
            o = np.lexsort((ff, vv)); mapc = ((1 / (1 + np.exp(-L / T))) > 0.5).all(1)[o]
            for rep in range(SPLITS):
                rng = np.random.default_rng(rep); uv = np.unique(vids)
                tmask = ~np.isin(vids, rng.choice(uv, len(uv) // 2, replace=False))
                y, tv, conf = cvs[tmask], vids[tmask], mapc[tmask]
                per_op = np.array([((conf & ~y)[tv == v]).sum() for v in np.unique(tv)])
                rows.append({"run": run, "rep": rep, "alpha": None, "method": "Point world (MAP)",
                             "ops_false_ge1": float((per_op >= 1).mean()), "ops_false_gt1": float((per_op > 1).mean()),
                             "frame_fr": float((conf & ~y).sum() / max((~y).sum(), 1)),
                             "sens": float((conf & y).sum() / max(y.sum(), 1)),
                             "fdr": float((conf & ~y).sum() / max(conf.sum(), 1)), "confirm_rate": float(conf.mean())})
            print(run, "done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "experiments" / "compare_all"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / a.out, index=False)
    df["arch"] = df.run.str.replace(r"_s\d+$", "", regex=True); df["alpha"] = df.alpha.fillna(-1)
    g = df.groupby(["arch", "alpha", "method"])[["ops_false_ge1", "ops_false_gt1", "frame_fr", "sens", "fdr", "confirm_rate"]].mean().round(3)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(g.to_string())


if __name__ == "__main__":
    main()
