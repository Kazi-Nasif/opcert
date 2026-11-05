"""Shared loading for the review experiments (same splits, temperatures and reference as
the main paper: 10 fixed selection videos fit the temperature; the 71 remaining val+test
videos form the calibration/evaluation pool, halved at random by video)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cdss.data import keyframes                       # noqa: E402
from evaluate import causal_logits, fit_temperature   # noqa: E402

ALPHAS = (0.05, 0.10, 0.20)
ARCHS = ("dinov2l_b8", "resnet50", "surgenetxl", "endovit")


def sel_vids():
    return np.random.default_rng(0).choice(np.unique(keyframes("val").vid), 10, replace=False)


def order_stat(m, alpha):
    """ceil((n+1)(1-alpha))-th smallest of m (inf if not reachable)."""
    m = np.sort(np.asarray(m, float)); k = math.ceil((len(m) + 1) * (1 - alpha))
    return math.inf if len(m) == 0 or k > len(m) else float(m[k - 1])


def temperature(run: Path):
    va = keyframes("val"); sel = np.isin(va.vid, sel_vids())
    return fit_temperature(causal_logits(run / "logits_val.npz", va, 1)[sel], va.soft[sel])


def endoscapes_ops(arch="dinov2l_b8", seed=0, windows=(1, 10), selection=False):
    """List of operations (time-ordered CVS keyframes) with
    p[w]: P(CVS) after w-second causal smoothing, pc: (n,3) criterion probs (w=1),
    raters: (n,3,3), cvs: majority reference.  selection=True returns the 10 selection videos
    instead (never used for calibration or evaluation)."""
    run = ROOT / "experiments" / "finetune" / f"{arch}_s{seed}"
    T = temperature(run)
    sv = set(sel_vids().tolist())
    ops = []
    for split in ("val", "test"):
        kf = keyframes(split)
        Ls = {w: causal_logits(run / f"logits_{split}.npz", kf, w) for w in windows}
        for v in np.unique(kf.vid):
            if (int(v) in sv) != selection:
                continue
            m = np.flatnonzero(kf.vid == v); m = m[np.argsort(kf.frame[m])]
            pc = 1 / (1 + np.exp(-Ls[1][m] / T)) if 1 in Ls else None
            ops.append({"vid": int(v), "frame": kf.frame[m], "raters": kf.raters[m],
                        "cvs": (kf.raters[m].sum(1) >= 2).all(1), "pc": pc,
                        "p": {w: np.prod(1 / (1 + np.exp(-Ls[w][m] / T)), 1) for w in windows}})
    return ops


def halves(n_ops, reps=200):
    """The same 200 video-level halvings as scripts/compare_all.py (operations are in sorted video
    order, so rng.choice over indices selects the same calibration videos as over video ids)."""
    for rep in range(reps):
        ci = np.sort(np.random.default_rng(rep).choice(n_ops, n_ops // 2, replace=False))
        yield rep, ci, np.setdiff1d(np.arange(n_ops), ci)


def op_metrics(conf_list, neg_list, pos_list):
    """conf/neg/pos: per-operation boolean arrays (certified, reference negative, positive)."""
    nfalse = np.array([(c & n).sum() for c, n in zip(conf_list, neg_list)])
    tp = sum((c & p).sum() for c, p in zip(conf_list, pos_list))
    fp = int(nfalse.sum()); npos = sum(p.sum() for p in pos_list); nneg = sum(n.sum() for n in neg_list)
    return {"ops_false_ge1": float((nfalse >= 1).mean()), "ops_false_gt1": float((nfalse > 1).mean()),
            "frame_fr": fp / max(nneg, 1), "sens": tp / max(npos, 1), "fdr": fp / max(fp + tp, 1),
            "coverage": sum(c.sum() for c in conf_list) / max(sum(len(c) for c in conf_list), 1)}


def save(rows, name):
    out = ROOT / "experiments" / "review"; out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(out / f"{name}.csv", index=False)
    return df


def summary(df, by, cols):
    d = df.copy()
    for b in by:
        if d[b].dtype == object:
            d[b] = d[b].fillna("-")
        else:
            d[b] = d[b].fillna(-1)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(d.groupby(by)[cols].mean().round(4).to_string(), flush=True)
