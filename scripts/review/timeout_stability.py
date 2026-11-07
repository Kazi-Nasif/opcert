"""Review Q4: stability of the time-out thresholds lam_c (confirm) and mu (alarm) when few
calibration operations have CVS at the clip.

Endoscapes, DINOv2-L, 200 random halvings x 3 seeds, smoothing 1 / 5 s.  For each split:
  thresholds lam_c, mu; # calibration operations with / without CVS at the clip;
  risk (false confirm | no CVS, false alarm joint) and utility (confirm | CVS, alarm | no CVS);
  decision instability: re-derive the thresholds on a bootstrap resample of the same
  calibration operations and count test operations whose decision changes.
Alarm-threshold grids (CRC is valid on any fixed grid): 'fine' = 400 quantiles of the
calibration scores (paper), 'coarse' = 0.05, 0.10, ..., 0.95 (data-independent).

    python scripts/review/timeout_stability.py
"""
from __future__ import annotations

import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import ALPHAS, ROOT, order_stat, save, sel_vids, summary, temperature  # noqa: E402
from cdss import riskcontrol as RC                                                # noqa: E402
from cdss.data import keyframes                                                   # noqa: E402
from evaluate import causal_logits                                                # noqa: E402

COARSE = np.r_[np.arange(0.05, 0.96, 0.05), np.inf]


def ops_at_clip(seed, w):
    run = ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{seed}"
    T = temperature(run); sv = set(sel_vids().tolist())
    P, C = [], []
    for split in ("val", "test"):
        kf = keyframes(split)
        p = np.prod(1 / (1 + np.exp(-causal_logits(run / f"logits_{split}.npz", kf, w) / T)), 1)
        c = kf.cvs("majority")
        for v in np.unique(kf.vid):
            if int(v) in sv:
                continue
            m = np.flatnonzero(kf.vid == v); last = m[np.argmax(kf.frame[m])]
            P.append(p[last]); C.append(bool(c[last]))
    return np.array(P), np.array(C)


def thresholds(p, c, alpha, grid_kind):
    lam_c = order_stat(p[~c], alpha)
    grid = RC.threshold_grid(p) if grid_kind == "fine" else COARSE
    L = (p[:, None] < grid[None, :]) & c[:, None]
    g = RC.crc_index(L[:, ::-1], alpha)
    mu = -math.inf if g is None else grid[::-1][g]
    return lam_c, mu


def decide(p, lam_c, mu):
    return np.where(p > lam_c, 1, np.where(p < mu, -1, 0))        # 1 confirm, -1 alarm, 0 defer


def job(args):
    seed, w, P, C, rep = args
    h = len(P) // 2                     # the same halvings as main Table 3 (common.halves)
    ci = np.sort(np.random.default_rng(rep).choice(len(P), h, replace=False)); ti = np.setdiff1d(np.arange(len(P)), ci)
    pc, cc, pt, ct = P[ci], C[ci], P[ti], C[ti]
    rng = np.random.default_rng(10_000 + rep)          # bootstrap resampling only
    rows = []
    for alpha in ALPHAS:
        for gk in ("fine", "coarse"):
            lam_c, mu = thresholds(pc, cc, alpha, gk)
            d = decide(pt, lam_c, mu)
            b = rng.integers(0, h, h)
            d2 = decide(pt, *thresholds(pc[b], cc[b], alpha, gk))
            rows.append({"seed": seed, "w": w, "rep": rep, "alpha": alpha, "grid": gk,
                         "lam_c": lam_c, "mu": mu, "n_pos_cal": int(cc.sum()), "n_neg_cal": int((~cc).sum()),
                         "false_confirm_given_noCVS": float(((d == 1) & ~ct).sum() / max((~ct).sum(), 1)),
                         "false_alarm_joint": float(((d == -1) & ct).mean()),
                         "confirm_given_CVS": float(((d == 1) & ct).sum() / max(ct.sum(), 1)),
                         "alarm_given_noCVS": float(((d == -1) & ~ct).sum() / max((~ct).sum(), 1)),
                         "defer": float((d == 0).mean()), "flip_rate_bootstrap": float((d != d2).mean())})
    return rows


def main():
    rows = []
    for seed in (0, 1, 2):
        for w in (1, 5):
            P, C = ops_at_clip(seed, w)
            with Pool(20) as pool:
                for r in pool.imap_unordered(job, [(seed, w, P, C, rep) for rep in range(200)], chunksize=5):
                    rows += r
    df = save(rows, "timeout_stability")
    df["mu_f"] = df.mu.replace(-np.inf, np.nan)
    g = df.groupby(["w", "alpha", "grid"])
    out = g.agg(lam_c_mean=("lam_c", "mean"), lam_c_sd=("lam_c", "std"),
                lam_c_q05=("lam_c", lambda x: x.quantile(.05)), lam_c_q95=("lam_c", lambda x: x.quantile(.95)),
                mu_mean=("mu_f", "mean"), mu_sd=("mu_f", "std"),
                mu_q05=("mu_f", lambda x: x.quantile(.05)), mu_q95=("mu_f", lambda x: x.quantile(.95)),
                n_pos=("n_pos_cal", "mean"), fc=("false_confirm_given_noCVS", "mean"),
                fa=("false_alarm_joint", "mean"), conf=("confirm_given_CVS", "mean"),
                alarm=("alarm_given_noCVS", "mean"), defer=("defer", "mean"), flip=("flip_rate_bootstrap", "mean"))
    import pandas as pd
    pd.set_option("display.width", 250)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
