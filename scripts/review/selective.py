"""Review: selective conformal risk control (SCRC; Xu, Guo & Wei, arXiv:2512.12844) and its
operation-level adaptation, with and without k-tolerance.

Selection score (fixed, label-free): g(x) = min_c |2 p_c(x) - 1|, how decisive the model is
on all three CVS criteria (raw frame); for operations, the mean of g over the first 120 s.

  SCRC-I (frames)       Stage 1 accepts frames with g >= q_xi (calibration quantile, target
                        acceptance xi); stage 2 runs CRC on the augmented loss
                        S(x) 1[p > lam, not CVS] with budget ceil((n+1) a xi_LCB) - 1, xi_LCB the
                        DKW lower bound (delta = 0.1).  Bounds P(false confirm | accepted) <= a
                        w.p. >= 1 - delta, treating frames as i.i.d.
  Selective, surgery-level (ours)
                        Same frame filter, cut-off q_xi fixed on the 10 selection videos (independent
                        of calibration and test, so the filter is a fixed function and operations
                        stay exchangeable); stage 2 = surgery-level on accepted frames:
                        m_v = (k+1)-th largest p over the accepted non-CVS frames of v.
                        Bounds P(> k false confirmations in an operation) <= a exactly.
  Operation-selective (ours)
                        Accept an OPERATION if its mean g over the first 120 s is >= a cut-off fixed
                        on the selection videos (target acceptance xi); certify only after 120 s,
                        surgery-level calibrated on accepted calibration operations.
                        Bounds P(any false confirmation | operation accepted) <= a exactly
                        (SCRC-T logic with operations as the units).

    python scripts/review/selective.py
"""
from __future__ import annotations

import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import ALPHAS, ARCHS, endoscapes_ops, halves, op_metrics, order_stat, save, summary  # noqa: E402
from cdss import riskcontrol as RC                                                             # noqa: E402

XIS = (0.5, 0.75, 0.9)
DELTA = 0.1
EARLY_S = 120


def g_frame(op):
    return np.abs(2 * op["pc"] - 1).min(1)


def kth(s, mask, k):
    x = np.sort(s[mask])[::-1]
    return x[k] if len(x) > k else -np.inf


def scrc_i(cal, tst, alpha, xi):
    g = np.concatenate([g_frame(o) for o in cal]); p = np.concatenate([o["p"][1] for o in cal])
    y = np.concatenate([o["cvs"] for o in cal]); n = len(g)
    q = np.quantile(g, 1 - xi)
    S = g >= q
    xi_lcb = max(S.mean() - math.sqrt(math.log(2 / DELTA) / (2 * n)), 0.0)
    budget = math.ceil((n + 1) * alpha * xi_lcb) - 1
    if budget < 0:
        lam = np.inf
    else:
        grid = RC.threshold_grid(p)
        loss = (S[:, None] & (p[:, None] > grid[None]) & ~y[:, None]).sum(0)   # non-increasing in grid
        ok = np.flatnonzero(loss <= budget)
        lam = grid[ok[0]] if len(ok) else np.inf
    return [(g_frame(o) >= q) & (o["p"][1] > lam) for o in tst]


def selective_surgery(cal, tst, alpha, xi, k, w, SEL):
    q = np.quantile(np.concatenate([g_frame(o) for o in SEL]), 1 - xi)      # fixed on selection videos
    m = [kth(o["p"][w], (g_frame(o) >= q) & ~o["cvs"], k) for o in cal]
    lam = order_stat(m, alpha)
    return [(g_frame(o) >= q) & (o["p"][w] > lam) for o in tst]


def op_selective(cal, tst, alpha, xi, w, SEL):
    def early(o):
        return (o["frame"] - o["frame"][0]) / 25.0 >= EARLY_S
    def gop(o):
        e = ~early(o)
        return g_frame(o)[e].mean() if e.any() else 0.0
    q = np.quantile(np.array([gop(o) for o in SEL]), 1 - xi)               # fixed on selection videos
    acc_c = [o for o in cal if gop(o) >= q]
    lam = order_stat([kth(o["p"][w], early(o) & ~o["cvs"], 0) for o in acc_c], alpha)
    conf = [(early(o) & (o["p"][w] > lam)) if gop(o) >= q else np.zeros(len(o["cvs"]), bool) for o in tst]
    acc_t = np.array([gop(o) >= q for o in tst])
    return conf, acc_t


def job(args):
    arch, seed, ops, SEL, rep, ci, ti = args
    cal, tst = [ops[i] for i in ci], [ops[i] for i in ti]
    neg, pos = [~o["cvs"] for o in tst], [o["cvs"] for o in tst]
    base = {"arch": arch, "seed": seed, "rep": rep}
    rows = []
    for alpha in ALPHAS:
        for xi in XIS:
            rows.append({**base, "alpha": alpha, "xi": xi, "method": "SCRC-I (frames)", "k": 0,
                         **op_metrics(scrc_i(cal, tst, alpha, xi), neg, pos)})
            for w in (1, 10):
                for k in (0, 1):
                    rows.append({**base, "alpha": alpha, "xi": xi, "w": w, "k": k,
                                 "method": "Selective surgery-level (ours)",
                                 **op_metrics(selective_surgery(cal, tst, alpha, xi, k, w, SEL), neg, pos)})
                conf, acc = op_selective(cal, tst, alpha, xi, w, SEL)
                r = op_metrics(conf, neg, pos)
                ra = op_metrics([c for c, a in zip(conf, acc) if a], [n for n, a in zip(neg, acc) if a],
                                [p for p, a in zip(pos, acc) if a]) if acc.any() else {}
                rows.append({**base, "alpha": alpha, "xi": xi, "w": w, "k": 0,
                             "method": "Operation-selective surgery-level (ours)", **r,
                             "op_accept": float(acc.mean()),
                             "ops_false_ge1_given_acc": ra.get("ops_false_ge1", np.nan),
                             "sens_given_acc": ra.get("sens", np.nan)})
    return rows


def main():
    rows = []
    for arch in ARCHS:
        for seed in (0, 1, 2):
            ops = endoscapes_ops(arch, seed, (1, 10))
            SEL = endoscapes_ops(arch, seed, (1, 10), selection=True)
            with Pool(20) as pool:
                for r in pool.imap_unordered(job, [(arch, seed, ops, SEL, rep, ci, ti) for rep, ci, ti in halves(len(ops))],
                                             chunksize=4):
                    rows += r
            print(arch, seed, "done", flush=True)
    df = save(rows, "selective")
    summary(df[df.arch == "dinov2l_b8"], ["alpha", "method", "xi", "w", "k"],
            ["ops_false_ge1", "ops_false_gt1", "frame_fr", "sens", "fdr", "coverage", "op_accept",
             "ops_false_ge1_given_acc", "sens_given_acc"])


if __name__ == "__main__":
    main()
