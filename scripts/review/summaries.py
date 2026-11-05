"""Review Q5 / weakness 3: operation-level summaries beyond the max, and units between the
frame and the operation.

Every certificate below issues "CVS achieved" on keyframe t iff a causal statistic s_t
exceeds a threshold lam; its calibration score per operation is the (k+1)-th largest s_t
over the operation's non-CVS keyframes, so P(> k false confirmations per operation) <= a
exactly (Props. 4-5).  Statistics:
  raw         s_t = P(CVS) at t
  smooth w    s_t = P(CVS) from logits averaged over the last w seconds (1 fps)
  persist r   s_t = min of the smoothed P(CVS) over the last r keyframes (5 s apart):
              confirm only a sustained CVS signal
Intermediate units (frames < windows < operation), CRC over units, frames pooled within:
  window-pooled  units = consecutive 60 s windows of every calibration operation, unit
                 score = max over its non-CVS frames; CRC over all windows (asymptotic)
  window-1/op    one random 60 s window per calibration operation (exact per window)
Summary chosen from data: half of the calibration operations pick the (w, r) with the best
sensitivity at the target level, the other half calibrates it (exact, fewer operations).

    python scripts/review/summaries.py
"""
from __future__ import annotations

import itertools
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ALPHAS, ARCHS, endoscapes_ops, halves, op_metrics, order_stat, save, summary  # noqa: E402

WINDOWS = (1, 5, 10, 30)
PERSIST = (1, 2, 3)
KS = (0, 1, 2)
VARIANTS = [(w, r) for w in WINDOWS for r in PERSIST]


def stat(op, w, r):
    p = op["p"][w]
    if r == 1:
        return p
    return np.array([p[max(0, i - r + 1):i + 1].min() for i in range(len(p))])


def kth(s, neg, k):
    x = np.sort(s[neg])[::-1]
    return x[k] if len(x) > k else -np.inf


def thr(ops, S, k, alpha):
    # k = 0: operations without an unsafe frame cannot err and are left out (as in Table 3)
    return order_stat([kth(S[o["vid"]], o["cvs"] == 0, k) for o in ops if k > 0 or (o["cvs"] == 0).any()], alpha)


def evaluate(ops, S, lam):
    return op_metrics([S[o["vid"]] > lam for o in ops], [~o["cvs"] for o in ops], [o["cvs"] for o in ops])


def windows_of(op, width_s=60):
    t = op["frame"] / 25.0; t0 = t[0]
    return [np.flatnonzero((t >= a) & (t < a + width_s)) for a in np.arange(t0, t[-1] + 1e-9, width_s)]


def job(args):
    arch, seed, ops, stats, rep, ci, ti = args
    cal, tst = [ops[i] for i in ci], [ops[i] for i in ti]
    rows = []
    base = {"arch": arch, "seed": seed, "rep": rep, "V": len(cal)}
    for alpha in ALPHAS:
        for (w, r), S in stats.items():
            for k in KS:
                if k > 0 and (w, r) not in ((1, 1), (10, 1)):
                    continue
                lam = thr(cal, S, k, alpha)
                rows.append({**base, "alpha": alpha, "method": "surgery-level", "w": w, "r": r, "k": k,
                             **evaluate(tst, S, lam)})
        # intermediate units on raw scores
        S = stats[(1, 1)]
        units, rng = [], np.random.default_rng(rep)
        one = []
        for o in cal:
            sc = [(S[o["vid"]][ix][~o["cvs"][ix]].max() if (~o["cvs"][ix]).any() else -np.inf)
                  for ix in windows_of(o) if len(ix)]
            units += sc; one.append(sc[rng.integers(len(sc))])
        for name, u in (("window-pooled (CRC)", units), ("window, one per op", one)):
            rows.append({**base, "alpha": alpha, "method": name, "w": 1, "r": 1, "k": 0,
                         **evaluate(tst, S, order_stat(u, alpha))})
        # summary chosen on half of the calibration operations, calibrated on the other half
        h = len(cal) // 2; pick, calib = cal[:h], cal[h:]
        best, best_sens = (1, 1), -1.0
        for (w, r), S in stats.items():
            lam = thr(pick, S, 0, alpha)
            if not math.isfinite(lam):
                continue
            sens = evaluate(pick, S, lam)["sens"]          # in-sample on the picking half
            if sens > best_sens:
                best, best_sens = (w, r), sens
        S = stats[best]
        rows.append({**base, "alpha": alpha, "method": "surgery-level, summary chosen (split)",
                     "w": best[0], "r": best[1], "k": 0, **evaluate(tst, S, thr(calib, S, 0, alpha))})
    return rows


def main():
    rows = []
    for arch in ARCHS:
        for seed in (0, 1, 2):
            ops = endoscapes_ops(arch, seed, WINDOWS)
            stats = {(w, r): {o["vid"]: stat(o, w, r) for o in ops} for w, r in VARIANTS}
            args = [(arch, seed, ops, stats, rep, ci, ti) for rep, ci, ti in halves(len(ops))]
            with Pool(24) as pool:
                for r in pool.imap_unordered(job, args, chunksize=4):
                    rows += r
            print(arch, seed, "done", flush=True)
    df = save(rows, "summaries")
    d = df[df.arch == "dinov2l_b8"]
    summary(d, ["alpha", "method", "w", "r", "k"], ["ops_false_ge1", "ops_false_gt1", "frame_fr", "sens", "fdr"])
    summary(df[(df.method == "surgery-level") & (df.k == 0)], ["arch", "alpha", "w", "r"], ["ops_false_ge1", "sens"])


if __name__ == "__main__":
    main()
