"""Experiments requested by the pre-submission reviews (R1 = PDF review, R2 = text review).

Endoscapes CVS stream (rated keyframes, 5 s apart, dissection phase up to the first clip), the
same 200 video-level halvings x 3 seeds as Table 3, DINOv2-L (and the other backbones):

  surgery-level        Prop. 4(a): ceil((V+1)(1-a))-th smallest per-operation max      (ours)
  op-CRC               conformal risk control over OPERATIONS, loss 1[max_unsafe p > lam]
                       on the grid of observed maxima: must coincide with the row above (R1 Q1, R2.1)
  op-LTT               Learn-then-Test over operations, same loss (Hoeffding-Bentkus, delta 0.1):
                       P(per-operation risk <= a) >= 0.9 over calibration sets
  op-PAC               training-conditional order statistic: smallest j with
                       P(Beta(V+1-j, j) > a) <= delta = 0.1  (Vovk 2012)
  CRC / LTT Bonferroni frame-level methods at a / N, N = max or mean number of unsafe keyframes
                       per operation on the 10 selection videos, fixed before calibration (R1 W2)
Also: per-split thresholds (equivalence check), a fixed held-out evaluation (31 official
calibration videos -> 40 official test videos) with Clopper-Pearson intervals over operations
(R2.7), utility of the stream (time to first correct confirmation after the surgeons' CVS
onset, persistence, operations confirmed before the clip; R1 W5, R2.8) and unsafe keyframes
per operation (R1 minor 1).

    python scripts/review2/rebuttal.py
"""
from __future__ import annotations

import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "review"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import ARCHS, ROOT, endoscapes_ops, halves, op_metrics, order_stat, save  # noqa: E402
from cdss import riskcontrol as RC                                                   # noqa: E402
from cdss.data import keyframes                                                      # noqa: E402

ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.1


def mmax(o, w):
    s = o["p"][w][~o["cvs"]]
    return s.max() if len(s) else -np.inf


def op_crc(m, alpha):
    grid = np.r_[np.unique(m[np.isfinite(m)]), np.inf]
    L = (m[:, None] > grid[None, :]).astype(float)
    g = RC.crc_index(L, alpha)
    return np.inf if g is None else grid[g]


def op_ltt(m, alpha):
    grid = np.r_[np.unique(m[np.isfinite(m)]), np.inf]
    L = (m[:, None] > grid[None, :]).astype(float)
    g = RC.ltt_index(L, alpha, DELTA)
    return np.inf if g is None else grid[g]


def pac_index(V, alpha, delta=DELTA):
    for j in range(1, V + 1):                       # risk of the j-th smallest ~ Beta(V+1-j, j)
        if beta.sf(alpha, V + 1 - j, j) <= delta:
            return j
    return None


def op_pac(m, alpha):
    m = np.sort(m); j = pac_index(len(m), alpha)
    return np.inf if j is None else float(m[j - 1])


def frame_crc(cal, w, a):
    v = np.concatenate([o["p"][w][~o["cvs"]] for o in cal])
    return order_stat(v, a)


def frame_ltt(cal, w, a):
    p = np.concatenate([o["p"][w] for o in cal]); y = np.concatenate([o["cvs"] for o in cal])
    grid = RC.threshold_grid(p)
    L = (p[:, None] > grid[None]) & ~y[:, None]
    L = L[~y]                                        # P(confirm | not CVS), as in Table 3
    g = RC.ltt_index(L.astype(float), a, DELTA)
    return np.inf if g is None else grid[g]


def utility(tst, w, lam):
    """Per operation with CVS: correct confirmation at all, delay from the surgeons' first CVS
    keyframe to the first correct confirmation, persistence after it; deferral share."""
    conf_ops, delays, persist, deferral = [], [], [], []
    for o in tst:
        c = o["p"][w] > lam; t = o["frame"] / 25.0
        deferral.append(1 - c.mean())
        if not o["cvs"].any():
            continue
        on = t[o["cvs"]][0]
        hit = np.flatnonzero(c & o["cvs"])
        conf_ops.append(len(hit) > 0)
        if len(hit):
            delays.append(t[hit[0]] - on)
            after = o["cvs"] & (t >= t[hit[0]])
            persist.append(c[after].mean())
    return {"ops_cvs_confirmed": float(np.mean(conf_ops)) if conf_ops else np.nan,
            "delay_s_median": float(np.median(delays)) if delays else np.nan,
            "persistence": float(np.mean(persist)) if persist else np.nan,
            "deferral_share": float(np.mean(deferral))}


def job(args):
    arch, seed, ops, N, rep, ci, ti = args
    cal, tst = [ops[i] for i in ci], [ops[i] for i in ti]
    neg, pos = [~o["cvs"] for o in tst], [o["cvs"] for o in tst]
    rows = []
    for w in (1, 10):
        m = np.array([mmax(o, w) for o in cal if (~o["cvs"]).any()])     # ops without unsafe frames cannot err
        for a in ALPHAS:
            lam = {"surgery-level": order_stat(m, a), "op-CRC": op_crc(m, a), "op-LTT": op_ltt(m, a),
                   "op-PAC": op_pac(m, a),
                   "CRC Bonferroni (N max)": frame_crc(cal, w, a / N["max"]),
                   "CRC Bonferroni (N mean)": frame_crc(cal, w, a / N["mean"]),
                   "LTT Bonferroni (N max)": frame_ltt(cal, w, a / N["max"]),
                   "CRC frame-level": frame_crc(cal, w, a)}
            for name, l in lam.items():
                r = {"arch": arch, "seed": seed, "rep": rep, "w": w, "alpha": a, "method": name, "lam": l,
                     **op_metrics([o["p"][w] > l for o in tst], neg, pos)}
                if arch == "dinov2l_b8":
                    r.update(utility(tst, w, l))
                rows.append(r)
    return rows


def unsafe_counts(ops):
    n = np.array([(~o["cvs"]).sum() for o in ops]); return {"max": int(n.max()), "mean": float(n.mean()),
                                                           "median": float(np.median(n))}


def fixed_holdout(arch, seed, ops, N):
    """31 official calibration (val) operations -> 40 official test operations."""
    te = set(keyframes("test").vid.tolist())
    cal = [o for o in ops if o["vid"] not in te]; tst = [o for o in ops if o["vid"] in te]
    from scipy.stats import beta as B
    rows = []
    for w in (1, 10):
        m = np.array([mmax(o, w) for o in cal if (~o["cvs"]).any()])
        for a in ALPHAS:
            for name, l in (("surgery-level", order_stat(m, a)), ("op-LTT", op_ltt(m, a)),
                            ("CRC frame-level", frame_crc(cal, w, a)),
                            ("CRC Bonferroni (N max)", frame_crc(cal, w, a / N["max"])),
                            ("LTT frame-level", frame_ltt(cal, w, a))):
                bad = np.array([((o["p"][w] > l) & ~o["cvs"]).any() for o in tst]); k, n = int(bad.sum()), len(bad)
                lo = 0.0 if k == 0 else B.ppf(0.025, k, n - k + 1); hi = 1.0 if k == n else B.ppf(0.975, k + 1, n - k)
                tp = sum(((o["p"][w] > l) & o["cvs"]).sum() for o in tst); npos = sum(o["cvs"].sum() for o in tst)
                rows.append({"arch": arch, "seed": seed, "w": w, "alpha": a, "method": name, "V_cal": len(cal),
                             "n_test": n, "ops_err": k / n, "ci_lo": lo, "ci_hi": hi, "sens": tp / max(npos, 1)})
    return rows


def main():
    rows, hold, counts = [], [], []
    for arch in ARCHS:
        for seed in (0, 1, 2):
            ops = endoscapes_ops(arch, seed, (1, 10))
            sel = endoscapes_ops(arch, seed, (1, 10), selection=True)
            N = unsafe_counts(sel)
            if arch == "dinov2l_b8" and seed == 0:
                counts.append({"set": "selection (10)", **N}); counts.append({"set": "pool (71)", **unsafe_counts(ops)})
            hold += fixed_holdout(arch, seed, ops, N)
            args = [(arch, seed, ops, N, rep, ci, ti) for rep, ci, ti in halves(len(ops))]
            with Pool(20) as pool:
                for r in pool.imap_unordered(job, args, chunksize=4):
                    rows += r
            print(arch, seed, "N", N, flush=True)
    df = save(rows, "rebuttal_stream"); save(hold, "rebuttal_holdout"); save(counts, "unsafe_counts")
    # equivalence: op-CRC vs order statistic
    piv = df[df.method.isin(["surgery-level", "op-CRC"])].pivot_table(index=["arch", "seed", "rep", "w", "alpha"],
                                                                          columns="method", values="lam")
    print("op-CRC == surgery-level threshold in", float((piv["op-CRC"] == piv["surgery-level"]).mean()), "of splits")
    pd.set_option("display.width", 250)
    d = df[df.arch == "dinov2l_b8"]
    print(d.groupby(["w", "alpha", "method"])[["ops_false_ge1", "ops_false_gt1", "frame_fr", "sens", "fdr",
                                               "ops_cvs_confirmed", "delay_s_median", "persistence",
                                               "deferral_share"]].mean().round(3).to_string())
    print(pd.DataFrame(hold).groupby(["arch", "w", "alpha", "method"])[["ops_err", "ci_lo", "ci_hi", "sens"]].mean().round(3).to_string())
    print(counts)


if __name__ == "__main__":
    main()
