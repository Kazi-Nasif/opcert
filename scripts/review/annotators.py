"""Review: annotator disagreement.  Endoscapes rates every keyframe by three surgeons.

Reference standards for "CVS achieved": majority (paper), unanimous (all three agree that
all criteria hold), any (each criterion met by some rater), rater1..3.
  * Cross-reference matrix: calibrate the surgery-level certificate on the negatives of
    reference A, evaluate per-operation false confirmations against reference B.
  * Every-rater certificate = calibrate on "unanimous": a frame is a negative as soon as ONE
    surgeon rejects CVS, so P(some surgeon rejects a confirmed frame, in an operation) <= a,
    simultaneously for all three surgeons.
  * Monte Carlo rater (Stutz et al., TMLR 2023, adapted per operation): each calibration
    operation's labels come from one surgeon drawn at random; the guarantee is against a
    randomly drawn surgeon.  Evaluated against a random surgeon per test operation.
Also reports the share of confirmations on frames where the surgeons disagree.

    python scripts/review/annotators.py
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ALPHAS, endoscapes_ops, halves, op_metrics, order_stat, save, summary  # noqa: E402

REFS = ("majority", "unanimous", "any", "rater1", "rater2", "rater3")


def ref(op, kind):
    R = op["raters"]
    if kind == "majority":
        return (R.sum(1) >= 2).all(1)
    if kind == "unanimous":
        return R.all(1).all(1)
    if kind == "any":
        return R.any(1).all(1)
    return R[:, int(kind[-1]) - 1].astype(bool).all(1)


def lam_for(cal, labels, alpha, w):
    m = []
    for o, y in zip(cal, labels):
        s = o["p"][w][~y]
        if len(s):                                   # operations without unsafe frames cannot err
            m.append(s.max())
    return order_stat(m, alpha)


def job(args):
    seed, ops, rep, ci, ti = args
    cal, tst = [ops[i] for i in ci], [ops[i] for i in ti]
    rng = np.random.default_rng(10_000 + rep)
    rows = []
    for alpha in ALPHAS:
        for w in (1, 10):
            for A in REFS:
                lam = lam_for(cal, [ref(o, A) for o in cal], alpha, w)
                conf = [o["p"][w] > lam for o in tst]
                for B in REFS:
                    yb = [ref(o, B) for o in tst]
                    rows.append({"seed": seed, "rep": rep, "alpha": alpha, "w": w, "calib": A, "eval": B,
                                 **op_metrics(conf, [~y for y in yb], yb)})
                # against ANY surgeon: a confirmation is false if some surgeon rejects it
                yu = [ref(o, "unanimous") for o in tst]
                dis = [ref(o, "any") & ~ref(o, "unanimous") for o in tst]
                rows.append({"seed": seed, "rep": rep, "alpha": alpha, "w": w, "calib": A, "eval": "every surgeon",
                             **op_metrics(conf, [~y for y in yu], [ref(o, "majority") for o in tst]),
                             "share_conf_on_disagreement": sum((c & d).sum() for c, d in zip(conf, dis))
                             / max(sum(c.sum() for c in conf), 1)})
            # Monte Carlo rater per operation
            rc = [ref(o, f"rater{rng.integers(1, 4)}") for o in cal]
            lam = lam_for(cal, rc, alpha, w)
            conf = [o["p"][w] > lam for o in tst]
            rt = [ref(o, f"rater{rng.integers(1, 4)}") for o in tst]
            rows.append({"seed": seed, "rep": rep, "alpha": alpha, "w": w, "calib": "random rater",
                         "eval": "random rater", **op_metrics(conf, [~y for y in rt], rt)})
    return rows


def main():
    rows = []
    for seed in (0, 1, 2):
        ops = endoscapes_ops("dinov2l_b8", seed, (1, 10))
        with Pool(20) as pool:
            for r in pool.imap_unordered(job, [(seed, ops, rep, ci, ti) for rep, ci, ti in halves(len(ops))], chunksize=4):
                rows += r
        print(seed, "done", flush=True)
    df = save(rows, "annotators")
    summary(df[df.w == 1], ["alpha", "calib", "eval"], ["ops_false_ge1", "sens", "fdr", "share_conf_on_disagreement"])
    summary(df[(df.w == 10) & (df["eval"].isin(["every surgeon", "random rater", "majority"]))],
            ["alpha", "calib", "eval"], ["ops_false_ge1", "sens", "fdr"])


if __name__ == "__main__":
    main()
