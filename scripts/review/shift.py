"""Review: transfer Endoscapes -> Cholec80 without full recalibration.

Operation-level covariates phi_v (label-free, from the CVS model's outputs on the
operation's frames sampled every 5 s): mean and s.d. of the three criterion probabilities,
mean P(CVS), fraction of frames with P(CVS) > 0.1 / > 0.5, log number of frames.

  As-is                 Endoscapes calibration (71 operations) deployed unchanged.
  Weighted (ours)       Weighted conformal under covariate shift (Tibshirani et al. 2019) at
                        the operation level: w(phi) = P(Cholec80 | phi) / P(Endoscapes | phi)
                        from a logistic domain classifier trained on the 71 Endoscapes
                        operations and the UNLABELLED Cholec80 operations of the adaptation
                        half; threshold = weighted (1 - a) quantile of {m_v} with the test
                        operation's weight on +inf.  Valid if only phi shifts (P(labels | phi)
                        fixed) -- a different annotation protocol violates exactly this.
  Few-shot recal.       n in {5, 10, 20} labelled Cholec80 operations from the adaptation half.
  Pooled                71 Endoscapes + n Cholec80 operations, unweighted / weighted.
Evaluated on the other half (37 operations), 100 random halvings x 3 seeds, for the streaming
surgery-level certificate (1 s and 10 s smoothing) and the time-out confirmation.

    python scripts/review/shift.py
"""
from __future__ import annotations

import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ALPHAS, ROOT, endoscapes_ops, order_stat, save, summary, temperature  # noqa: E402

NS = (5, 10, 20)


def cholec80_ops(seed, T, windows=(1, 10)):
    run = ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{seed}"
    df = pd.read_csv(ROOT / "data" / "cholec80" / "cholec80_cvs_frames.csv")
    z = np.load(run / "logits_cholec80.npz")
    ops = []
    for v in df.video.unique():
        m = np.flatnonzero(df.video.to_numpy() == v); o = m[np.argsort(df.frame.to_numpy()[m])]
        L = z["logits"][o]
        P = {}
        for w in windows:
            Ls = np.stack([L[max(0, i - w + 1):i + 1].mean(0) for i in range(len(L))])
            P[w] = np.prod(1 / (1 + np.exp(-Ls / T)), 1)
        ops.append({"vid": int(v), "p": P, "pc": 1 / (1 + np.exp(-L / T)),
                    "cvs": df.cvs_authors.to_numpy()[o].astype(bool), "step": 5})
    return ops


def phi(o):
    step = o.get("step", 1)
    pc, p = o["pc"][::step], o["p"][1][::step]
    return np.concatenate([pc.mean(0), pc.std(0), [p.mean(), (p > 0.1).mean(), (p > 0.5).mean(), math.log(len(p))]])


def mstat(o, w, timeout=False):
    if timeout:
        return o["p"][w][-1] if not o["cvs"][-1] else None       # only operations without CVS at the clip
    s = o["p"][w][~o["cvs"]]
    return s.max() if len(s) else -np.inf


def weighted_threshold(m, wcal, wtest, alpha):
    order = np.argsort(m); m, wcal = np.asarray(m)[order], np.asarray(wcal)[order]
    tot = wcal.sum() + wtest
    cum = np.cumsum(wcal) / tot
    ok = np.flatnonzero(cum >= 1 - alpha)
    return m[ok[0]] if len(ok) else np.inf


def evaluate(tst, lam_per_op, w, timeout):
    if timeout:
        neg = [o for o, l in zip(tst, lam_per_op) if not o["cvs"][-1]]
        lneg = [l for o, l in zip(tst, lam_per_op) if not o["cvs"][-1]]
        pos = [(o, l) for o, l in zip(tst, lam_per_op) if o["cvs"][-1]]
        return {"ops_false": float(np.mean([o["p"][w][-1] > l for o, l in zip(neg, lneg)])) if neg else np.nan,
                "sens": float(np.mean([o["p"][w][-1] > l for o, l in pos])) if pos else np.nan}
    bad = [((o["p"][w] > l) & ~o["cvs"]).any() for o, l in zip(tst, lam_per_op)]
    tp = sum(((o["p"][w] > l) & o["cvs"]).sum() for o, l in zip(tst, lam_per_op))
    return {"ops_false": float(np.mean(bad)), "sens": tp / max(sum(o["cvs"].sum() for o in tst), 1)}


def job(args):
    seed, E, C, rep = args
    rng = np.random.default_rng(rep)
    idx = rng.permutation(len(C)); ad, ev = [C[i] for i in idx[:37]], [C[i] for i in idx[37:]]
    X = np.stack([phi(o) for o in E + ad]); y = np.r_[np.zeros(len(E)), np.ones(len(ad))]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)
    def wt(o):
        q = clf.predict_proba(sc.transform(phi(o)[None]))[0, 1]
        return q / max(1 - q, 1e-6)
    wE = np.array([wt(o) for o in E]); wA = np.array([wt(o) for o in ad]); wT = np.array([wt(o) for o in ev])
    rows = []
    for timeout in (False, True):
        for w in (1, 10):
            def ms(ops, ws=None):
                out = [(mstat(o, w, timeout), (ws[i] if ws is not None else 1.0)) for i, o in enumerate(ops)]
                return [a for a, _ in out if a is not None], [b for a, b in out if a is not None]
            mE, wEu = ms(E, wE)
            for alpha in ALPHAS:
                base = {"seed": seed, "rep": rep, "alpha": alpha, "w": w, "decision": "time-out" if timeout else "stream"}
                lam = order_stat(mE, alpha)
                rows.append({**base, "method": "As-is (Endoscapes)", "n": 0, **evaluate(ev, [lam] * len(ev), w, timeout)})
                lams = [weighted_threshold(mE, wEu, wt_, alpha) for wt_ in wT]
                rows.append({**base, "method": "Weighted (covariate shift)", "n": 0, **evaluate(ev, lams, w, timeout)})
                for n in NS:
                    lab = ad[:n]
                    mL, wL = ms(lab, wA[:n])
                    rows.append({**base, "method": "Few-shot recalibration", "n": n,
                                 **evaluate(ev, [order_stat(mL, alpha)] * len(ev), w, timeout)})
                    rows.append({**base, "method": "Pooled (unweighted)", "n": n,
                                 **evaluate(ev, [order_stat(mE + mL, alpha)] * len(ev), w, timeout)})
                    lams = [weighted_threshold(mE + mL, wEu + [1.0] * len(mL) if False else list(wEu) + list(wL), wt_, alpha)
                            for wt_ in wT]
                    rows.append({**base, "method": "Pooled + weighted", "n": n, **evaluate(ev, lams, w, timeout)})
                # full recalibration reference (all 37 adaptation operations labelled)
                mA, _ = ms(ad)
                rows.append({**base, "method": "Full recalibration (37)", "n": 37,
                             **evaluate(ev, [order_stat(mA, alpha)] * len(ev), w, timeout)})
    return rows


def main():
    rows = []
    for seed in (0, 1, 2):
        T = temperature(ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{seed}")
        E = endoscapes_ops("dinov2l_b8", seed, (1, 10)); C = cholec80_ops(seed, T)
        with Pool(20) as pool:
            for r in pool.imap_unordered(job, [(seed, E, C, rep) for rep in range(100)], chunksize=2):
                rows += r
        print(seed, "done", flush=True)
    df = save(rows, "shift")
    summary(df, ["decision", "w", "alpha", "method", "n"], ["ops_false", "sens"])


if __name__ == "__main__":
    main()
