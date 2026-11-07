"""Time-out certificate with a conditional STOP (R1 W6, R2.8).

Same operations, smoothing and splits as scripts/preclip.py (Endoscapes: 200 halvings;
Cholec80 recalibrated on 37 operations: 100 splits), 3 seeds.  Confirmation is unchanged
(class-conditional over calibration operations without CVS at the clip).  Two STOP rules:

  joint (paper v1)   alarm iff p < mu, mu = CRC threshold for 1[p < mu and CVS]:
                     P(alarm and CVS) <= a  -- may alarm on many operations that HAD CVS
  asymmetric (new)   confirm at a_c = 0.10, conditional STOP at a_a = 0.20: false reassurance is the
                     dangerous error, a false STOP costs time
  conditional (new)  alarm iff p < s_(r), the r-th smallest clip-time score among the V1
                     calibration operations WITH CVS at the clip, r = floor(a (V1 + 1)):
                     P(alarm | CVS at the clip) <= a  (exact, class-conditional);
                     if r = 0 (too few positive operations) the system never alarms: it defers.
Reports, per rule: P(confirm | no CVS), P(confirm | CVS), P(alarm | CVS), P(alarm | no CVS),
P(alarm and CVS), deferral, and the share of splits in which a STOP is possible at all.

    python scripts/review2/timeout_alarm.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cdss import riskcontrol as RC                                    # noqa: E402
from cdss.data import keyframes                                       # noqa: E402
from evaluate import causal_logits, fit_temperature                   # noqa: E402
from preclip import ALPHAS, SMOOTH, cholec80_ops, endoscapes_ops, order_stat  # noqa: E402


def thresholds(cal, alpha):
    p = np.array([o["p"][-1] for o in cal]); c = np.array([bool(o["cvs"][-1]) for o in cal])
    lam_c = order_stat(p[~c], alpha)
    grid = RC.threshold_grid(p)
    L = (p[:, None] < grid[None, :]) & c[:, None]
    g = RC.crc_index(L[:, ::-1], alpha)
    mu_joint = -math.inf if g is None else grid[::-1][g]
    pos = np.sort(p[c]); r = math.floor(alpha * (len(pos) + 1))
    mu_cond = -math.inf if r == 0 else float(pos[r - 1])
    return lam_c, mu_joint, mu_cond


def score(tst, lam_c, mu, rule, alpha, base):
    p = np.array([o["p"][-1] for o in tst]); c = np.array([bool(o["cvs"][-1]) for o in tst])
    conf = p > lam_c; alarm = (p < mu) & ~conf
    nc, pc = max((~c).sum(), 1), max(c.sum(), 1)
    return {**base, "alpha": alpha, "rule": rule,
            "confirm_given_noCVS": (conf & ~c).sum() / nc if (~c).any() else np.nan,
            "confirm_given_CVS": (conf & c).sum() / pc if c.any() else np.nan,   # splits without CVS left out, as in timeout_final
            "alarm_given_CVS": (alarm & c).sum() / pc if c.any() else np.nan,
            "alarm_given_noCVS": (alarm & ~c).sum() / nc, "alarm_and_CVS": float((alarm & c).mean()),
            "defer": float((~conf & ~alarm).mean()), "stop_possible": float(mu > -math.inf),
            "has_cvs_ops": float(c.any())}


def point_world(tst, base):
    r = score(tst, 0.5, 0.5, "point world", np.nan, base)
    return r


def main():
    va = keyframes("val")
    sel_vids = set(np.random.default_rng(0).choice(np.unique(va.vid), 10, replace=False).tolist())
    rows = []
    for seed in (0, 1, 2):
        run = ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{seed}"
        sel = np.isin(va.vid, list(sel_vids))
        T = fit_temperature(causal_logits(run / "logits_val.npz", va, 1)[sel], va.soft[sel])
        for s in SMOOTH:
            E = endoscapes_ops(seed, s, sel_vids); C = cholec80_ops(seed, s, T)
            for rep in range(200):
                rng = np.random.default_rng(rep)
                ci = np.sort(np.random.default_rng(rep).choice(len(E), len(E) // 2, replace=False))   # = common.halves
                cal = [E[i] for i in ci]; tst = [E[i] for i in np.setdiff1d(np.arange(len(E)), ci)]
                base = {"seed": seed, "smooth_s": s, "rep": rep, "dataset": "Endoscapes"}
                for a in ALPHAS:
                    lc, mj, mc = thresholds(cal, a)
                    rows += [score(tst, lc, mj, "joint", a, base), score(tst, lc, mc, "conditional", a, base)]
                lc, _, _ = thresholds(cal, 0.10); _, _, mc = thresholds(cal, 0.20)
                rows.append(score(tst, lc, mc, "asymmetric (confirm .10, STOP .20)", 0.10, base))
                rows.append(point_world(tst, base))
                if rep < 100:
                    ci = np.sort(np.random.default_rng(rep).choice(len(C), len(C) // 2, replace=False))
                    cal = [C[i] for i in ci]; tst = [C[i] for i in np.setdiff1d(np.arange(len(C)), ci)]
                    base = {"seed": seed, "smooth_s": s, "rep": rep, "dataset": "Cholec80 (recal.)"}
                    for a in ALPHAS:
                        lc, mj, mc = thresholds(cal, a)
                        rows += [score(tst, lc, mj, "joint", a, base), score(tst, lc, mc, "conditional", a, base)]
                    lc, _, _ = thresholds(cal, 0.10); _, _, mc = thresholds(cal, 0.20)
                    rows.append(score(tst, lc, mc, "asymmetric (confirm .10, STOP .20)", 0.10, base))
                    rows.append(point_world(tst, base))
        print("seed", seed, flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "experiments" / "review"; df.to_csv(out / "timeout_alarm.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.groupby(["dataset", "smooth_s", "alpha", "rule"])[
        ["confirm_given_noCVS", "confirm_given_CVS", "alarm_given_CVS", "alarm_given_noCVS", "alarm_and_CVS",
         "defer", "stop_possible"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
