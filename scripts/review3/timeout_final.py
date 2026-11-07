"""Time-out analysis of the FINAL policy (review round 3: R-A T3/T4/P0, R-B W3/W5, Q1/Q2/Q5).

One decision per operation at the last scored frame before the first clip, smoothed score p.
Same video-level halvings as Table 3 (scripts/review/common.halves) for Endoscapes; Cholec80
recalibrated on half of its 74 operations (100 halvings).  DINOv2-L, 3 seeds.

Policies (all on the same scores, splits and reference):
  ours         CONFIRM if p > lam_c (class-conditional over calibration ops WITHOUT CVS at the clip,
               alpha_c); else STOP if p < s_(r), r = floor(alpha_s (V1+1)) over ops WITH CVS (r = 0:
               never stop); else DEFER.  alpha_s = 0.20, alpha_c in {0.05, 0.10, 0.20}.
  frame CRC    pooled frame-level CRC threshold (all unsafe scored frames of the calibration ops)
               applied at the time-out frame; confirm-or-defer (R-B Q1)
  frame LTT    same with Learn-then-Test (delta 0.1)
  point world  confirm iff p > 0.5, else STOP
  conf. sel.   conformal selection (Jin & Candes) over the test batch: conformal p-values against the
               calibration ops without CVS, Benjamini-Hochberg at q: bounds the FDR of confirmations
Per split: counts and rates by reference class, precision of CONFIRM and of STOP, deferral.
Reference robustness (R-A P1): Endoscapes majority / unanimous ('every surgeon' accepts) /
per-surgeon evaluation of a majority-calibrated policy; Cholec80 dataset (total >= 5) / strict.
Fixed official split (31 -> 40) with exact intervals (R-A 'avoid inflated precision').

    python scripts/review3/timeout_final.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as B

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "review"))
from cdss import riskcontrol as RC                                   # noqa: E402
from cdss.data import keyframes                                      # noqa: E402
from common import endoscapes_ops, halves, order_stat, temperature  # noqa: E402

SMOOTH = (1, 5, 10)
ALPHA_C = (0.05, 0.10, 0.20)
ALPHA_S = 0.20
DELTA = 0.1


def cholec80_ops(seed, w, T):
    run = ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{seed}"
    df = pd.read_csv(ROOT / "data" / "cholec80" / "cholec80_cvs_frames.csv")
    z = np.load(run / "logits_cholec80.npz")
    ops = []
    for v in df.video.unique():
        m = np.flatnonzero(df.video.to_numpy() == v); o = m[np.argsort(df.frame.to_numpy()[m])]
        L = z["logits"][o]
        Ls = np.stack([L[max(0, i - w + 1):i + 1].mean(0) for i in range(len(L))])
        ops.append({"vid": int(v), "p": np.prod(1 / (1 + np.exp(-Ls / T)), 1), "t": df.frame.to_numpy()[o] / 25.0,
                    "ref": {"dataset": df.cvs_authors.to_numpy()[o].astype(bool),
                            "strict": df.cvs_strict.to_numpy()[o].astype(bool)}})
    return ops


def endo_ops(seed, w):
    ops = endoscapes_ops("dinov2l_b8", seed, (w,))
    out = []
    for o in ops:
        R = o["raters"]                                    # (n, 3 raters, 3 criteria)
        ref = {"majority": o["cvs"], "unanimous": R.all(1).all(1)}
        for k in range(3):
            ref[f"surgeon{k + 1}"] = R[:, k].astype(bool).all(1)
        out.append({"vid": o["vid"], "p": o["p"][w], "t": o["frame"] / 25.0, "ref": ref})
    return out


def frame_thresholds(cal, ref, a):
    neg = np.concatenate([o["p"][~o["ref"][ref]] for o in cal])
    lam_crc = order_stat(neg, a)
    p = np.concatenate([o["p"] for o in cal]); y = np.concatenate([o["ref"][ref] for o in cal])
    grid = RC.threshold_grid(p)
    L = ((p[:, None] > grid[None]) & ~y[:, None])[~y].astype(float)
    g = RC.ltt_index(L, a, DELTA)
    return lam_crc, (math.inf if g is None else grid[g])


def bh_select(cal_neg, s_test, q):
    n0 = len(cal_neg)
    pv = np.array([(1 + (cal_neg >= s).sum()) / (n0 + 1) for s in s_test])
    order = np.argsort(pv); m = len(pv)
    ok = pv[order] <= q * np.arange(1, m + 1) / m
    k = np.flatnonzero(ok).max() + 1 if ok.any() else 0
    sel = np.zeros(m, bool); sel[order[:k]] = True
    return sel


def decide(p, lam_c, mu):
    conf = p > lam_c
    stop = (p < mu) & ~conf
    return conf, stop


def metrics(conf, stop, b):
    nc, n1 = int((~b).sum()), int(b.sum())
    tp, fp = int((conf & b).sum()), int((conf & ~b).sum())
    sf, st = int((stop & b).sum()), int((stop & ~b).sum())
    return {"n_ops": len(b), "n_cvs": n1, "n_conf": tp + fp, "tp": tp, "fp": fp, "n_stop": sf + st, "stop_on_cvs": sf,
            "conf_noCVS": fp / nc if nc else np.nan, "conf_CVS": tp / n1 if n1 else np.nan,
            "stop_CVS": sf / n1 if n1 else np.nan, "stop_noCVS": st / nc if nc else np.nan,
            "defer": float((~conf & ~stop).mean()),
            "precision": tp / (tp + fp) if tp + fp else np.nan, "fdp": fp / (tp + fp) if tp + fp else 0.0,
            "stop_precision": st / (sf + st) if sf + st else np.nan, "any_conf": float(tp + fp > 0)}


def policies(cal, tst, cref, a_c):
    """Yield (name, conf, stop) for every policy; calibration on reference cref."""
    pc = np.array([o["p"][-1] for o in cal]); bc = np.array([bool(o["ref"][cref][-1]) for o in cal])
    pt = np.array([o["p"][-1] for o in tst])
    lam_c = order_stat(pc[~bc], a_c)
    pos = np.sort(pc[bc]); r = math.floor(ALPHA_S * (len(pos) + 1))
    mu = -math.inf if r == 0 else float(pos[r - 1])
    yield "ours", *decide(pt, lam_c, mu), float(r > 0)
    lam_crc, lam_ltt = frame_thresholds(cal, cref, a_c)
    yield "frame CRC", *decide(pt, lam_crc, -math.inf), np.nan
    yield "frame LTT", *decide(pt, lam_ltt, -math.inf), np.nan
    yield "conf. selection", bh_select(pc[~bc], pt, a_c), np.zeros(len(pt), bool), np.nan


def run_split(cal, tst, refs, base):
    rows = []
    pt = np.array([o["p"][-1] for o in tst])
    for eref in refs:                                           # point world, per evaluation reference
        b = np.array([bool(o["ref"][eref][-1]) for o in tst])
        rows.append({**base, "policy": "point world", "alpha_c": np.nan, "calib": "-", "eval": eref,
                     **metrics(pt > 0.5, pt <= 0.5, b)})
    for cref in refs[:2]:
        for a_c in ALPHA_C:
            for name, conf, stop, stop_possible in policies(cal, tst, cref, a_c):
                for eref in refs:
                    if cref != refs[0] and eref != cref:
                        continue
                    b = np.array([bool(o["ref"][eref][-1]) for o in tst])
                    rows.append({**base, "policy": name, "alpha_c": a_c, "calib": cref, "eval": eref,
                                 "stop_possible": stop_possible, **metrics(conf, stop, b)})
    return rows


def main():
    rows, fixed = [], []
    te = set(keyframes("test").vid.tolist())
    for seed in (0, 1, 2):
        T = temperature(ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{seed}")
        for w in SMOOTH:
            E = endo_ops(seed, w)
            refs = ["majority", "unanimous", "surgeon1", "surgeon2", "surgeon3"]
            for rep, ci, ti in halves(len(E)):
                rows += run_split([E[i] for i in ci], [E[i] for i in ti], refs,
                                  {"dataset": "Endoscapes", "seed": seed, "smooth_s": w, "rep": rep})
            cal = [o for o in E if o["vid"] not in te]; tst = [o for o in E if o["vid"] in te]
            fixed += run_split(cal, tst, refs, {"dataset": "Endoscapes official", "seed": seed, "smooth_s": w, "rep": -1})
            C = cholec80_ops(seed, w, T)
            for rep, ci, ti in halves(len(C), 100):
                rows += run_split([C[i] for i in ci], [C[i] for i in ti], ["dataset", "strict"],
                                  {"dataset": "Cholec80 (recal.)", "seed": seed, "smooth_s": w, "rep": rep})
            print(seed, w, flush=True)
    df = pd.DataFrame(rows); fx = pd.DataFrame(fixed)
    out = ROOT / "experiments" / "review3"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "timeout_final.csv", index=False); fx.to_csv(out / "timeout_fixed.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400)
    d = df[df.smooth_s == 5]
    agg = d.groupby(["dataset", "calib", "eval", "policy", "alpha_c"], dropna=False).agg(
        conf_noCVS=("conf_noCVS", "mean"), conf_CVS=("conf_CVS", "mean"), stop_CVS=("stop_CVS", "mean"),
        stop_noCVS=("stop_noCVS", "mean"), defer=("defer", "mean"), fdr=("fdp", "mean"),
        prec_mean=("precision", "mean"), any_conf=("any_conf", "mean"), stop_prec=("stop_precision", "mean"),
        tp=("tp", "sum"), fp=("fp", "sum"), n_cvs=("n_cvs", "mean"), n_ops=("n_ops", "mean"))
    agg["prec_pooled"] = agg.tp / (agg.tp + agg.fp)
    print(agg.round(3).to_string())
    f = fx[(fx.smooth_s == 5) & (fx["eval"] == "majority") & (fx.calib.isin(["majority", "-"]))]
    g = f.groupby(["policy", "alpha_c"], dropna=False)[["n_ops", "n_cvs", "tp", "fp", "n_stop", "stop_on_cvs"]].sum()
    n0 = g.n_ops - g.n_cvs
    g["conf_noCVS"] = g.fp / n0
    g["cp_hi"] = [1.0 if k == n else B.ppf(0.975, k + 1, n - k) for k, n in zip(g.fp, n0)]
    g["precision"] = g.tp / (g.tp + g.fp)
    print("fixed official split, 3 seeds summed (40 test ops each):\n", g.round(3).to_string())


if __name__ == "__main__":
    main()
