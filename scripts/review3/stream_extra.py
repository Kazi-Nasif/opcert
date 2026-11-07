"""CVS stream during dissection: analyses requested in review round 3.

  (a) precision per operation (R-B W5/Q2): among test operations in which a method confirms CVS at
      least once, the share containing a false confirmation; frame-level precision; timing on
      operations with CVS before the clip, never-confirmed operations in the denominator
      (confirmed before the clip, delay from the first CVS rating, lead time before the clip)
  (b) calibration-size curve (R-A P1): V in {11, 15, 20, 25, 30, 35} calibration operations drawn
      from the calibration half, marginal and PAC (delta 0.1); finite-threshold rate, error, sensitivity
  (c) non-random splits (R-B W2/Q3): video-ID order (lower half calibrates the upper and reverse),
      official split and its reverse; exact intervals over the test operations
  (d) 1 fps deployment (R-B W4/Q4): every 1-fps frame of the dissection interval receives the label of
      the most recent rated keyframe ('hold') or is unsafe if either neighbouring keyframe is
      ('conservative'); keyframe-calibrated certificates evaluated on the 1-fps stream, and
      certificates calibrated on the 1-fps stream itself
Same 200 halvings x 3 seeds as Table 3, DINOv2-L.

    python scripts/review3/stream_extra.py
"""
from __future__ import annotations

import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as B

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "review"))
sys.path.insert(0, str(ROOT / "scripts" / "review2"))
from cdss.data import keyframes                                         # noqa: E402
from common import endoscapes_ops, halves, order_stat, temperature     # noqa: E402
from rebuttal import frame_crc, frame_ltt, mmax, pac_index             # noqa: E402

ALPHAS = (0.05, 0.10, 0.20)
WS = (1, 10)
VS = (11, 15, 20, 25, 30, 35)


def pac(m, a):
    m = np.sort(m); j = pac_index(len(m), a)
    return math.inf if j is None else float(m[j - 1])


def surg(cal, w, a, kind="marginal"):
    m = np.array([mmax(o, w) for o in cal if (~o["cvs"]).any()])
    return order_stat(m, a) if kind == "marginal" else pac(m, a)


def stream_stats(tst, w, lam):
    """Per-operation and timing statistics of one threshold on the test operations."""
    any_c, any_f, tp, fp, before, delay, lead, ncvs = 0, 0, 0, 0, 0, [], [], 0
    for o in tst:
        c = o["p"][w] > lam; y = o["cvs"]; t = o["frame"] / 25.0
        any_c += c.any(); any_f += (c & ~y).any(); tp += (c & y).sum(); fp += (c & ~y).sum()
        if y.any():
            ncvs += 1
            hit = np.flatnonzero(c & y)
            if len(hit):
                before += 1; delay.append(t[hit[0]] - t[y][0]); lead.append(t[-1] - t[hit[0]])
    n = len(tst)
    return {"ops_err": any_f / n, "ops_any_conf": any_c / n,
            "ops_err_given_conf": any_f / any_c if any_c else np.nan, "frame_precision": tp / (tp + fp) if tp + fp else np.nan,
            "sens": tp / max(sum(o["cvs"].sum() for o in tst), 1),
            "cvs_ops_confirmed": before / ncvs if ncvs else np.nan,
            "delay_med": float(np.median(delay)) if delay else np.nan, "lead_med": float(np.median(lead)) if lead else np.nan}


def job(args):
    seed, ops, rep, ci, ti = args
    cal, tst = [ops[i] for i in ci], [ops[i] for i in ti]
    rows = []
    base = {"seed": seed, "rep": rep}
    for w in WS:
        for a in ALPHAS:
            for name, lam in (("ours", surg(cal, w, a)), ("ours PAC", surg(cal, w, a, "pac")),
                              ("CRC frame-level", frame_crc(cal, w, a)), ("LTT frame-level", frame_ltt(cal, w, a))):
                rows.append({**base, "part": "a", "w": w, "alpha": a, "method": name, **stream_stats(tst, w, lam)})
    rng = np.random.default_rng(50_000 + rep)
    for V in VS:
        sub = [cal[i] for i in np.sort(rng.choice(len(cal), V, replace=False))]
        for a in ALPHAS:
            for kind in ("marginal", "pac"):
                lam = surg(sub, 10, a, kind)
                rows.append({**base, "part": "b", "w": 10, "alpha": a, "V": V, "method": f"ours {kind}",
                             "finite": float(math.isfinite(lam)), **stream_stats(tst, 10, lam)})
    return rows


def fixed_splits(seed, ops):
    te = set(keyframes("test").vid.tolist())
    vids = sorted(o["vid"] for o in ops); h = len(vids) // 2
    lo = set(vids[:h])
    defs = {"official (val->test)": lambda o: o["vid"] not in te, "official reversed": lambda o: o["vid"] in te,
            "video ID: lower->upper": lambda o: o["vid"] in lo, "video ID: upper->lower": lambda o: o["vid"] not in lo}
    rows = []
    for name, is_cal in defs.items():
        cal = [o for o in ops if is_cal(o)]; tst = [o for o in ops if not is_cal(o)]
        for w in WS:
            for a in ALPHAS:
                for mname, lam in (("ours", surg(cal, w, a)), ("ours PAC", surg(cal, w, a, "pac")),
                                   ("CRC frame-level", frame_crc(cal, w, a)), ("LTT frame-level", frame_ltt(cal, w, a))):
                    bad = np.array([((o["p"][w] > lam) & ~o["cvs"]).any() for o in tst]); k, n = int(bad.sum()), len(bad)
                    rows.append({"seed": seed, "split": name, "V": len(cal), "n_test": n, "w": w, "alpha": a, "method": mname,
                                 "k_err": k, "ops_err": k / n, "cp_hi": 1.0 if k == n else B.ppf(0.975, k + 1, n - k),
                                 "sens": stream_stats(tst, w, lam)["sens"]})
    return rows


def onefps_ops(seed, w, T):
    """1-fps stream of the dissection interval of every pool operation with propagated labels."""
    run = ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{seed}"
    sv = set(np.random.default_rng(0).choice(np.unique(keyframes("val").vid), 10, replace=False).tolist())
    ops = []
    for split in ("val", "test"):
        z = np.load(run / f"logits_{split}.npz"); kf = keyframes(split)
        cvs = (kf.raters.sum(1) >= 2).all(1)
        for v in np.unique(kf.vid):
            if int(v) in sv:
                continue
            km = np.flatnonzero(kf.vid == v); km = km[np.argsort(kf.frame[km])]
            kfr, ky = kf.frame[km], cvs[km]
            r = np.flatnonzero(z["vid"] == v); r = r[np.argsort(z["frame"][r])]
            fr, L = z["frame"][r], z["logits"][r].astype(float)
            Ls = np.stack([L[max(0, i - w + 1):i + 1].mean(0) for i in range(len(L))])
            p = np.prod(1 / (1 + np.exp(-Ls / T)), 1)
            keep = (fr >= kfr[0]) & (fr <= kfr[-1])
            fr, p = fr[keep], p[keep]
            prev = np.searchsorted(kfr, fr, side="right") - 1
            nxt = np.minimum(np.searchsorted(kfr, fr, side="left"), len(kfr) - 1)
            hold = ky[prev]; cons = ky[prev] & ky[nxt]
            iskf = np.isin(fr, kfr)
            ops.append({"vid": int(v), "p": p, "hold": hold, "cons": cons, "iskf": iskf})
    return ops


def onefps(seed):
    T = temperature(ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{seed}")
    rows = []
    for w in WS:
        ops = onefps_ops(seed, w, T)
        for rep, ci, ti in halves(len(ops)):
            cal, tst = [ops[i] for i in ci], [ops[i] for i in ti]
            for a in ALPHAS:
                for lab in ("hold", "cons"):
                    m_kf = [o["p"][o["iskf"] & ~o[lab]].max() for o in cal if (o["iskf"] & ~o[lab]).any()]
                    m_1f = [o["p"][~o[lab]].max() for o in cal if (~o[lab]).any()]
                    for name, lam in (("ours, calibrated on keyframes", order_stat(m_kf, a)),
                                      ("ours, calibrated at 1 fps", order_stat(m_1f, a)),
                                      ("CRC frame-level at 1 fps", order_stat(np.concatenate([o["p"][~o[lab]] for o in cal]), a))):
                        err = np.mean([((o["p"] > lam) & ~o[lab]).any() for o in tst])
                        err_kf = np.mean([((o["p"] > lam) & ~o[lab] & o["iskf"]).any() for o in tst])
                        tp = sum(((o["p"] > lam) & o[lab]).sum() for o in tst); npos = sum(o[lab].sum() for o in tst)
                        rows.append({"seed": seed, "rep": rep, "w": w, "alpha": a, "label": lab, "method": name,
                                     "ops_err_1fps": err, "ops_err_keyframes": err_kf, "sens_1fps": tp / max(npos, 1)})
    return rows


def main():
    rows, fx, of = [], [], []
    for seed in (0, 1, 2):
        ops = endoscapes_ops("dinov2l_b8", seed, WS)
        with Pool(20) as pool:
            for r in pool.imap_unordered(job, [(seed, ops, rep, ci, ti) for rep, ci, ti in halves(len(ops))], chunksize=4):
                rows += r
        fx += fixed_splits(seed, ops)
        of += onefps(seed)
        print("seed", seed, flush=True)
    out = ROOT / "experiments" / "review3"; out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(out / "stream_extra.csv", index=False)
    fx = pd.DataFrame(fx); fx.to_csv(out / "fixed_splits.csv", index=False)
    of = pd.DataFrame(of); of.to_csv(out / "onefps.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400)
    print(df[df.part == "a"].groupby(["w", "alpha", "method"])[
        ["ops_err", "ops_any_conf", "ops_err_given_conf", "frame_precision", "sens", "cvs_ops_confirmed", "delay_med", "lead_med"]].mean().round(3).to_string())
    print(df[df.part == "b"].groupby(["alpha", "method", "V"])[["finite", "ops_err", "sens"]].mean().round(3).to_string())
    print(fx.groupby(["split", "w", "alpha", "method"])[["V", "n_test", "k_err", "ops_err", "cp_hi", "sens"]].mean().round(3).to_string())
    print(of.groupby(["w", "label", "alpha", "method"])[["ops_err_1fps", "ops_err_keyframes", "sens_1fps"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
