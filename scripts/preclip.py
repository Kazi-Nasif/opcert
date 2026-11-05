"""Clip-time ("time-out") certificates: the decision that matters is made just before the
first clip. One decision per operation -> operations are the exchangeable units and every
guarantee below is exact under exchangeability of operations.

Per operation, at the clip-time frame (Endoscapes: last CVS keyframe of the region of
interest, which ends at the first clip/cut; Cholec80: last frame of the calot-triangle-
dissection phase), with P(CVS) averaged causally over the last s seconds (1 fps):
  CONFIRM  'CVS achieved, clipping supported'  bound  P(confirm | no CVS at clip time) <= a
           (class-conditional, calibrated on operations WITHOUT CVS at clip time)
  ALARM    'CVS NOT achieved, stop'            bound  P(alarm and CVS at clip time) <= a
           (conformal risk control over operations; class-conditional would need >= 1/a - 1
            calibration operations WITH CVS, which few datasets have)
  DEFER    otherwise
Window certificate: P(any false confirmation in the last W s before the clip) <= a
           (max over the window's non-CVS frames, one score per operation).
Point world: confirm iff the smoothed P(CVS) > 0.5, alarm otherwise.

    python scripts/preclip.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cdss import riskcontrol as RC                    # noqa: E402
from cdss.data import keyframes                       # noqa: E402
from evaluate import causal_logits, fit_temperature   # noqa: E402

ALPHAS = (0.05, 0.10, 0.20)
SMOOTH = (1, 5, 10)
WINDOW_S = 30


def order_stat(m, alpha):
    m = np.sort(np.asarray(m, float)); k = math.ceil((len(m) + 1) * (1 - alpha))
    return math.inf if len(m) == 0 or k > len(m) else float(m[k - 1])


def endoscapes_ops(seed, smooth, sel_vids):
    """[{'vid', 'p': (n,) smoothed P(CVS) on CVS keyframes in time order, 'cvs': (n,), 't': (n,) seconds}]"""
    run = ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{seed}"
    va, te = keyframes("val"), keyframes("test")
    sel = np.isin(va.vid, sel_vids)
    T = fit_temperature(causal_logits(run / "logits_val.npz", va, 1)[sel], va.soft[sel])
    ops = []
    for split, kf in (("val", va), ("test", te)):
        L = causal_logits(run / f"logits_{split}.npz", kf, smooth)
        p = np.prod(1 / (1 + np.exp(-L / T)), 1)
        cvs = kf.cvs("majority")
        for v in np.unique(kf.vid):
            if v in sel_vids:
                continue
            m = kf.vid == v; o = np.argsort(kf.frame[m])
            ops.append({"vid": int(v), "p": p[m][o], "cvs": cvs[m][o], "t": kf.frame[m][o] / 25.0})
    return ops


def cholec80_ops(seed, smooth, T):
    run = ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{seed}"
    df = pd.read_csv(ROOT / "data" / "cholec80" / "cholec80_cvs_frames.csv")
    z = np.load(run / "logits_cholec80.npz")
    ops = []
    for v in df.video.unique():
        m = np.flatnonzero(df.video.to_numpy() == v); o = m[np.argsort(df.frame.to_numpy()[m])]
        L = z["logits"][o]
        Ls = np.stack([L[max(0, i - smooth + 1):i + 1].mean(0) for i in range(len(L))])
        ops.append({"vid": int(v), "p": np.prod(1 / (1 + np.exp(-Ls / T)), 1),
                    "cvs": df.cvs_authors.to_numpy()[o], "t": df.frame.to_numpy()[o] / 25.0})
    return ops


def certify(cal, alpha):
    last_p = np.array([o["p"][-1] for o in cal]); last_c = np.array([o["cvs"][-1] for o in cal])
    lam_conf = order_stat(last_p[~last_c], alpha)                      # exact, class-conditional
    grid = RC.threshold_grid(last_p)                                    # alarm iff p < mu
    L = (last_p[:, None] < grid[None, :]) & last_c[:, None]             # non-decreasing in mu
    g = RC.crc_index(L[:, ::-1], alpha)                                 # reverse: risk non-increasing
    mu = -math.inf if g is None else grid[::-1][g]
    win = [o["p"][(o["t"] >= o["t"][-1] - WINDOW_S) & ~o["cvs"]] for o in cal]
    lam_win = order_stat([w.max() if len(w) else -math.inf for w in win], alpha)
    return lam_conf, mu, lam_win


def score(ops, lam_conf, mu, lam_win, name):
    rows = []
    for o in ops:
        p, c = o["p"][-1], bool(o["cvs"][-1])
        conf, alarm = p > lam_conf, p < mu
        w = o["t"] >= o["t"][-1] - WINDOW_S
        rows.append({"cvs_at_clip": c, "confirm": conf, "alarm": alarm and not conf,
                     "win_false_confirm": bool(((o["p"] > lam_win) & ~o["cvs"] & w).any()),
                     "win_true_confirm_frames": int(((o["p"] > lam_win) & o["cvs"] & w).sum()),
                     "win_cvs_frames": int((o["cvs"] & w).sum())})
    d = pd.DataFrame(rows); c = d.cvs_at_clip
    return {"method": name,
            "false_confirm_given_noCVS": float((d.confirm & ~c).sum() / max((~c).sum(), 1)),
            "confirm_given_CVS": float((d.confirm & c).sum() / max(c.sum(), 1)),
            "alarm_given_noCVS": float((d.alarm & ~c).sum() / max((~c).sum(), 1)),
            "false_alarm_joint": float((d.alarm & c).mean()),
            "false_alarm_given_CVS": float((d.alarm & c).sum() / max(c.sum(), 1)),
            "defer": float((~d.confirm & ~d.alarm).mean()),
            "window_ops_false_confirm": float(d.win_false_confirm.mean()),
            "window_sensitivity": float(d.win_true_confirm_frames.sum() / max(d.win_cvs_frames.sum(), 1)),
            "cvs_at_clip_rate": float(c.mean())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--splits", type=int, default=200)
    a = ap.parse_args()
    va = keyframes("val")
    sel_vids = set(np.random.default_rng(0).choice(np.unique(va.vid), 10, replace=False).tolist())
    rows = []
    for seed in a.seeds:
        run = ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{seed}"
        sel = np.isin(va.vid, list(sel_vids))
        T = fit_temperature(causal_logits(run / "logits_val.npz", va, 1)[sel], va.soft[sel])
        for s in SMOOTH:
            E = endoscapes_ops(seed, s, sel_vids)
            C = cholec80_ops(seed, s, T)
            for rep in range(a.splits):
                rng = np.random.default_rng(rep)
                idx = rng.permutation(len(E)); cal = [E[i] for i in idx[:len(E) // 2]]; tst = [E[i] for i in idx[len(E) // 2:]]
                base = {"seed": seed, "smooth_s": s, "rep": rep}
                rows.append({**base, "dataset": "Endoscapes", "alpha": None, **score(tst, 0.5, 0.5, 0.5, "Point world")})
                for alpha in ALPHAS:
                    rows.append({**base, "dataset": "Endoscapes", "alpha": alpha,
                                 **score(tst, *certify(cal, alpha), "time-out certificate")})
                if rep < 100:                                   # Cholec80: recalibrate on 37 videos
                    idx = rng.permutation(len(C)); cal = [C[i] for i in idx[:37]]; tst = [C[i] for i in idx[37:]]
                    rows.append({**base, "dataset": "Cholec80 (recal.)", "alpha": None, **score(tst, 0.5, 0.5, 0.5, "Point world")})
                    for alpha in ALPHAS:
                        rows.append({**base, "dataset": "Cholec80 (recal.)", "alpha": alpha,
                                     **score(tst, *certify(cal, alpha), "time-out certificate")})
            for alpha in ALPHAS:                                  # Cholec80 as-is: calibrate on all Endoscapes ops
                rows.append({"seed": seed, "smooth_s": s, "rep": -1, "dataset": "Cholec80 (as-is)", "alpha": alpha,
                             **score(C, *certify(E, alpha), "time-out certificate")})
            rows.append({"seed": seed, "smooth_s": s, "rep": -1, "dataset": "Cholec80 (as-is)", "alpha": None,
                         **score(C, 0.5, 0.5, 0.5, "Point world")})
            print(f"seed {seed} smooth {s} done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "experiments" / "preclip"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "preclip.csv", index=False)
    df["alpha"] = df.alpha.fillna(-1)
    cols = ["cvs_at_clip_rate", "false_confirm_given_noCVS", "confirm_given_CVS", "alarm_given_noCVS",
            "false_alarm_joint", "false_alarm_given_CVS", "defer", "window_ops_false_confirm", "window_sensitivity"]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(df.groupby(["dataset", "smooth_s", "alpha", "method"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
