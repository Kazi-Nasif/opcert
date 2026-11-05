"""Timeline of one test operation: P(CVS) over time (1 fps, causal 10 s smoothing), the
surgeons' CVS label on keyframes, the frame-level (CRC) and surgery-level thresholds at
alpha = 0.05 (both calibrated on the 31 calibration videos of the official split), and the
first clip. Picks, automatically, a test operation where the frame-level rule falsely
confirms CVS at least once and CVS is truly achieved later.

    python scripts/figures/fig_timeline.py
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cdss import decide                               # noqa: E402
from cdss.data import keyframes                       # noqa: E402
from evaluate import causal_logits, fit_temperature   # noqa: E402

OUT = ROOT / "paper" / "figures"
ALPHA, SMOOTH = 0.05, 10
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.fontsize": 6.8})


def main():
    run = ROOT / "experiments" / "finetune" / "dinov2l_b8_s0"
    va, te = keyframes("val"), keyframes("test")
    sel = np.isin(va.vid, np.random.default_rng(0).choice(np.unique(va.vid), 10, replace=False))
    T = fit_temperature(causal_logits(run / "logits_val.npz", va, 1)[sel], va.soft[sel])
    pc = np.prod(1 / (1 + np.exp(-causal_logits(run / "logits_val.npz", va, SMOOTH)[~sel] / T)), 1)
    yc = va.cvs("majority")[~sel]; vc = va.vid[~sel]
    v0 = np.sort(pc[~yc]); lam_frame = v0[math.ceil((len(v0) + 1) * (1 - ALPHA)) - 1]
    m = np.sort([pc[(vc == v) & ~yc].max() for v in np.unique(vc) if ((vc == v) & ~yc).any()])
    k = math.ceil((len(m) + 1) * (1 - ALPHA)); lam_op = m[k - 1] if k <= len(m) else np.inf

    # all 1-fps frames of the test videos, smoothed causally
    z = np.load(run / "logits_test.npz")
    yt = te.cvs("majority")
    best = None
    for v in np.unique(te.vid):
        rows = np.flatnonzero(z["vid"] == v); rows = rows[np.argsort(z["frame"][rows])]
        L = z["logits"][rows]
        Ls = np.stack([L[max(0, i - SMOOTH + 1):i + 1].mean(0) for i in range(len(L))])
        p = np.prod(1 / (1 + np.exp(-Ls / T)), 1); t = z["frame"][rows] / 25.0
        km = te.vid == v; kt = te.frame[km] / 25.0; ky = yt[km]
        # keyframe-level false confirmations under each rule
        kp = np.interp(kt, t, p)
        ff = (kp > lam_frame) & ~ky; fo = (kp > lam_op) & ~ky
        if ff.sum() >= 2 and ky.any() and not fo.any():
            score = ff.sum() + ky.sum() / 10
            if best is None or score > best[0]:
                best = (score, v, t, p, kt, ky, kp)
    _, v, t, p, kt, ky, kp = best
    t0 = t.min(); t, kt = (t - t0) / 60, (kt - t0) / 60

    fig, ax = plt.subplots(figsize=(3.3, 1.75))
    for a_, b_ in zip(kt[:-1], kt[1:]):
        pass
    on = np.flatnonzero(ky)
    for i in on:                                              # shade keyframes where surgeons say CVS
        ax.axvspan(kt[i] - 2.5 / 60, kt[i] + 2.5 / 60, color="#2e8b57", alpha=0.18, lw=0)
    ax.plot(t, p, c="k", lw=0.9, label=r"$\hat P$(CVS), 10 s smoothing")
    ax.axhline(lam_frame, c="#e07b1a", lw=1.0, ls="--", label=f"frame-level threshold (CRC), α={ALPHA:.2f}")
    ax.axhline(min(lam_op, 1.0), c="#1f4e9c", lw=1.2, label=f"surgery-level threshold (ours), α={ALPHA:.2f}")
    fx = (kp > lam_frame) & ~ky
    ax.scatter(kt[fx], kp[fx], marker="x", c="#c0392b", s=14, lw=1.0, zorder=5, label="false CVS confirmation (frame-level)")
    ax.axvline(kt.max(), c="#7a7a7a", lw=0.8, ls=":")
    ax.text(kt.max(), 1.02, "first clip", ha="right", va="bottom", fontsize=6.5, color="#555")
    ax.set_xlabel("time in dissection phase (min)"); ax.set_ylabel(r"$\hat P$(CVS achieved)")
    ax.set_ylim(0, 1.08); ax.set_xlim(0, kt.max() + 0.2)
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.1)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_timeline.{ext}", dpi=300, bbox_inches="tight")
    print("video", v, "frame threshold", round(float(lam_frame), 3), "op threshold", round(float(lam_op), 3),
          "false confirmations (frame-level)", int(fx.sum()))


if __name__ == "__main__":
    main()
