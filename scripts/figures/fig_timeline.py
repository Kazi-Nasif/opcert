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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ST  # noqa: E402

ST.setup()


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

    fig, ax = plt.subplots(figsize=(ST.SINGLE, 1.95))
    on = np.flatnonzero(ky)
    for j, i in enumerate(on):                                # keyframes the surgeons rate as CVS
        ax.axvspan(kt[i] - 2.5 / 60, kt[i] + 2.5 / 60, color="#cfe9dc", lw=0, zorder=0,
                   label="surgeons: CVS achieved" if j == 0 else None)
    ax.plot(t, p, c=ST.INK, lw=0.9, zorder=2, label="model score (10 s smoothing)")
    ax.axhline(lam_frame, c=ST.CRC, lw=1.2, ls=(0, (4, 2)), zorder=3, label=f"frame-level threshold")
    ax.axhline(min(lam_op, 1.0), c=ST.OURS, lw=1.4, zorder=3, label=f"per-operation threshold (ours)")
    fx = (kp > lam_frame) & ~ky
    ax.scatter(kt[fx], kp[fx], marker="x", c=ST.CRC, s=16, lw=1.1, zorder=5, label="false CVS confirmation")
    ax.axvline(kt.max(), c=ST.MUTED, lw=0.8, ls=":")
    ax.text(kt.max() - 0.03, 0.98, "first clip", ha="right", va="top", fontsize=6.5, color=ST.MUTED)
    ax.set_xlabel("time in dissection (min)"); ax.set_ylabel("score")
    ax.set_ylim(0, 1.0); ax.set_xlim(0, kt.max() + 0.2); ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.0, handlelength=1.4,
              fontsize=6.5)
    fig.tight_layout(pad=0.2)
    ST.save(fig, OUT / "fig_timeline")
    print("video", v, "frame threshold", round(float(lam_frame), 3), "op threshold", round(float(lam_op), 3),
          "false confirmations (frame-level)", int(fx.sum()))


if __name__ == "__main__":
    main()
