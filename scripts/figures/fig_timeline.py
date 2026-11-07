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

    from PIL import Image
    files = {(int(a_), int(b_)): f for a_, b_, f in zip(te.vid, te.frame, te.files)}
    kfr = te.frame[te.vid == v][np.argsort(te.frame[te.vid == v])]
    fx = (kp > lam_frame) & ~ky
    i_false1 = int(np.flatnonzero(fx)[0]); i_falsemax = int(np.flatnonzero(fx)[np.argmax(kp[fx])])
    i_cvs = int(np.flatnonzero(ky)[-1])
    picks = [(i_false1, "first false confirmation\n(frame-level)", ST.CRC), (i_falsemax, "highest-scoring false\nconfirmation", ST.CRC),
             (i_cvs, "CVS achieved\n(surgeons)", "#138a60")]
    fig = plt.figure(figsize=(ST.DOUBLE, 3.35))
    ax = fig.add_axes([0.065, 0.335, 0.925, 0.395])
    for j, i in enumerate(np.flatnonzero(ky)):                 # reference: CVS keyframes
        ax.axvspan(kt[i] - 2.5 / 60, kt[i] + 2.5 / 60, color="#cfe9dc", lw=0, zorder=0,
                   label="surgeons rate CVS achieved" if j == 0 else None)
    ax.plot(t, p, c=ST.INK, lw=0.9, zorder=2, label="model score (10 s smoothing)")
    ax.axhline(lam_frame, c=ST.CRC, lw=1.2, ls=(0, (4, 2)), zorder=3, label="frame-level threshold (CRC), α = 0.05")
    ax.axhline(min(lam_op, 1.0), c=ST.OURS, lw=1.4, zorder=3, label="per-operation threshold (ours), α = 0.05")
    ax.scatter(kt[fx], kp[fx], marker="x", c=ST.CRC, s=18, lw=1.1, zorder=5, label="false CVS confirmation")
    ax.axvline(kt.max(), c=ST.MUTED, lw=0.8, ls=":")
    ax.text(kt.max() - 0.03, 0.92, "first clip", ha="right", va="top", fontsize=6.5, color=ST.MUTED)
    ax.set_ylabel("score"); ax.tick_params(labelbottom=False)
    ax.set_ylim(0, 1.0); ax.set_xlim(0, kt.max() + 0.2); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3, columnspacing=1.0, handlelength=1.4, fontsize=6.3, bbox_to_anchor=(0.0, 1.0))
    # decision tracks at the assessed timestamps (5 s apart); the video in between is not assessed
    tr = fig.add_axes([0.065, 0.135, 0.925, 0.19], sharex=ax)
    oc = kp > lam_op
    rows_ = [("reference (surgeons)", [("#138a60" if y_ else "#d9d8d3") for y_ in ky]),
             ("frame-level CRC", [(ST.CRC if (c_ and not y_) else ("#138a60" if c_ else "none")) for c_, y_ in zip(kp > lam_frame, ky)]),
             ("per operation (ours)", [(ST.CRC if (c_ and not y_) else ("#138a60" if c_ else "none")) for c_, y_ in zip(oc, ky)])]
    for r_, (name, cols) in enumerate(rows_):
        y_ = 2 - r_
        for x_, c_ in zip(kt, cols):
            tr.add_patch(plt.Rectangle((x_ - 2.2 / 60, y_ - 0.36), 4.4 / 60, 0.72, fc=c_, ec="#b9b8b2" if c_ == "none" else "none",
                                       lw=0.4, zorder=2))
        tr.text(-0.004, y_, name, ha="right", va="center", fontsize=6.3, transform=tr.get_yaxis_transform())
    tr.set_ylim(-0.6, 2.6); tr.set_yticks([]); tr.grid(False)
    for sp in tr.spines.values():
        sp.set_visible(False)
    tr.set_xlabel("time in dissection (min)")
    fig.text(0.065, 0.005, "decision tracks at the assessed timestamps (5 s apart; the video in between is not assessed): "
             "green = CVS / correct confirmation, red = false confirmation, outline = no confirmation",
             fontsize=5.8, color=ST.MUTED, va="bottom")
    # thumbnails above the plot, tied to their timestamps
    xs = [0.12, 0.40, 0.78]
    for (i, lab, col), xc in zip(picks, xs):
        im = Image.open(files[(v, int(kfr[i]))]).convert("RGB"); W_, H_ = im.size
        im = im.crop((int(0.08 * W_), 0, int(0.92 * W_), H_))
        w_ = 0.155; h_ = w_ * (im.height / im.width) * (ST.DOUBLE / 3.35)
        sub = fig.add_axes([xc - w_ / 2, 0.76, w_, h_]); sub.imshow(np.asarray(im)); sub.axis("off")
        for sp in sub.spines.values():
            sp.set_visible(True); sp.set_edgecolor(col); sp.set_linewidth(1.5)
        sub.set_xticks([]); sub.set_yticks([])
        sub.axis("on"); sub.set_xticks([]); sub.set_yticks([])
        fig.text(xc, 0.76 + h_ + 0.012, f"{kt[i]:.1f} min · " + lab.replace("\n", " "), ha="center", va="bottom", fontsize=6.2, color=col)
        xd = 0.065 + 0.925 * (kt[i] / (kt.max() + 0.2))
        fig.add_artist(plt.Line2D([xc, xd], [0.76, 0.73], transform=fig.transFigure, c=col, lw=0.6, ls=":"))
    ST.save(fig, OUT / "fig_timeline")
    print("video", v, "frame threshold", round(float(lam_frame), 3), "op threshold", round(float(lam_op), 3),
          "false confirmations (frame-level)", int(fx.sum()))


if __name__ == "__main__":
    main()
