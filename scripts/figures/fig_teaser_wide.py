"""Full-width teaser in the style of an accident-anticipation figure: four original frames of one
Endoscapes test operation above the 1 fps CVS score trace, with the two thresholds, the first false
confirmation of the frame-level rule, and the time it keeps reassuring the surgeon before the surgeons
actually rate CVS achieved.

Operation, thresholds and smoothing come from fig_timeline.select_operation() (video 194, alpha = 0.05,
10 s causal smoothing, DINOv2-L seed 0).  Bands are measured, not modelled:
  wheat  min-max of the smoothed score across the three model seeds (each with its own temperature)
  blue   min-max of the unsmoothed 1 fps scores inside the 10 s causal window of seed 0
Four frames: first false confirmation, highest-scoring false confirmation, first surgeon-rated CVS
timestamp, and the last assessed timestamp (CVS achieved).

    python scripts/figures/fig_teaser_wide.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
ASSETS = OUT / "teaser_wide_assets"     # separate frames and plot for manual composition
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdss.data import keyframes                        # noqa: E402
from evaluate import causal_logits, fit_temperature    # noqa: E402
from fig_timeline import ALPHA, SMOOTH, select_operation  # noqa: E402
import style as ST  # noqa: E402

ST.setup()
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})
WHEAT, BLUE, BLUE_DARK, RED, BLACK = "#F5DEB3", "#9ED8F0", "#2F4FA2", "#E53B2C", "#111111"
W, H = ST.DOUBLE, 2.35


def seed_trace(seed, v, t0):
    """Smoothed and raw 1 fps score traces of video v for one seed, each with the seed's own temperature."""
    run = ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{seed}"
    va = keyframes("val")
    sel = np.isin(va.vid, np.random.default_rng(0).choice(np.unique(va.vid), 10, replace=False))
    T = fit_temperature(causal_logits(run / "logits_val.npz", va, 1)[sel], va.soft[sel])
    z = np.load(run / "logits_test.npz")
    rows = np.flatnonzero(z["vid"] == v); rows = rows[np.argsort(z["frame"][rows])]
    L = z["logits"][rows]
    Ls = np.stack([L[max(0, i - SMOOTH + 1):i + 1].mean(0) for i in range(len(L))])
    p_s = np.prod(1 / (1 + np.exp(-Ls / T)), 1)
    p_r = np.prod(1 / (1 + np.exp(-L / T)), 1)
    t = (z["frame"][rows] / 25.0 - t0) / 60
    return t, p_s, p_r


def main():
    S = select_operation()
    v, t, p, kt, ky, kp, lam_frame, lam_op, files, kfr = (S[k] for k in ("v", "t", "p", "kt", "ky", "kp", "lam_frame", "lam_op", "files", "kfr"))
    z = np.load(ROOT / "experiments" / "finetune" / "dinov2l_b8_s0" / "logits_test.npz")
    t0 = z["frame"][z["vid"] == v].min() / 25.0
    traces = [seed_trace(s, v, t0) for s in (0, 1, 2)]
    assert np.allclose(traces[0][1], p, atol=1e-6), "seed-0 trace differs from fig_timeline"
    P = np.stack([tr[1] for tr in traces]); lo, hi = P.min(0), P.max(0)
    raw = traces[0][2]
    rlo = np.array([raw[max(0, i - SMOOTH + 1):i + 1].min() for i in range(len(raw))])
    rhi = np.array([raw[max(0, i - SMOOTH + 1):i + 1].max() for i in range(len(raw))])

    fx = (kp > lam_frame) & ~ky
    i_false1 = int(np.flatnonzero(fx)[0]); i_falsemax = int(np.flatnonzero(fx)[np.argmax(kp[fx])])
    i_cvs1 = int(np.flatnonzero(ky)[0]); i_last = int(np.flatnonzero(ky)[-1])
    picks = [(i_false1, "first false confirmation"), (i_falsemax, "highest false score"),
             (i_cvs1, "surgeons: CVS achieved"), (i_last, "CVS achieved, last assessed")]
    t_false, t_cvs = kt[i_false1], kt[i_cvs1]

    fig = plt.figure(figsize=(W, H))
    # ---- frames row
    n = 4; fh = 0.92; top = 0.02
    im0 = Image.open(files[(v, int(kfr[picks[0][0]]))]); W_, H_ = im0.size
    fw = fh * 0.84 * W_ / H_; gap = (W - 0.6 - n * fw) / (n - 1); x0 = 0.3
    centers = []
    for j, (i, lab) in enumerate(picks):
        im = Image.open(files[(v, int(kfr[i]))]).convert("RGB")
        im = im.crop((int(0.08 * W_), 0, int(0.92 * W_), H_)); im.thumbnail((900, 900))
        ASSETS.mkdir(exist_ok=True); im.save(ASSETS / f"frame_{j + 1}_{kt[i]:.1f}min_{lab.replace(' ', '_').replace(':', '').replace(',', '')}.png")
        xl = x0 + j * (fw + gap)
        ax = fig.add_axes([xl / W, 1 - (top + fh) / H, fw / W, fh / H])
        ax.imshow(np.asarray(im)); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555555"); sp.set_linewidth(0.6)
        ax.text(0.02, 0.96, f"{kt[i]:.1f} min · {lab}", transform=ax.transAxes, fontsize=6.6, color="w", va="top",
                bbox=dict(fc="black", alpha=0.55, ec="none", pad=1.2))
        centers.append((xl + fw / 2, 1 - (top + fh) / H))
    # ---- score trace
    ax = fig.add_axes([0.055, 0.13, 0.935, 1 - (top + fh + 0.12) / H - 0.13])
    tt = traces[0][0]
    ax.fill_between(tt, rlo, rhi, color=BLUE, lw=0, zorder=1, alpha=0.8, label="unsmoothed 1 fps scores, 10 s window")
    ax.fill_between(tt, lo, hi, color=WHEAT, lw=0, zorder=0, alpha=0.9, label="range over three model seeds")
    ax.plot(tt, p, c=BLACK, lw=1.6, zorder=4, label="CVS score, 10 s smoothing")
    ax.axhline(lam_frame, c=BLUE_DARK, ls=(0, (5, 3)), lw=1.1, zorder=3)
    ax.axhline(lam_op, c=BLUE_DARK, ls=(0, (2, 2)), lw=1.1, zorder=3)
    ax.text(0.03, lam_frame + 0.025, "frame-level threshold (CRC), α = 0.05", fontsize=7.2, color=BLUE_DARK, va="bottom")
    ax.text(0.03, lam_op + 0.025, "per-operation threshold (ours), α = 0.05", fontsize=7.2, color=BLUE_DARK, va="bottom")
    # first false confirmation and the reassurance window until the surgeons rate CVS
    ax.scatter([t_false], [kp[i_false1]], s=55, c=RED, zorder=6)
    ax.plot([t_false, t_false], [0, kp[i_false1]], c=BLUE_DARK, ls=(0, (3, 2)), lw=1.0, zorder=3)
    ax.axvline(t_cvs, c=RED, ls=(0, (4, 2)), lw=1.1, zorder=3)
    ax.annotate("", xy=(t_cvs, 0.5), xytext=(t_false, 0.5), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6, mutation_scale=12), zorder=5)
    ax.text((t_false + t_cvs) / 2, 0.56, f"falsely confirmed for {60 * (t_cvs - t_false):.0f} s\nbefore CVS is achieved", fontsize=7.5,
            color=RED, ha="center", va="bottom", linespacing=1.1)
    ax.text(t_cvs + 0.04, 0.72, "CVS achieved\n(surgeons)", fontsize=7.5, color=RED, va="top", linespacing=1.1)
    ax.axvline(kt.max(), c="#777777", ls=":", lw=0.9); ax.text(kt.max() - 0.03, 0.05, "first clip", fontsize=7, color="#777777", ha="right")
    ax.set_xlim(0, kt.max() + 0.25); ax.set_ylim(0, 1.0); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    # arrows from the frames to their timestamps on the trace, resolved in data coordinates at draw time
    for (xc, yb), (i, _) in zip(centers, picks):
        fig.add_artist(matplotlib.patches.ConnectionPatch(
            xyA=(xc / W, yb - 0.008), coordsA="figure fraction", xyB=(kt[i], float(np.interp(kt[i], tt, p))), coordsB="data",
            axesB=ax, arrowstyle="-|>", color=BLUE_DARK, lw=0.9, mutation_scale=9, shrinkA=0, shrinkB=3, zorder=7))
    ax.set_xlabel("time in dissection (min)", labelpad=1, loc="left"); ax.set_ylabel("P(CVS)")
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True); ax.spines[sp].set_color(ST.MUTED)
    hh, ll = ax.get_legend_handles_labels()
    fig.legend(hh, ll, loc="lower right", bbox_to_anchor=(0.992, 0.005), ncol=3, fontsize=6.8, frameon=False,
               handlelength=1.4, columnspacing=1.0, handletextpad=0.5)
    ST.save(fig, OUT / "fig_teaser_wide")
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(OUT / "fig_teaser_wide.svg")
    # plot alone (same axes, no frames, no arrows) for manual composition
    for a_ in list(fig.artists):
        a_.remove()
    for ax_ in [a for a in fig.axes if a is not ax]:
        ax_.remove()
    fig.savefig(ASSETS / "plot_only.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(ASSETS / "plot_only.png", dpi=300, bbox_inches="tight", pad_inches=0.02)


    print(f"video {v}; thresholds frame {lam_frame:.3f} op {lam_op:.3f}; first false {t_false:.2f} min, first CVS {t_cvs:.2f} min, "
          f"window {60 * (t_cvs - t_false):.0f} s; frames at " + ", ".join(f"{kt[i]:.2f}" for i, _ in picks))


if __name__ == "__main__":
    main()
