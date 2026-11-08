"""First-page teaser (main Fig. 1), single column: why the guarantee must hold per operation.

Top: three original frames of the automatically selected Endoscapes test operation of
fig_timeline.py (first false confirmation under frame-level CRC, its highest-scoring false
confirmation, and a frame the surgeons rate as CVS), the three decision tracks at the 36 assessed
timestamps (5 s apart) under the same thresholds (alpha = 0.05, 10 s smoothing).  Bottom: the
aggregate result of the same two policies from experiments/compare_all/compare_all.csv (DINOv2-L,
200 overlapping video-level splits x 3 seeds, alpha = 0.05, 10 s smoothing): operations with >= 1
false confirmation and CVS-frame sensitivity.  The operation, thresholds and decisions are the ones
of the supplementary timeline figure; nothing is drawn that is not in the stored outputs.

Writes paper/figures/fig_teaser.{pdf,png,svg}; the SVG keeps text editable.

    python scripts/figures/fig_teaser.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ST  # noqa: E402
from fig_timeline import ALPHA, SMOOTH, select_operation  # noqa: E402

ST.setup()
GREEN, NEG, INK2 = "#138A60", "#D9D8D3", "#202B36"
W, H = ST.SINGLE, 2.78                     # inches


def aggregate():
    """Means at alpha = ALPHA with SMOOTH s smoothing (the rows of main Table 1)."""
    d = pd.read_csv(ROOT / "experiments" / "compare_all" / "compare_all.csv")
    d = d[d.run.str.startswith("dinov2l_b8") & (d.alpha == ALPHA)]
    rows = {"crc": "CRC + smoothing (10 s)", "ours": "Surgery-level + smoothing (ours)"}
    out = {}
    for k, m in rows.items():
        g = d[d.method == m]
        assert len(g) == 600, (m, len(g))            # 200 splits x 3 seeds
        out[k] = (100 * g.ops_false_ge1.mean(), 100 * g.sens.mean())
    return out


def fx_(x):      # inches -> figure fraction
    return x / W


def fy_(y):      # inches from the top -> figure fraction
    return 1 - y / H


def main():
    S = select_operation()
    kt, ky, kp, lam_frame, lam_op = (S[k] for k in ("kt", "ky", "kp", "lam_frame", "lam_op"))
    files, kfr, v = S["files"], S["kfr"], S["v"]
    A = aggregate()
    assert len(kt) == 36 and np.allclose(np.diff(kt) * 60, 5.0)

    fx = (kp > lam_frame) & ~ky
    i_false1 = int(np.flatnonzero(fx)[0]); i_falsemax = int(np.flatnonzero(fx)[np.argmax(kp[fx])])
    i_cvs = int(np.flatnonzero(ky)[-1])
    picks = [(i_false1, "first false confirmation", ST.CRC), (i_falsemax, "highest false score", ST.CRC),
             (i_cvs, "CVS achieved (surgeons)", GREEN)]

    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    # headline
    fig.text(fx_(0.02), fy_(0.10), "The frame is the wrong unit of guarantee", fontsize=8.6, fontweight="bold",
             color=INK2, ha="left", va="center")
    fig.text(fx_(0.02), fy_(0.235), f"same model · same nominal level α = {int(ALPHA * 100)}% · {SMOOTH} s smoothing",
             fontsize=6.6, color=ST.MUTED, ha="left", va="center")

    # three original frames, 8 % cropped at each side as in the supplementary timeline
    gap, x0 = 0.07, 0.02
    wi = (W - 2 * x0 - 2 * gap) / 3
    top = 0.33
    for j, (i, lab, col) in enumerate(picks):
        im = Image.open(files[(v, int(kfr[i]))]).convert("RGB"); W_, H_ = im.size
        im = im.crop((int(0.08 * W_), 0, int(0.92 * W_), H_))
        im.thumbnail((720, 720))                                   # 300 dpi at the printed width
        hi = wi * im.height / im.width
        xl = x0 + j * (wi + gap)
        ax = fig.add_axes([fx_(xl), fy_(top + hi), fx_(wi), hi / H])
        ax.imshow(np.asarray(im)); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(col); sp.set_linewidth(1.2)
        fig.text(fx_(xl), fy_(top + hi + 0.075), f"{kt[i]:.1f} min", fontsize=6.6, fontweight="bold", color=ST.MUTED,
                 ha="left", va="center")
        fig.text(fx_(xl), fy_(top + hi + 0.175), lab, fontsize=6.3, color=col, ha="left", va="center")
    y = top + hi + 0.30

    # decision tracks at the 36 assessed timestamps
    fig.text(fx_(0.02), fy_(y), "One operation · assessed timestamps only", fontsize=6.8, fontweight="bold",
             color=INK2, ha="left", va="center")
    oc = kp > lam_op
    rows = [("Surgeons", INK2, "normal", [(GREEN if y_ else NEG) for y_ in ky]),
            ("Frame-level CRC", ST.CRC, "normal", [(ST.CRC if (c_ and not y_) else (GREEN if c_ else "none")) for c_, y_ in zip(kp > lam_frame, ky)]),
            ("Per operation (ours)", ST.OURS, "bold", [(ST.CRC if (c_ and not y_) else (GREEN if c_ else "none")) for c_, y_ in zip(oc, ky)])]
    pitch, ch = 0.125, 0.085
    xl, xr = 0.96, W - 0.02
    tr = fig.add_axes([fx_(xl), fy_(y + 0.10 + 3 * pitch), fx_(xr - xl), 3 * pitch / H])
    tr.set_xlim(kt.min() - 2.5 / 60, kt.max() + 2.5 / 60); tr.set_ylim(0, 3 * pitch); tr.axis("off")
    cw = 4.0 / 60                                                 # cell width in minutes (5 s pitch)
    for r, (name, col, wt, cols) in enumerate(rows):
        yc = (2 - r) * pitch + pitch / 2
        for x_, c_ in zip(kt, cols):
            tr.add_patch(plt.Rectangle((x_ - cw / 2, yc - ch / 2), cw, ch, fc=c_, ec="#b9b8b2" if c_ == "none" else "none",
                                       lw=0.4, zorder=2))
        fig.text(fx_(xl - 0.05), fy_(y + 0.10 + r * pitch + pitch / 2), name, fontsize=6.4, fontweight=wt, color=col,
                 ha="right", va="center")
    y += 0.10 + 3 * pitch
    fig.text(fx_(xr), fy_(y + 0.06), "5 s apart · outline = no confirmation", fontsize=6.2, color=ST.MUTED,
             ha="right", va="center")
    y += 0.20

    # aggregate: the same two policies over 200 splits x 3 seeds
    fig.text(fx_(0.02), fy_(y), "All operations · means over 200 splits × 3 seeds", fontsize=6.8, fontweight="bold",
             color=INK2, ha="left", va="center")
    y += 0.10
    panels = [("operations with ≥1\nfalse confirmation (%)", 0, 5.0), ("CVS-frame\nsensitivity (%)", 1, None)]
    pw, ph, px0 = 1.02, 0.42, 0.96
    for j, (title, k, target) in enumerate(panels):
        x = px0 + j * (pw + 0.22)
        ax = fig.add_axes([fx_(x), fy_(y + 0.20 + ph), fx_(pw), ph / H])
        vals = [A["crc"][k], A["ours"][k]]
        ax.barh([1, 0], vals, height=0.62, color=[ST.CRC, ST.OURS], zorder=3)
        for yy, val, col in zip([1, 0], vals, [ST.CRC, ST.OURS]):
            xt = val + 1.5 if target is None or val > target + 8 else target + 1.5    # clear of the target line
            ax.text(xt, yy, f"{val:.1f}%", fontsize=6.6, fontweight="bold", color=col, ha="left", va="center")
        if target is not None:
            ax.axvline(target, ls=(0, (1.2, 1.4)), c=ST.INK, lw=0.8, zorder=4)
            ax.text(target + 1.5, 1.52, "nominal level 5%", fontsize=6.0, color=ST.INK, ha="left", va="center")
        ax.set_xlim(0, 68); ax.set_ylim(-0.5, 1.75); ax.axis("off")
        ax.text(0, 1.0, title, transform=ax.transAxes, fontsize=6.2, color=ST.MUTED, ha="left", va="bottom", linespacing=1.15)
    for r, (name, col, wt) in enumerate([("Frame-level CRC", ST.CRC, "normal"), ("Per operation (ours)", ST.OURS, "bold")]):
        yc = y + 0.20 + ph * (1 - (1 - r + 0.5) / 2.25)           # bar centres of the data limits above
        fig.text(fx_(px0 - 0.05), fy_(yc), name, fontsize=6.4, fontweight=wt, color=col, ha="right", va="center")
    y += 0.20 + ph

    ST.save(fig, OUT / "fig_teaser")
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(OUT / "fig_teaser.svg")
    print(f"video {v}; thresholds frame {lam_frame:.3f} op {lam_op:.3f}; false confirmations (frame-level) {int(fx.sum())}; "
          f"frames at {kt[i_false1]:.1f}, {kt[i_falsemax]:.1f}, {kt[i_cvs]:.1f} min")
    for k, name in (("crc", "frame-level CRC"), ("ours", "per operation")):
        print(f"{name}: ops with >=1 false confirmation {A[k][0]:.1f}%  CVS-frame sensitivity {A[k][1]:.1f}%")
    print("content height used (in):", round(y, 2), "of", H)


if __name__ == "__main__":
    main()
