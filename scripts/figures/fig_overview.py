"""Figure 1 (overview), drawn from real data: official Endoscapes split, DINOv2-L seed 0, 10 s smoothing,
alpha = 0.10.  Left band: offline calibration on past operations (real keyframes, their scores and
the surgeons' CVS reference; one score per operation = its highest score on a frame without CVS; the
threshold is an order statistic of these maxima).  Right band: online certification of a new test
operation (real keyframes and score trace; confirm CVS above the threshold, otherwise defer; at the
time-out confirm / STOP / defer).  Bottom strip: the unit of the guarantee.

    python scripts/figures/fig_overview.py
"""
import math
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(ROOT / "scripts" / "review"))
sys.path.insert(0, str(ROOT / "src"))
import style as ST  # noqa: E402
from cdss.data import keyframes  # noqa: E402
from common import endoscapes_ops  # noqa: E402

ST.setup()
import matplotlib.pyplot as plt  # noqa: E402

ALPHA, W = 0.10, 10
OUT = ROOT / "paper" / "figures"
SAFE, UNSAFE = "#138a60", "#d03b3b"
BAND = "#f4f3ef"


def files():
    out = {}
    for split in ("val", "test"):
        kf = keyframes(split)
        for v, f, p in zip(kf.vid, kf.frame, kf.files):
            out[(int(v), int(f))] = p
    return out


def thumb(path, w=320):
    im = Image.open(path).convert("RGB")
    return np.asarray(im.resize((w, int(w * im.height / im.width))))


def box(ax, x, y, w, h, fc, ec="none", r=0.012, lw=0.8, z=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z, transform=ax.transAxes))


def arrow(ax, x0, y0, x1, y1, c=ST.MUTED, lw=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=7,
                                 color=c, lw=lw, zorder=5))


def img(fig, ax, arr, x, y, w, aspect=480 / 854):
    """Place an image in axes-fraction coordinates of `ax` (width w, height from the aspect)."""
    bb = ax.get_position(); fw, fh = fig.get_size_inches()
    h = w * aspect * (bb.width * fw) / (bb.height * fh)
    sub = fig.add_axes([bb.x0 + x * bb.width, bb.y0 + y * bb.height, w * bb.width, h * bb.height])
    sub.imshow(arr); sub.axis("off")
    return h


def main():
    te = set(keyframes("test").vid.tolist())
    ops = endoscapes_ops("dinov2l_b8", 0, (W,))
    F = files()
    cal = [o for o in ops if o["vid"] not in te]
    m = np.sort([o["p"][W][~o["cvs"]].max() for o in cal if (~o["cvs"]).any()])
    j = math.ceil((len(m) + 1) * (1 - ALPHA)); lam = m[j - 1]
    # three calibration operations spanning the range of operation scores: the highest (sets the tail),
    # one near the threshold, and one that reaches CVS
    cand = [o for o in cal if (~o["cvs"]).any()]
    mv = {o["vid"]: o["p"][W][~o["cvs"]].max() for o in cand}
    srt = sorted(cand, key=lambda o: mv[o["vid"]])
    hi, near = srt[-1], srt[len(srt) // 2]
    low_cvs = [o for o in srt[: len(srt) // 2] if o["cvs"].any()]
    mid = low_cvs[0] if low_cvs else srt[len(srt) // 4]
    show = sorted([hi, near, mid], key=lambda o: -mv[o["vid"]])
    new = next(o for o in ops if o["vid"] == 196)

    fig = plt.figure(figsize=(ST.DOUBLE, 2.55))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # bands
    box(ax, 0.005, 0.20, 0.475, 0.79, BAND); box(ax, 0.495, 0.20, 0.50, 0.79, BAND)
    ax.text(0.015, 0.955, "Offline · calibrate on past operations", fontsize=8, weight="bold", va="center")
    ax.text(0.505, 0.955, "Online · certify a new operation", fontsize=8, weight="bold", va="center")

    # --- offline: three past operations, 4 keyframes each, score and reference under each frame
    ys = [0.73, 0.505, 0.28]
    for row, (o, y) in enumerate(zip(show, ys)):
        p, c, fr = o["p"][W], o["cvs"], o["frame"]
        unsafe = np.flatnonzero(~c); top = unsafe[np.argmax(p[unsafe])]
        pick = {top, unsafe[0]}
        if c.any():
            pick.add(np.flatnonzero(c)[np.argmax(p[c])])
        rest = [i for i in np.linspace(0, len(p) - 1, 9).round().astype(int) if i not in pick]
        while len(pick) < 4:
            pick.add(rest.pop(len(rest) // 2))
        for k, i in enumerate(sorted(pick)):
            x = 0.015 + k * 0.067
            h = img(fig, ax, thumb(F[(o["vid"], int(fr[i]))]), x, y, 0.062)
            ax.add_patch(Rectangle((x, y - 0.035), 0.062, 0.028, transform=ax.transAxes, fc=SAFE if c[i] else UNSAFE,
                                   lw=0, alpha=0.9))
            ax.text(x + 0.031, y - 0.021, f"{p[i]:.2f}", ha="center", va="center", fontsize=6.3, color="white",
                    weight="bold" if i == top else "normal")
            if i == top:
                ax.add_patch(Rectangle((x - 0.002, y - 0.037), 0.066, h + 0.04, transform=ax.transAxes, fill=False,
                                       ec=ST.INK, lw=1.0, zorder=6))
        ax.text(0.015, y + h + 0.012, f"operation {row + 1}", fontsize=6.5, color=ST.MUTED, va="bottom")
        arrow(ax, 0.288, y + h / 2 - 0.01, 0.318, y + h / 2 - 0.01)
        mval = p[~c].max()
        box(ax, 0.322, y + h / 2 - 0.045, 0.058, 0.07, "white", ec=ST.INK, lw=0.7, z=4)
        ax.text(0.351, y + h / 2 - 0.01, f"{mval:.2f}", ha="center", va="center", fontsize=7, zorder=6)
    ax.text(0.351, 0.905, "one score\nper operation", ha="center", va="center", fontsize=6.5, color=ST.MUTED)
    # threshold from the order statistic of the operation scores (all 31 calibration operations)
    xa, y0, y1 = 0.425, 0.28, 0.86                        # vertical axis: operation score
    top_m = max(m.max(), 0.05) * 1.05
    ys_ = lambda v: y0 + (y1 - y0) * min(v, top_m) / top_m
    ax.plot([xa, xa], [y0, y1], c=ST.MUTED, lw=0.6, transform=ax.transAxes)
    rng = np.random.default_rng(0)
    shown = {mv[o["vid"]] for o in show}
    for v in m:
        x = xa + rng.uniform(-0.018, 0.018)
        ax.scatter([x], [ys_(v)], s=9, c=ST.OURS if v <= lam else "#9ec5f4", ec="white", lw=0.3,
                   transform=ax.transAxes, zorder=6)
    for v in shown:
        ax.scatter([xa], [ys_(v)], s=26, facecolors="none", edgecolors=ST.INK, lw=0.8, transform=ax.transAxes, zorder=7)
    ylam = ys_(lam)
    ax.plot([xa - 0.035, xa + 0.035], [ylam, ylam], c=ST.OURS, lw=1.3, transform=ax.transAxes, zorder=5)
    ax.text(xa + 0.006, ylam - 0.016, f"λ={lam:.2f}", fontsize=6.6, color=ST.OURS, va="top", ha="left",
            weight="bold", bbox=dict(fc=BAND, ec="none", pad=0.4))
    ax.text(xa, 0.905, f"{len(m)} past\noperations", ha="center", va="center", fontsize=6.5, color=ST.MUTED)
    arrow(ax, 0.478, ylam, 0.508, ylam, c=ST.OURS, lw=1.2)

    # --- online: new operation, frames + score trace with the threshold
    p, c, fr = new["p"][W], new["cvs"], new["frame"]
    idx = np.linspace(0, len(p) - 1, 6).round().astype(int)
    for k, i in enumerate(idx):
        img(fig, ax, thumb(F[(196, int(fr[i]))]), 0.505 + k * 0.0655, 0.70, 0.061)
    t = np.arange(len(p)) / (len(p) - 1)
    tx0, tx1, ty0, ty1 = 0.51, 0.89, 0.27, 0.66
    sc = max(p.max(), lam) * 1.15
    ax.add_patch(Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0, transform=ax.transAxes, fc="white", ec="#d9d8d3", lw=0.6))
    for i in np.flatnonzero(c):
        ax.add_patch(Rectangle((tx0 + (tx1 - tx0) * (t[i] - 0.5 / len(p)), ty0), (tx1 - tx0) / len(p), ty1 - ty0,
                               transform=ax.transAxes, fc="#cfe9dc", lw=0))
    ax.plot(tx0 + (tx1 - tx0) * t, ty0 + (ty1 - ty0) * p / sc, c=ST.INK, lw=0.9, transform=ax.transAxes)
    conf = p > lam
    ax.scatter((tx0 + (tx1 - tx0) * t)[conf], (ty0 + (ty1 - ty0) * p / sc)[conf], s=7, c=ST.OURS, zorder=6,
               transform=ax.transAxes)
    ax.plot([tx0, tx1], [ty0 + (ty1 - ty0) * lam / sc] * 2, c=ST.OURS, lw=1.2, transform=ax.transAxes)
    ax.text(tx0 + 0.005, ty1 - 0.03, "score", fontsize=6.5, color=ST.MUTED, va="top")
    ax.text(tx1 - 0.004, ty0 + 0.02, "time →", fontsize=6.5, color=ST.MUTED, ha="right")
    ax.text(tx0 + 0.005, ty0 + (ty1 - ty0) * lam / sc + 0.012, "λ (streaming threshold)", fontsize=6.5, color=ST.OURS, weight="bold")
    for k, i in enumerate(idx):                           # tie each thumbnail to its time
        xt = tx0 + (tx1 - tx0) * t[i]; xi = 0.505 + k * 0.0655 + 0.0305
        ax.plot([xi, xt], [0.70, ty1], c="#c9c8c2", lw=0.5, transform=ax.transAxes, zorder=1)
    # outputs
    ox = 0.905
    for k, (lab, fc, ec, tc) in enumerate((("confirm\nCVS", ST.OURS, ST.OURS, "white"),
                                           ("defer to\nsurgeon", "white", ST.MUTED, ST.INK))):
        yy = 0.52 - k * 0.17
        box(ax, ox, yy, 0.083, 0.12, fc, ec=ec, lw=0.8, z=4)
        ax.text(ox + 0.0415, yy + 0.06, lab, ha="center", va="center", fontsize=6.6, color=tc, zorder=6,
                weight="bold" if k == 0 else "normal")
    arrow(ax, tx1 + 0.002, 0.52, ox - 0.002, 0.58, c=ST.OURS); arrow(ax, tx1 + 0.002, 0.42, ox - 0.002, 0.41)
    ax.text(0.95, 0.27, "time-out: own thresholds,\nconfirm / STOP / defer", ha="center", va="center", fontsize=6, color=ST.MUTED)
    ax.text(0.505, 0.235, "green: surgeons rate CVS achieved", fontsize=6, color=ST.MUTED, va="center")

    # --- bottom: the unit of the guarantee
    for yy, head, body, col, unit in ((0.115, "Frame-level guarantee", "≤ α of frames err, yet 45–57% of operations err",
                                       ST.CRC, "frame"),
                                      (0.045, "Per-operation guarantee (ours)", "≤ α of operations get any false confirmation",
                                       ST.OURS, "operation")):
        ax.text(0.015, yy, head, fontsize=7, weight="bold", va="center")
        ax.text(0.245, yy, body, fontsize=7, va="center", color=col)
        ax.text(0.85, yy, f"unit = {unit}", fontsize=6.5, color=ST.MUTED, va="center")
    for k in range(10):
        ax.add_patch(Rectangle((0.715 + k * 0.0125, 0.10), 0.0095, 0.03, transform=ax.transAxes,
                               fc=UNSAFE if k in (2, 3, 6) else "#d9d8d3", lw=0))
    ax.add_patch(FancyBboxPatch((0.715, 0.03), 0.122, 0.03, boxstyle="round,pad=0,rounding_size=0.006",
                                transform=ax.transAxes, fc="#d9d8d3", lw=0))
    ST.save(fig, OUT / "fig_overview")
    print("lambda", round(float(lam), 3), "V", len(m), "shown", [o["vid"] for o in show])


if __name__ == "__main__":
    main()
