"""Architecture overview (main Fig. 2), CVPR double column, drawn from real data and real frames.

(a) Offline calibration on labeled operations: past operations -> perception model -> one score per
    operation per statement -> order statistic at level alpha -> fixed thresholds.  The dot plot shows
    the real per-operation maxima (highest smoothed score on a frame without CVS) of the calibration
    operations of the official Endoscapes split (DINOv2-L seed 0, 10 s smoothing) and the threshold at
    alpha = 0.10.
(b) Online scoring of a new operation (labels unavailable): causal frame stream -> the same perception
    model -> CVS score (C1, C2, C3 -> p) and anatomy candidates (our Faster R-CNN output on a held-out
    test frame).
(c) Certification: each statement is issued only above its fixed threshold, otherwise the system
    defers; outputs during the operation, at the time-out, and for anatomy labels.

Frames: paper/figures/arch_assets (Endoscapes2023, CC BY-NC-SA 4.0; see SOURCE.txt there).
Writes paper/figures/fig_overview_v2.{pdf,png,svg}.

    python scripts/figures/fig_overview_v2.py
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(ROOT / "scripts" / "review"))
sys.path.insert(0, str(ROOT / "src"))
import style as ST  # noqa: E402
from cdss.data import keyframes  # noqa: E402
from common import endoscapes_ops  # noqa: E402

ST.setup()
ASSETS = ROOT / "paper" / "figures" / "arch_assets"
OUT = ROOT / "paper" / "figures"
ALPHA, SMOOTH = 0.10, 10
W, H = ST.DOUBLE, 2.98                                   # inches

# SafeWatch-like pastel palette, black ink
YEL, BLU, ORA, GRN, RED = "#FBF1CF", "#D6E3F0", "#F6E2CC", "#D9EBD6", "#F5D5D0"
INK, MUTE, LINE = "#111111", "#555555", "#333333"
plt.rcParams.update({"font.size": 7.2, "text.color": INK})


# ----------------------------------------------------------------------------- helpers (inch coords)
# The drawing axes uses inches with y increasing DOWNWARD (ylim = (H, 0)); x, y of a box = its top-left.
def fx(x): return x / W
def fy(y): return 1 - y / H


def rbox(ax, x, y, w, h, fc, ec="none", lw=0.8, r=0.05, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z))


def arrow(ax, x0, y0, x1, y1, c=LINE, lw=1.0, ls="-", ms=7, z=6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=c, lw=lw,
                                 linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, size=7.2, weight="normal", color=INK, ha="left", va="center", z=8, **kw):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z, **kw)


def image(fig, path, x, y, w, crop=None, border=None, z=3):
    """Image with its top-left corner at (x, y) inches and width w inches; returns its height."""
    im = (path if isinstance(path, Image.Image) else Image.open(path)).convert("RGB")
    if crop:
        im = im.crop(tuple(int(c * s) for c, s in zip(crop, (im.width, im.height, im.width, im.height))))
    im.thumbnail((900, 900))
    h = w * im.height / im.width
    ax = fig.add_axes([fx(x), fy(y + h), fx(w), h / H], zorder=z)
    ax.imshow(np.asarray(im), aspect="auto"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(border is not None)
        if border:
            sp.set_edgecolor(border); sp.set_linewidth(0.8)
    return h


def nn_icon(ax, x, y, s=0.26, c=INK):
    """3-2-3 network glyph, top-left (x, y), width s, height 0.76 s."""
    cols = [3, 2, 3]; xs = [x + s * i / 2 for i in range(3)]; r = s * 0.075
    pts = []
    for cx, n in zip(xs, cols):
        ys = [y + s * 0.38 + (k - (n - 1) / 2) * s * 0.38 for k in range(n)]
        pts.append([(cx, yy) for yy in ys])
    for a, b in zip(pts[:-1], pts[1:]):
        for p_ in a:
            for q in b:
                ax.plot([p_[0], q[0]], [p_[1], q[1]], c=c, lw=0.45, zorder=7)
    for col in pts:
        for p_ in col:
            ax.add_patch(Circle(p_, r, fc="white", ec=c, lw=0.6, zorder=8))


def clock_icon(ax, cx, cy, r=0.065, c=INK):
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=c, lw=0.8, zorder=8))
    ax.plot([cx, cx], [cy, cy - r * 0.62], c=c, lw=0.8, zorder=9)          # hand pointing up
    ax.plot([cx, cx + r * 0.5], [cy, cy], c=c, lw=0.8, zorder=9)


def check_icon(ax, cx, cy, r=0.065, fc="#2E9E5B"):
    ax.add_patch(Circle((cx, cy), r, fc=fc, ec="none", zorder=8))
    ax.plot([cx - r * 0.45, cx - r * 0.1, cx + r * 0.5], [cy, cy + r * 0.4, cy - r * 0.4], c="white", lw=1.1,
            zorder=9, solid_capstyle="round")


def tag_icon(ax, cx, cy, w=0.15, h=0.10, c=INK):
    """Bounding-box glyph: outlined box with a filled label strip on top."""
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="white", ec=c, lw=0.7, zorder=8))
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2 - 0.035), w * 0.55, 0.035, fc=c, ec="none", zorder=8))


def film(fig, path, x, y, w, n=5, gap=0.025):
    """Strip image of n frames, drawn as n thumbnails with a thin gap; returns the height."""
    im = Image.open(path).convert("RGB"); fw = im.width / n
    wi = (w - (n - 1) * gap) / n; h = wi * im.height / fw
    for k in range(n):
        sub = im.crop((int(k * fw), 0, int((k + 1) * fw), im.height))
        ax = fig.add_axes([fx(x + k * (wi + gap)), fy(y + h), fx(wi), h / H], zorder=3)
        ax.imshow(np.asarray(sub), aspect="auto"); ax.axis("off")
    return h


# ----------------------------------------------------------------------------- data
def calibration_maxima():
    te = set(keyframes("test").vid.tolist())
    ops = endoscapes_ops("dinov2l_b8", 0, (SMOOTH,))
    cal = [o for o in ops if o["vid"] not in te]
    m = np.sort([o["p"][SMOOTH][~o["cvs"]].max() for o in cal if (~o["cvs"]).any()])
    j = math.ceil((len(m) + 1) * (1 - ALPHA))
    return m, j, float(m[j - 1])


def main():
    m, j, lam = calibration_maxima()
    plt.rcParams["savefig.bbox"] = None                     # keep the inch geometry exact
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(H, 0)

    # ============================================================== (a) offline calibration (top band)
    text(ax, 0.06, 0.10, "(a) Offline: calibrate thresholds on labeled operations", 8.0, "bold")
    # labeled past operations, stacked
    sx, sy, sw = 0.06, 0.30, 0.80
    for k, name in enumerate(["past_op_3_labelled.png", "past_op_2_labelled.png", "past_op_1_labelled.png"]):
        off = (2 - k) * 0.06
        sh = image(fig, ASSETS / name, sx + off, sy + off, sw, border="white", z=3 + k)
    text(ax, sx, sy + 0.12 + sh + 0.08, "Past operations\n(surgeons' CVS rating,\nanatomy boxes)", 6.2, color=MUTE,
         va="top", linespacing=1.1)
    cy_a = 0.70                                              # centre line of the (a) flow
    # perception model (shared by a and b)
    px, py, pw, ph = 1.22, 0.28, 1.30, 0.84
    rbox(ax, px, py, pw, ph, BLU)
    text(ax, px + pw / 2, py + 0.10, "Perception model", 7.4, "bold", ha="center")
    nn_icon(ax, px + pw / 2 - 0.13, py + 0.19, 0.26)
    for k, s in enumerate(["DINOv2-L CVS scorer", "10 s causal smoothing", "Faster R-CNN anatomy"]):
        text(ax, px + pw / 2, py + 0.50 + k * 0.11, s, 6.2, ha="center")
    arrow(ax, sx + sw + 0.14, cy_a, px - 0.03, cy_a)
    # one score per operation
    ox, oy, ow, oh = 2.66, 0.26, 1.72, 0.88
    rbox(ax, ox, oy, ow, oh, YEL)
    text(ax, ox + ow / 2, oy + 0.10, "One score per operation", 7.4, "bold", ha="center")
    rows = [("CVS stream", "highest score on a frame\nwithout CVS"),
            ("Time-out", "score at the last frame\nbefore the first clip"),
            ("Anatomy", "most confident wrong\nduct / artery label")]
    for k, (a, b) in enumerate(rows):
        yy = oy + 0.30 + k * 0.215
        rbox(ax, ox + 0.06, yy - 0.075, 0.56, 0.15, "white", ec=LINE, lw=0.5, r=0.03, z=2)
        text(ax, ox + 0.34, yy, a, 6.0, "bold", ha="center")
        text(ax, ox + 0.68, yy, b, 5.8, linespacing=1.05)
    arrow(ax, px + pw + 0.03, cy_a, ox - 0.03, cy_a)
    # order statistic of the real calibration maxima
    dx, dy, dw, dh = 4.55, 0.38, 0.90, 0.78
    sub = fig.add_axes([fx(dx), fy(dy + dh), fx(dw), dh / H]); sub.set_zorder(4)
    sub.patch.set_alpha(0); sub.grid(False)
    V = len(m); sub.bar(np.arange(V), m, width=0.78, color="#9DBCDD", ec="none", zorder=2)
    sub.bar([j - 1], [m[j - 1]], width=0.78, color="#0B4F8C", ec="none", zorder=3)
    sub.axhline(lam, c=INK, ls=(0, (2.2, 1.6)), lw=0.8, zorder=4)
    ymax = float(m.max()) * 1.25
    sub.set_xlim(-0.8, V - 0.2); sub.set_ylim(0, ymax); sub.set_xticks([]); sub.set_yticks([])
    for name, sp in sub.spines.items():
        sp.set_visible(name in ("left", "bottom")); sp.set_color(LINE); sp.set_linewidth(0.6)
    text(ax, dx + dw / 2, dy - 0.03, f"scores of the {V} calibration\noperations, sorted", 5.9, ha="center", color=MUTE,
         va="bottom", linespacing=1.05)
    text(ax, dx + 0.04, dy + dh * (1 - lam / ymax) - 0.06, f"$\\lambda$ = {lam:.2f}", 6.4, "bold")
    text(ax, dx + dw / 2, dy + dh + 0.08, "$\\lceil(V{+}1)(1{-}\\alpha)\\rceil$-th smallest", 5.9, ha="center", color=MUTE)
    arrow(ax, ox + ow + 0.03, cy_a, dx - 0.02, cy_a)
    # fixed thresholds
    tx, ty, tw, th = 5.80, 0.28, 1.02, 0.84
    rbox(ax, tx, ty, tw, th, ORA)
    text(ax, tx + tw / 2, ty + 0.10, "Fixed thresholds", 7.4, "bold", ha="center")
    for k, s in enumerate(["CVS stream  $\\lambda$", "time-out  $\\lambda_c$, $s_{(r)}$", "anatomy  $\\lambda_a$"]):
        rbox(ax, tx + 0.08, ty + 0.24 + k * 0.185, tw - 0.16, 0.145, "white", ec=LINE, lw=0.5, r=0.03, z=2)
        text(ax, tx + tw / 2, ty + 0.24 + k * 0.185 + 0.072, s, 6.1, ha="center")
    arrow(ax, dx + dw + 0.12, cy_a, tx - 0.03, cy_a)

    # ============================================================== dividers
    ax.plot([0.04, W - 0.04], [1.38, 1.38], c="#BBBBBB", lw=0.8, zorder=1)
    ax.plot([3.86, 3.86], [1.44, H - 0.04], c="#BBBBBB", lw=0.8, zorder=1)

    # ============================================================== (b) online scoring (bottom left)
    yb = 1.50
    text(ax, 0.06, yb, "(b) Online: score a new operation (labels unavailable)", 8.0, "bold")
    # the new operation: three of its frames (20 s apart, from new_op_strip.png), stacked like the labeled ones
    strip = Image.open(ASSETS / "new_op_strip.png").convert("RGB"); fw5 = strip.width / 5
    sx, sy, sw = 0.06, yb + 0.20, 0.80
    for k, i in enumerate([4, 2, 0]):
        off = (2 - k) * 0.06
        sub_im = strip.crop((int(i * fw5), 0, int((i + 1) * fw5), strip.height))
        sh = image(fig, sub_im, sx + off, sy + off, sw, border="white", z=3 + k)
    text(ax, sx, sy + 0.12 + sh + 0.08, "New operation\n(live video, past frames\nonly; no labels)", 6.2, color=MUTE,
         va="top", linespacing=1.1)
    bx, by, bw, bh = 1.22, yb + 0.20, 0.92, 0.62
    rbox(ax, bx, by, bw, bh, BLU)
    text(ax, bx + bw / 2, by + 0.10, "Perception model", 7.0, "bold", ha="center")
    text(ax, bx + bw / 2, by + 0.22, "same as in (a), frozen", 5.9, ha="center", color=MUTE)
    nn_icon(ax, bx + bw / 2 - 0.15, by + 0.30, 0.30)
    arrow(ax, sx + sw + 0.14, by + bh / 2, bx - 0.03, by + bh / 2)
    # outputs of the model: CVS score, anatomy candidates
    cx, cy, cw, ch = 2.30, yb + 0.20, 1.10, 0.62
    rbox(ax, cx, cy, cw, ch, "white", ec=LINE, lw=0.6)
    text(ax, cx + 0.07, cy + 0.10, "CVS score", 6.6, "bold")
    for k, (lab, v) in enumerate(zip(["C1", "C2", "C3"], [0.92, 0.85, 0.58])):
        yy = cy + 0.23 + k * 0.095
        text(ax, cx + 0.08, yy, lab, 5.6)
        ax.add_patch(Rectangle((cx + 0.26, yy - 0.03), 0.76, 0.06, fc="#E6E6E6", ec="none", zorder=4))
        ax.add_patch(Rectangle((cx + 0.26, yy - 0.03), 0.76 * v, 0.06, fc="#0B4F8C", ec="none", zorder=5))
    text(ax, cx + 0.07, cy + ch - 0.08, "$p(x)=\\hat P(\\mathrm{C1}\\wedge\\mathrm{C2}\\wedge\\mathrm{C3}\\mid x)$", 5.6)
    arrow(ax, bx + bw + 0.03, by + 0.20, cx - 0.03, cy + 0.20)
    ay = yb + 0.92; aw = 0.56
    ah = image(fig, ASSETS / "anatomy_detector_output.png", cx, ay, aw, crop=(0.08, 0.0, 0.92, 1.0), border=LINE)
    text(ax, cx + aw + 0.06, ay + ah / 2, "anatomy candidates:\nregions and\nlabel scores", 6.0, linespacing=1.1)
    arrow(ax, bx + bw + 0.03, by + 0.44, cx - 0.03, ay + ah / 2)

    # ============================================================== (c) certification (bottom right)
    xc, yc = 3.98, 1.50
    text(ax, xc, yc, "(c) Certify a statement or defer", 8.0, "bold")
    fxx, fyy, fw_, fh_ = xc, yc + 0.22, 1.05, 0.50
    rbox(ax, fxx, fyy, fw_, fh_, YEL)
    text(ax, fxx + fw_ / 2, fyy + 0.11, "Compare with threshold", 6.6, "bold", ha="center")
    text(ax, fxx + fw_ / 2, fyy + 0.27, "issue iff  score $>\\lambda$", 6.6, ha="center")
    text(ax, fxx + fw_ / 2, fyy + 0.40, "else defer to the surgeon", 5.9, ha="center", color=MUTE)
    arrow(ax, 3.82, fyy + fh_ / 2, fxx - 0.03, fyy + fh_ / 2, lw=0.9)
    # fixed thresholds from (a): dashed elbow into the comparison box
    ex, ey = tx + tw / 2, fyy - 0.09
    ax.plot([ex, ex, fxx + fw_ / 2], [ty + th + 0.02, ey, ey], c=MUTE, lw=0.8, ls=(0, (3, 2)), zorder=6)
    arrow(ax, fxx + fw_ / 2, ey, fxx + fw_ / 2, fyy - 0.01, ls=(0, (3, 2)), lw=0.8, ms=6, c=MUTE)
    text(ax, ex - 0.05, 1.28, "fixed thresholds", 5.9, color=MUTE, ha="right")
    # three outputs
    ox2, ow2, oh2 = xc + 1.30, 1.56, 0.33
    outs = [(ORA, "During the operation", "CVS: confirm  /  defer", "check"),
            (BLU, "At the time-out", "confirm  /  STOP  /  defer", "clock"),
            (GRN, "Anatomy labels", "cystic duct / cystic artery / defer", "tag")]
    for k, (col, a, b, ic) in enumerate(outs):
        yy = yc + 0.20 + k * 0.40
        rbox(ax, ox2, yy, ow2, oh2, col)
        text(ax, ox2 + 0.30, yy + 0.11, a, 6.6, "bold")
        text(ax, ox2 + 0.30, yy + 0.24, b, 5.9)
        (check_icon if ic == "check" else clock_icon if ic == "clock" else tag_icon)(ax, ox2 + 0.15, yy + oh2 / 2)
        arrow(ax, fxx + fw_ + 0.03, fyy + fh_ / 2, ox2 - 0.03, yy + oh2 / 2, lw=0.8, ms=6)
    text(ax, xc, yc + 0.20 + 3 * 0.40 - 0.01, "Guarantee: a false statement on $\\leq\\alpha$ of operations, not of frames", 6.2,
         style="italic")

    ST.save(fig, OUT / "fig_overview_v2")
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(OUT / "fig_overview_v2.svg")
    print(f"calibration operations {len(m)}, j = {j}, lambda = {lam:.3f}")


if __name__ == "__main__":
    main()
