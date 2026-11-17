"""Architecture overview (main Fig. 2), icon style, dense: the complete pipeline: rounded groups, a trapezoid
encoder (frozen), 3-D tensor glyphs, a diamond comparator, pill outputs and a legend row.

Top group (dashed arrows = calibration, offline): V labeled operations -> perception model E ->
per-frame scores p(x) with the per-operation statistic m_v (its highest score on a frame without CVS;
real values of three calibration operations) -> order statistic of the V real calibration maxima
(official Endoscapes split, DINOv2-L seed 0, 10 s smoothing, alpha = 0.10) -> thresholds.
Bottom group (solid arrows = online): a new operation -> the same E -> score p_t and anatomy
candidates A_t (our Faster R-CNN on a held-out test frame) -> comparison with the fixed threshold ->
confirm / stop / defer for the three statements; bound P(error per operation) <= alpha.

Frames: paper/figures/arch_assets (Endoscapes2023, CC BY-NC-SA 4.0; see SOURCE.txt).
Writes paper/figures/fig_overview_v4.{pdf,png,svg}.

    python scripts/figures/fig_overview_v4.py
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
W, H = ST.DOUBLE, 3.24

ENC = "#92D4EE"; ENC_E = "#2F6F8F"
BLU = ("#DCE8F5", "#A9C6E6", "#6E9FD1")          # tensor cards, light -> dark
GRN = ("#C9F2BC", "#9FE090")                    # bar front, side
ORA = "#F9E3C7"; YEL = "#FFF8CF"; GREY = "#BBBBBB"
INK, MUTE, LINE = "#111111", "#555555", "#222222"
OK, STOP, DEF = "#2E9E5B", "#C8440A", "#8C8C8C"
plt.rcParams.update({"font.size": 7.5, "text.color": INK, "mathtext.fontset": "cm"})


# ----------------------------------------------------------------------------- helpers (inches, y down)
def fx(x): return x / W
def fy(y): return 1 - y / H


def rbox(ax, x, y, w, h, fc, ec="none", lw=0.8, r=0.06, z=1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z, linestyle=ls))


def arrow(ax, x0, y0, x1, y1, c=LINE, lw=0.9, ls="-", ms=7, z=6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw,
                                 linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, size=7.5, weight="normal", color=INK, ha="center", va="center", z=8, **kw):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z, **kw)


def image(fig, im, x, y, w, border=None, z=3):
    im = im.convert("RGB"); im.thumbnail((900, 900)); h = w * im.height / im.width
    sub = fig.add_axes([fx(x), fy(y + h), fx(w), h / H], zorder=z)
    sub.imshow(np.asarray(im), aspect="auto"); sub.set_xticks([]); sub.set_yticks([]); sub.grid(False)
    for sp in sub.spines.values():
        sp.set_visible(border is not None)
        if border:
            sp.set_edgecolor(border); sp.set_linewidth(0.8)
    return h


def stack(fig, ims, x, y, w, off=0.05):
    for k, im in enumerate(ims):
        o = (len(ims) - 1 - k) * off
        h = image(fig, im, x + o, y + o, w, border="white", z=3 + k)
    return h + (len(ims) - 1) * off


def encoder(ax, x, y, w, h, label=True):
    """Trapezoid encoder glyph, wide at the left."""
    ax.add_patch(Polygon([(x, y), (x + w, y + 0.22 * h), (x + w, y + 0.78 * h), (x, y + h)], closed=True, fc=ENC,
                         ec=ENC_E, lw=0.6, zorder=4))
    if label:
        text(ax, x + w / 2, y + h / 2, r"$\mathcal{E}$", 11)


def snowflake(ax, cx, cy, r=0.055, c="#3B7DD8"):
    for k in range(6):
        a = k * np.pi / 3
        ex, ey = cx + r * np.cos(a), cy + r * np.sin(a)
        ax.plot([cx, ex], [cy, ey], c=c, lw=0.8, zorder=9, solid_capstyle="round")
        for s in (1, -1):
            bx, by = cx + 0.6 * r * np.cos(a), cy + 0.6 * r * np.sin(a)
            ax.plot([bx, bx + 0.3 * r * np.cos(a + s * np.pi / 3)], [by, by + 0.3 * r * np.sin(a + s * np.pi / 3)],
                    c=c, lw=0.7, zorder=9, solid_capstyle="round")


def bar3d(ax, x, y, w, h, cf, cs, d=0.04, z=4):
    """Horizontal 3-D bar with its front face top-left at (x, y)."""
    ax.add_patch(Rectangle((x, y), w, h, fc=cf, ec="none", zorder=z))
    ax.add_patch(Polygon([(x, y), (x + d, y - d), (x + w + d, y - d), (x + w, y)], closed=True, fc=cs, ec="none", zorder=z))
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y - d), (x + w + d, y + h - d), (x + w, y + h)], closed=True,
                         fc=cs, ec="none", zorder=z))


def cards(ax, x, y, w, h, cols=BLU, off=0.05, z=4):
    """Three offset tensor cards (deepest first)."""
    for k, c in enumerate(cols):
        o = (len(cols) - 1 - k) * off
        ax.add_patch(Rectangle((x + o, y - o), w, h, fc=c, ec="white", lw=0.5, zorder=z + k))


def diamond(ax, cx, cy, w, h, fc=YEL):
    ax.add_patch(Polygon([(cx - w / 2, cy), (cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2)], closed=True, fc=fc,
                         ec=LINE, lw=0.7, zorder=4))


def clock_icon(ax, cx, cy, r=0.06, c=INK):
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=c, lw=0.8, zorder=8))
    ax.plot([cx, cx], [cy, cy - r * 0.62], c=c, lw=0.8, zorder=9); ax.plot([cx, cx + r * 0.5], [cy, cy], c=c, lw=0.8, zorder=9)


def check_icon(ax, cx, cy, r=0.06, fc=OK):
    ax.add_patch(Circle((cx, cy), r, fc=fc, ec="none", zorder=8))
    ax.plot([cx - r * 0.45, cx - r * 0.1, cx + r * 0.5], [cy, cy + r * 0.4, cy - r * 0.4], c="white", lw=1.1, zorder=9,
            solid_capstyle="round")


def tag_icon(ax, cx, cy, w=0.14, h=0.09, c=INK):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="white", ec=c, lw=0.7, zorder=8))
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2 - 0.03), w * 0.55, 0.03, fc=c, ec="none", zorder=8))


def pill(ax, x, y, w, h, s, fc, color="white"):
    rbox(ax, x, y, w, h, fc, r=h / 2, z=4); text(ax, x + w / 2, y + h / 2, s, 6.6, "bold", color=color)


# ----------------------------------------------------------------------------- data
def calibration():
    te = set(keyframes("test").vid.tolist())
    ops = endoscapes_ops("dinov2l_b8", 0, (SMOOTH,))
    cal = [o for o in ops if o["vid"] not in te and (~o["cvs"]).any()]
    m = np.sort([o["p"][SMOOTH][~o["cvs"]].max() for o in cal])
    j = math.ceil((len(m) + 1) * (1 - ALPHA))
    # three operations for the m_v glyph: 6 evenly spaced keyframes each
    srt = sorted(cal, key=lambda o: o["p"][SMOOTH][~o["cvs"]].max())
    show = [srt[-1], srt[len(srt) // 2], srt[len(srt) // 5]]
    rows = []
    for o in show:
        p, c = o["p"][SMOOTH], o["cvs"]
        idx = np.linspace(0, len(p) - 1, 6).round().astype(int)
        top = np.flatnonzero(~c)[np.argmax(p[~c])]
        if top not in idx:
            idx[np.argmin(np.abs(idx - top))] = top
        idx = np.sort(idx); rows.append((p[idx], c[idx], int(np.flatnonzero(idx == top)[0])))
    return m, j, float(m[j - 1]), rows


def main():
    m, j, lam, rows = calibration()
    plt.rcParams["savefig.bbox"] = None
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(H, 0)
    DASH = (0, (3, 2)); S, XS = 6.2, 5.7                   # small / extra-small text

    # ================================================================= calibration (top band)
    rbox(ax, 0.06, 0.06, 6.76, 1.42, "white", ec=GREY, lw=0.9, r=0.14, z=0)
    text(ax, 0.16, 0.19, "Calibration", 8, "bold", ha="left", style="italic")
    text(ax, 0.86, 0.19, r"offline, $V$ labeled operations, level $\alpha$", 6.5, color=MUTE, ha="left")
    cy = 0.82
    past = [Image.open(ASSETS / f"past_op_{k}_labelled.png") for k in (3, 2, 1)]
    sh = stack(fig, past, 0.14, 0.38, 0.60)
    text(ax, 0.49, 0.38 + sh + 0.10, "surgeons' CVS rating,\nduct / artery boxes", XS, color=MUTE, linespacing=1.05)
    arrow(ax, 0.90, cy, 1.00, cy, ls=DASH)
    # perception model with its chain
    px, py, pw, ph = 1.02, 0.36, 1.56, 0.92
    rbox(ax, px, py, pw, ph, "#EAF4FA", ec=ENC_E, lw=0.5, r=0.06)
    text(ax, px + pw / 2, py + 0.10, "Perception model", 6.8, "bold")
    encoder(ax, px + 0.08, py + 0.26, 0.26, 0.50); snowflake(ax, px + 0.07, py + 0.22)
    lines = [r"DINOv2-L $\to$ C1, C2, C3", r"causal 10 s smoothing $\to$ $p(x)$", r"Faster R-CNN $\to$ regions $A$"]
    for k, s in enumerate(lines):
        text(ax, px + 0.42, py + 0.33 + k * 0.18, s, XS, ha="left")
    text(ax, px + 0.42, py + 0.33 + 3 * 0.18, r"$p=\hat P(\mathrm{C1}\wedge\mathrm{C2}\wedge\mathrm{C3}\mid x)$", XS, ha="left", color=MUTE)
    arrow(ax, px + pw + 0.02, cy, px + pw + 0.12, cy, ls=DASH)
    # per-frame scores of three real operations, m_v outlined
    gx, cw, gap = 2.72, 0.075, 0.01
    gy = cy - (3 * cw + 2 * 0.04) / 2
    for r, (p, c, top) in enumerate(rows):
        yy = gy + r * (cw + 0.04)
        for k, (pv, cv) in enumerate(zip(p, c)):
            xx = gx + k * (cw + gap)
            col = plt.cm.Blues(0.15 + 0.75 * pv) if not cv else "#BFE6C3"
            ax.add_patch(Rectangle((xx, yy), cw, cw, fc=col, ec="none", zorder=4))
            if k == top:
                ax.add_patch(Rectangle((xx - 0.008, yy - 0.008), cw + 0.016, cw + 0.016, fill=False, ec=INK, lw=0.9, zorder=6))
        text(ax, gx + 6 * (cw + gap) + 0.04, yy + cw / 2, r"$m_v$", 6.5, ha="left")
    text(ax, gx + 0.38, gy + 3 * (cw + 0.04) + 0.06, r"$p(x)$ per frame", XS, color=MUTE)
    text(ax, gx + 0.38, gy - 0.08, "green = CVS frames", XS, color=MUTE)
    arrow(ax, 3.50, cy, 3.60, cy, ls=DASH)
    # one score per operation: the three statistics
    sx_, sy_, sw_, sh_ = 3.62, 0.36, 1.40, 0.92
    rbox(ax, sx_, sy_, sw_, sh_, YEL, r=0.06)
    text(ax, sx_ + sw_ / 2, sy_ + 0.10, "One score per operation", 6.8, "bold")
    stats = [(check_icon, r"CVS stream: $m_v=\max\,p$ on frames", "without CVS (any false confirmation)"),
             (clock_icon, r"time-out: $m_v=p$ at the last frame", "before the first clip"),
             (tag_icon, r"anatomy: $m_v=$ most confident", "wrong duct / artery score")]
    for k, (ic, a, b) in enumerate(stats):
        yy = sy_ + 0.29 + k * 0.23
        ic(ax, sx_ + 0.12, yy + 0.05, **({} if ic is tag_icon else {"r": 0.05}))
        text(ax, sx_ + 0.24, yy - 0.01, a, XS, ha="left"); text(ax, sx_ + 0.24, yy + 0.10, b, XS, ha="left", color=MUTE)
    arrow(ax, sx_ + sw_ + 0.02, cy, sx_ + sw_ + 0.12, cy, ls=DASH)
    # order statistic of the real calibration maxima
    dx, dy, dw, dh = 5.16, 0.40, 0.60, 0.74
    sub = fig.add_axes([fx(dx), fy(dy + dh), fx(dw), dh / H]); sub.set_zorder(4); sub.patch.set_alpha(0); sub.grid(False)
    V = len(m); ymax = float(m.max()) * 1.35
    sub.bar(np.arange(V), m, width=0.78, color=BLU[1], ec="none", zorder=2)
    sub.bar([j - 1], [m[j - 1]], width=0.78, color=BLU[2], ec="none", zorder=3)
    sub.axhline(lam, c=INK, ls=(0, (2.2, 1.6)), lw=0.8, zorder=4)
    sub.set_xlim(-0.8, V - 0.2); sub.set_ylim(0, ymax); sub.set_xticks([]); sub.set_yticks([])
    for name, sp in sub.spines.items():
        sp.set_visible(name in ("left", "bottom")); sp.set_color(LINE); sp.set_linewidth(0.6)
    text(ax, dx + 0.04, dy + dh * (1 - lam / ymax) - 0.07, r"$m_{(j)}$", 6.5, ha="left")
    text(ax, dx + dw / 2, dy + dh + 0.08, r"$j=\lceil(V{+}1)(1{-}\alpha)\rceil$", XS, color=MUTE)
    text(ax, dx + dw / 2, dy - 0.06, f"{V} sorted $m_v$", XS, color=MUTE)
    arrow(ax, dx + dw + 0.03, cy, dx + dw + 0.13, cy, ls=DASH)
    # thresholds
    tx, ty, tw, th = 5.92, 0.36, 0.84, 0.92
    rbox(ax, tx, ty, tw, th, ORA, r=0.06)
    text(ax, tx + tw / 2, ty + 0.10, "Thresholds", 6.8, "bold")
    for k, s in enumerate([r"$\hat\lambda_{\rm CVS}=m_{(j)}$", r"$\hat\lambda_c,\ s_{(r)}$", r"$\hat\lambda_{\rm anat}$"]):
        rbox(ax, tx + 0.07, ty + 0.22 + k * 0.22, tw - 0.14, 0.17, "white", ec=LINE, lw=0.4, r=0.03, z=2)
        text(ax, tx + tw / 2, ty + 0.22 + k * 0.22 + 0.085, s, 6.3)
    # thresholds -> comparators (dashed, in the gap between the bands)
    ex = tx + tw / 2; cxm = 3.14
    ax.plot([ex, ex, cxm], [ty + th + 0.02, 1.56, 1.56], c=MUTE, lw=0.8, ls=DASH, zorder=6)
    arrow(ax, cxm, 1.56, cxm, 1.80, ls=DASH, lw=0.8, ms=6, c=MUTE)
    text(ax, 4.5, 1.49, "fixed thresholds", XS, color=MUTE)

    # ================================================================= certification (bottom band)
    rbox(ax, 0.06, 1.64, 6.76, 1.22, "white", ec=GREY, lw=0.9, r=0.14, z=0)
    text(ax, 0.16, 1.77, "Certification", 8, "bold", ha="left", style="italic")
    text(ax, 0.98, 1.77, "online, a new operation, no labels", 6.5, color=MUTE, ha="left")
    cb = 2.30
    strip = Image.open(ASSETS / "new_op_strip.png").convert("RGB"); f5 = strip.width / 5
    new = [strip.crop((int(i * f5), 0, int((i + 1) * f5), strip.height)) for i in (4, 2, 0)]
    sh = stack(fig, new, 0.14, 1.92, 0.60)
    text(ax, 0.49, 1.92 + sh + 0.10, r"frames $x_1,\dots,x_t$" + "\n(past frames only)", XS, color=MUTE, linespacing=1.05)
    arrow(ax, 0.90, cb, 1.00, cb)
    encoder(ax, 1.04, cb - 0.26, 0.26, 0.52); snowflake(ax, 1.03, cb - 0.30)
    text(ax, 1.17, cb + 0.36, "same model", XS, color=MUTE)
    # model outputs: score, anatomy candidates, time-out frame
    bar3d(ax, 1.50, 1.92, 0.44, 0.09, GRN[0], GRN[1]); text(ax, 2.00, 1.965, r"$p_t$", 6.5, ha="left")
    ah = image(fig, Image.open(ASSETS / "anatomy_detector_output.png").crop((52, 0, 597, 480)), 1.50, 2.10, 0.42, border=LINE)
    text(ax, 1.98, 2.10 + ah / 2, r"$A_t$", 6.5, ha="left")
    clock_icon(ax, 1.56, 2.66, r=0.05); text(ax, 1.66, 2.66, r"last $p$ before clip: $p_v$", XS, ha="left")
    arrow(ax, 1.32, cb - 0.10, 1.46, 1.965); arrow(ax, 1.32, cb + 0.06, 1.46, 2.10 + ah / 2); arrow(ax, 1.32, cb + 0.20, 1.46, 2.66)
    # per-statement comparators
    comps = [r"$p_t>\hat\lambda_{\rm CVS}$", r"$p_v>\hat\lambda_c$ / $p_v<s_{(r)}$", r"$s>\hat\lambda_{\rm anat}$"]
    ys = [1.96, 2.33, 2.70]
    for k, (s, yy) in enumerate(zip(comps, ys)):
        rbox(ax, 2.62, yy - 0.10, 1.04, 0.20, YEL, ec=LINE, lw=0.5, r=0.10, z=4)
        text(ax, 3.14, yy, s, 6.0)
        arrow(ax, 2.46 if k == 0 else 2.46, [1.965, 2.10 + ah / 2, 2.66][k], 2.60, yy, lw=0.7, ms=5)
        arrow(ax, 3.68, yy, 3.80, yy, lw=0.8, ms=6)
    # outcomes
    icons = [check_icon, clock_icon, tag_icon]
    outs = [[("confirm", OK), ("defer", DEF)], [("confirm", OK), ("stop", STOP), ("defer", DEF)], [("duct", OK), ("artery", OK), ("defer", DEF)]]
    for k, (ic, o, yy) in enumerate(zip(icons, outs, ys)):
        ic(ax, 3.92, yy)
        for q, (s, c) in enumerate(o):
            pill(ax, 4.06 + q * 0.40, yy - 0.09, 0.36, 0.18, s, c)
    text(ax, 3.84, 1.80, "issue, else defer to the surgeon", XS, color=MUTE, ha="left")
    # guarantee block
    gx_, gy_, gw_, gh_ = 5.30, 1.86, 1.46, 0.98
    rbox(ax, gx_, gy_, gw_, gh_, YEL, ec=LINE, lw=0.7, r=0.05, ls=DASH)
    text(ax, gx_ + gw_ / 2, gy_ + 0.10, "Guarantee", 6.6, "bold")
    text(ax, gx_ + gw_ / 2, gy_ + 0.22, "operations exchangeable", XS, color=MUTE)
    glines = [r"$P(\mathrm{false\ statement}\mid\mathrm{operation})\leq\alpha$",
              r"risk $R\sim\mathrm{Beta}(V{+}1{-}j,\,j)$",
              r"time-out: $P(\mathrm{confirm}\mid\neg\mathrm{CVS})\leq\alpha_c$",
              r"$\quad\quad\ \ P(\mathrm{stop}\mid\mathrm{CVS})\leq\alpha_s$",
              r"clip: anatomy $\wedge$ CVS, $\alpha/2$ each"]
    for k, s in enumerate(glines):
        text(ax, gx_ + 0.06, gy_ + 0.36 + k * 0.125, s, 5.4, ha="left")

    # ================================================================= legend
    ly = 3.06
    items = [("snow", "Frozen"), ("enc", "Perception"), ("ord", "Order statistic"), ("lam", "Threshold"),
             ("chk", "CVS"), ("clk", "Time-out"), ("tag", "Anatomy"), ("fwd", "Online"), ("cal", "Offline")]
    x = 0.10; LW = 0.051
    for k, lab in items:
        iw = 0.30 if k in ("fwd", "cal") else 0.20
        if k == "snow":
            snowflake(ax, x + 0.06, ly)
        elif k == "enc":
            encoder(ax, x, ly - 0.11, 0.16, 0.22, label=False); text(ax, x + 0.08, ly, r"$\mathcal{E}$", 7)
        elif k == "ord":
            for i, hgt in enumerate([0.04, 0.07, 0.10, 0.14]):
                ax.add_patch(Rectangle((x + i * 0.04, ly + 0.08 - hgt), 0.032, hgt, fc=BLU[1] if i < 3 else BLU[2], zorder=4))
            ax.plot([x - 0.01, x + 0.17], [ly - 0.04, ly - 0.04], c=INK, lw=0.7, ls=(0, (2, 1.5)), zorder=5)
        elif k == "lam":
            rbox(ax, x, ly - 0.09, 0.18, 0.18, ORA, r=0.03, z=4); text(ax, x + 0.09, ly, r"$\hat\lambda$", 7)
        elif k == "chk":
            check_icon(ax, x + 0.06, ly)
        elif k == "clk":
            clock_icon(ax, x + 0.06, ly)
        elif k == "tag":
            tag_icon(ax, x + 0.07, ly)
        elif k == "fwd":
            arrow(ax, x, ly, x + 0.26, ly)
        else:
            arrow(ax, x, ly, x + 0.26, ly, ls=DASH)
        text(ax, x + iw, ly, lab, 6.5, ha="left")
        x += iw + len(lab) * LW + 0.11
    assert x <= W - 0.04, x

    ST.save(fig, OUT / "fig_overview_v4")
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(OUT / "fig_overview_v4.svg")
    print(f"V = {len(m)}, j = {j}, lambda = {lam:.3f}")


if __name__ == "__main__":
    main()
