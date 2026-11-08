"""One visual style for every figure of the paper and supplement.

Palette (muted; checked with scripts/figures/validate_palette.py, OKLab + Machado-2009 CVD simulation:
every adjacent method pair has protan/deutan Delta E >= 8; the purple and olive sit below the
validator's chroma rule, which is stylistic, not an accessibility criterion):
    ours  #204A73 (navy)   frame-level CRC  #C47654 (terracotta)   Learn-then-Test  #9380B5 (purple)
    other  #556B2F (olive)   point world  #92979E (grey, long dashes)
Identity is never colour alone: every method also has its own marker and line style, so the figures
read in grayscale. No coloured panel fills: the nominal target is a thin dotted charcoal line
labelled "target"; our method is drawn heaviest (2.2 pt), baselines at 1.5 pt, whiskers 0.8 pt in a
lighter tint of the method colour.
Sizes are for print: double-column figures 6.875 in wide, single-column 3.25 in; text 7-8 pt
(captions are 9 pt), lines 1.2 pt, markers 4 pt.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OURS, CRC, LTT, OTHER, POINT = "#204A73", "#C47654", "#9380B5", "#556B2F", "#92979E"
INK, MUTED, GRID = "#30343B", "#5c6068", "#E9EDF1"
VIOLATE = "#ffffff"          # (no longer used) former fill above the nominal level; panels stay white
OURS_TINT, CRC_TINT = "#E4E9EE", "#F3E4DD"


def tint(hex_colour, amount=0.45):
    """Mix a colour with white: amount = 1 keeps the colour, 0 gives white."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(round(255 - amount * (255 - v))) for v in (r, g, b))
METHODS = {  # key -> (label, colour, marker)
    "ours": ("Per operation (ours)", OURS, "o"),
    "crc": ("Frame-level CRC", CRC, "s"),
    "ltt": ("Frame-level LTT", LTT, "^"),
    "other": ("Other", OTHER, "D"),
    "point": ("Point world", POINT, None),
}
DOUBLE, SINGLE = 6.875, 3.25


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "lines.linewidth": 1.2, "lines.markersize": 4, "errorbar.capsize": 1.8,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "figure.dpi": 150,
    })


def nominal(ax, xmax, label=True):
    """Diagonal target y = alpha (thin dotted charcoal line, no fill)."""
    ax.plot([0, xmax], [0, xmax], ls=(0, (1.2, 1.4)), c=INK, lw=0.8, zorder=1, label="target (nominal level)" if label else None)


def save(fig, path):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
