"""One visual style for every figure of the paper and supplement.

Palette validated with scripts/figures/validate_palette.py (OKLab; Machado-2009 protan/deutan
adjacent-pair Delta E >= 8, normal-vision >= 15, WCAG contrast >= 3 on white):
    ours  #2a78d6   frame-level CRC  #d9531e   Learn-then-Test  #138a60   other  #6250d6
Point world is a neutral grey dashed line (not a categorical hue). Identity is never colour alone:
every method also has its own marker. Status tints: violation region #fbe7e3, nominal line black.
Sizes are for print: double-column figures 6.875 in wide, single-column 3.25 in; text 7-8 pt
(captions are 9 pt), lines 1.2 pt, markers 4 pt.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OURS, CRC, LTT, OTHER, POINT = "#2a78d6", "#d9531e", "#138a60", "#6250d6", "#6f6e69"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
VIOLATE = "#fdf0ec"          # region above the nominal level
OURS_TINT, CRC_TINT = "#d7e7fa", "#fbe0d4"
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
    """Diagonal y = alpha and the shaded violation region above it."""
    ax.fill_between([0, xmax], [0, xmax], [1.05, 1.05], color=VIOLATE, lw=0, zorder=0)
    ax.plot([0, xmax], [0, xmax], ls=(0, (2, 1.5)), c=INK, lw=0.8, zorder=1, label="nominal level" if label else None)


def save(fig, path):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
