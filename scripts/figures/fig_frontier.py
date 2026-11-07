"""Risk-utility frontier: per-operation false-confirmation rate (x) vs sensitivity (y), one
curve per method across alpha in {0.05, 0.10, 0.20}; Endoscapes, DINOv2-L, mean over
200 splits x 3 seeds. Methods controlling different quantities become comparable here.

    python scripts/figures/fig_frontier.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ST  # noqa: E402

ST.setup()
G = "#9a9893"   # neutral grey for secondary baselines (identity by marker and label)
STYLE = {  # method: (label, colour, marker, linestyle, z)
    "Surgery-level + smoothing (ours)": ("Per operation + smoothing (ours)", ST.OURS, "o", "-", 6),
    "Surgery-level (ours)": ("Per operation (ours)", ST.OURS, "o", (0, (3, 1.5)), 5),
    "CRC (= frame-level)": ("Frame-level CRC", ST.CRC, "s", "-", 4),
    "CRC + smoothing (10 s)": ("Frame-level CRC + smoothing", ST.CRC, "s", (0, (3, 1.5)), 4),
    "LTT": ("Frame-level LTT", ST.LTT, "^", "-", 3),
    "Conformal selection (BH)": ("Conformal selection", ST.OTHER, "D", "-", 3),
    "One frame / video (Dunn)": ("One frame / video", G, "v", "-", 2),
    "ACI (label feedback)": ("ACI", G, "x", (0, (1, 1)), 2),
}


def main(arch="dinov2l_b8"):
    d = pd.read_csv(ROOT / "experiments" / "compare_all" / "compare_all.csv")
    d = d[d.run.str.startswith(arch)]
    fig, ax = plt.subplots(figsize=(ST.SINGLE, 2.6))
    for m, (lab, c, mk, ls, z) in STYLE.items():
        g = d[d.method == m].groupby("alpha")[["ops_false_ge1", "sens"]].mean().sort_index()
        ax.plot(g.ops_false_ge1, g.sens, c=c, marker=mk, ls=ls, mec="white", mew=0.5, label=lab, zorder=z)
    mp = d[d.method == "Point world (MAP)"][["ops_false_ge1", "sens"]].mean()
    ax.scatter([mp.ops_false_ge1], [mp.sens], marker="*", s=60, c=ST.POINT, zorder=7, label="Point world")
    for a in (0.05, 0.10, 0.20):
        ax.axvline(a, c=ST.MUTED, lw=0.5, ls=":", zorder=0)
        ax.text(a, 0.97, f"{a:.2f}".lstrip("0"), ha="center", va="bottom", fontsize=6, color=ST.MUTED)
    ax.set_xlabel("operations with ≥1 false CVS confirmation (%)")
    ax.set_ylabel("sensitivity (%)")
    ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.95)
    ax.set_xticks([0, .2, .4, .6, .8]); ax.set_xticklabels(["0", "20", "40", "60", "80"])
    ax.set_yticks([0, .2, .4, .6, .8]); ax.set_yticklabels(["0", "20", "40", "60", "80"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, columnspacing=0.8, labelspacing=0.3,
              fontsize=6.3)
    fig.tight_layout(pad=0.2)
    ST.save(fig, OUT / "fig_frontier")
    print("wrote fig_frontier")


if __name__ == "__main__":
    main()
