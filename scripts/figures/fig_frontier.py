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
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.fontsize": 6.2})
STYLE = {  # method: (label, colour, marker, linestyle, z)
    "Surgery-level + smoothing (ours)": ("Surgery-level + smoothing (ours)", "#1f4e9c", "o", "-", 6),
    "Surgery-level (ours)": ("Surgery-level (ours)", "#6b8fd0", "o", "--", 5),
    "Conformal selection (BH)": ("Conformal selection (BH)", "#8e44ad", "D", "-", 4),
    "One frame / video (Dunn)": ("One frame / video (Dunn)", "#16a085", "v", "-", 3),
    "Repeated subsampling (Dunn, a/2)": ("Repeated subsampling (Dunn)", "#16a085", "^", ":", 3),
    "LTT": ("Learn-then-Test", "#3a9d5d", "^", "-", 3),
    "CRC (= frame-level)": ("CRC (frame-level)", "#e07b1a", "s", "-", 3),
    "CRC + smoothing (10 s)": ("CRC + smoothing", "#e07b1a", "s", "--", 3),
    "ACI (label feedback)": ("ACI (label feedback)", "#b9770e", "x", "-", 2),
}


def main(arch="dinov2l_b8"):
    d = pd.read_csv(ROOT / "experiments" / "compare_all" / "compare_all.csv")
    d = d[d.run.str.startswith(arch)]
    fig, ax = plt.subplots(figsize=(3.3, 2.55))
    for m, (lab, c, mk, ls, z) in STYLE.items():
        g = d[d.method == m].groupby("alpha")[["ops_false_ge1", "sens"]].mean().sort_index()
        ax.plot(g.ops_false_ge1, g.sens, c=c, marker=mk, ls=ls, lw=1.1, ms=3.6, label=lab, zorder=z)
    mp = d[d.method == "Point world (MAP)"][["ops_false_ge1", "sens"]].mean()
    ax.scatter([mp.ops_false_ge1], [mp.sens], marker="*", s=60, c="#555", zorder=7, label="Point world (MAP)")
    for a in (0.05, 0.10, 0.20):
        ax.axvline(a, c="#cccccc", lw=0.6, ls=":", zorder=0)
    ax.set_xlabel("operations with ≥1 false CVS confirmation")
    ax.set_ylabel("sensitivity (CVS frames confirmed)")
    ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.95)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, frameon=False, handlelength=1.8,
              columnspacing=0.8, labelspacing=0.25, fontsize=5.9)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_frontier.{ext}", dpi=300, bbox_inches="tight")
    print("wrote fig_frontier")


if __name__ == "__main__":
    main()
