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
    CAL = (0.05, 0.10, 0.20)                       # calibrated operating points; other alphas are a descriptive sweep
    fig, axes = plt.subplots(1, 2, figsize=(ST.DOUBLE, 2.5), gridspec_kw=dict(width_ratios=[1, 1]))
    for ax, xmax, ymax, title in ((axes[0], 0.9, 0.95, "a  full range"), (axes[1], 0.25, 0.45, "b  low-risk region (zoom)")):
        for m, (lab, c, mk, ls, z) in STYLE.items():
            g = d[d.method == m].groupby("alpha")[["ops_false_ge1", "sens"]].mean().sort_index()
            ax.plot(g.ops_false_ge1, g.sens, c=c, ls=ls, lw=1.0, zorder=z, label=lab if ax is axes[0] else None)
            cal = g[g.index.isin(CAL)]; swp = g[~g.index.isin(CAL)]
            ax.scatter(cal.ops_false_ge1, cal.sens, marker=mk, s=22, c=c, ec="white", lw=0.5, zorder=z + 1)
            ax.scatter(swp.ops_false_ge1, swp.sens, marker=mk, s=18, facecolors="white", edgecolors=c, lw=0.8, zorder=z + 1)
        mp = d[d.method == "Point world (MAP)"][["ops_false_ge1", "sens"]].mean()
        ax.scatter([mp.ops_false_ge1], [mp.sens], marker="*", s=60, c=ST.POINT, zorder=7, label="Point world" if ax is axes[0] else None)
        for a_ in CAL:
            if a_ <= xmax:
                ax.axvline(a_, c=ST.MUTED, lw=0.5, ls=":", zorder=0)
                if ax is axes[1]:
                    ax.text(a_, ymax * 0.98, f" α = {a_:.2f}".replace("0.", "."), ha="left", va="top", fontsize=6, color=ST.MUTED)
        ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.set_title(title)
        ax.set_xlabel("operations with ≥1 false CVS confirmation (% of 36 test operations)")
        ax.set_ylabel("sensitivity (% of CVS keyframes confirmed)")
        xt = [x for x in (0, .1, .2, .4, .6, .8) if x <= xmax]; ax.set_xticks(xt); ax.set_xticklabels([f"{100 * x:.0f}" for x in xt])
        yt = [y for y in (0, .2, .4, .6, .8) if y <= ymax]; ax.set_yticks(yt); ax.set_yticklabels([f"{100 * y:.0f}" for y in yt])
    axes[0].scatter([], [], marker="o", s=22, c=ST.MUTED, ec="white", label="calibrated operating point (α = .05, .10, .20)")
    axes[0].scatter([], [], marker="o", s=18, facecolors="white", edgecolors=ST.MUTED, label="descriptive sweep (other α)")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, columnspacing=1.0, handletextpad=0.4, fontsize=6.3)
    fig.tight_layout()
    ST.save(fig, OUT / "fig_frontier")
    print("wrote fig_frontier")


if __name__ == "__main__":
    main()
