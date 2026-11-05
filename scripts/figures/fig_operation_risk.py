"""Main figure: fraction of operations with >= 1 false certificate vs nominal alpha.

Panels: (a) Endoscapes CVS confirmation, (b) Endoscapes duct/artery identification,
(c) Cholec80 CVS confirmation after recalibration on 20 Cholec80 videos.
Points: mean over random video-level splits (and seeds); bars: 2.5-97.5 % over splits.
Full CVPR text width (6.875 in).

    python scripts/figures/fig_operation_risk.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"; OUT.mkdir(parents=True, exist_ok=True)

STYLE = {  # method label in the CSVs -> (legend, colour, marker)
    "surgery": ("Surgery-level (ours)", "#1f4e9c", "o"),
    "pooled": ("Frame-level, frames pooled (CRC)", "#e07b1a", "s"),
    "ltt": ("Learn-then-Test, frames pooled", "#3a9d5d", "^"),
    "map": ("Point world (MAP)", "#7a7a7a", None),
}
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 8,
                     "axes.labelsize": 7.5, "legend.fontsize": 6.8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False})


def series(df, col, method_map):
    out = {}
    for key, m in method_map.items():
        d = df[df.method == m]
        if key == "map":
            out[key] = (d[col].mean(), d[col].quantile(0.025), d[col].quantile(0.975))
            continue
        g = d.groupby("alpha")[col]
        out[key] = (g.mean(), g.quantile(0.025), g.quantile(0.975))
    return out


def panel(ax, S, title, alphas):
    ax.plot([0, 0.22], [0, 0.22], ls=":", c="k", lw=0.8, label="nominal  y = α")
    for key in ("pooled", "ltt", "surgery"):
        m, lo, hi = S[key]
        m, lo, hi = m.loc[alphas], lo.loc[alphas], hi.loc[alphas]
        lab, c, mk = STYLE[key]
        ax.errorbar(np.array(alphas), m.values, yerr=[m.values - lo.values, hi.values - m.values],
                    c=c, marker=mk, ms=3.5, lw=1.1, capsize=1.5, elinewidth=0.7, label=lab)
    mm = S["map"][0]
    ax.axhline(mm, ls="--", c=STYLE["map"][1], lw=0.9, label=STYLE["map"][0])
    ax.set_title(title); ax.set_xlabel("nominal α"); ax.set_xticks(alphas)
    ax.set_ylim(0, 1.0); ax.set_xlim(0.0, 0.22); ax.grid(alpha=0.25, lw=0.5)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(6.875, 2.15), sharey=True)
    # (a) Endoscapes CVS, DINOv2 seeds
    r = pd.read_csv(ROOT / "experiments" / "resplit" / "resplit.csv")
    r = r[r.run.str.startswith("dinov2l")]
    S = series(r, "video_false_confirm", {"surgery": "surgery-level (video max)",
                                          "pooled": "CRC-Mondrian (frames pooled)",
                                          "ltt": "LTT (frames pooled)", "map": "MAP (point world)"})
    panel(axes[0], S, "(a) Endoscapes: CVS confirmation", [0.05, 0.10, 0.20])
    axes[0].set_ylabel("operations with ≥ 1 false certificate")
    # (b) Endoscapes identification
    a = pd.read_csv(ROOT / "experiments" / "surgery_level" / "anatomy_surgery_level.csv")
    a = a[a.task == "identify"]
    S = series(a, "surgeries_with_error", {"surgery": "Surgery-level",
                                           "pooled": "Frame-level ours (frames pooled)",
                                           "ltt": "LTT (frames pooled)", "map": "MAP (point world)"})
    panel(axes[1], S, "(b) Endoscapes: duct / artery labels", [0.05, 0.10, 0.20])
    # (c) Cholec80 after local recalibration (20 videos), dataset's CVS definition
    c = pd.read_csv(ROOT / "experiments" / "surgery_level" / "cholec80_surgery_level.csv")
    cm = c[(c.ref == "cvs_authors") & (c.scenario == "B: recalibrate on 20 Cholec80 videos")]
    S = series(cm, "surgeries_with_false_confirm", {"surgery": "Surgery-level",
                                                    "pooled": "CRC-Mondrian (frames pooled)",
                                                    "ltt": "LTT (frames pooled)", "map": "MAP (point world)"})
    ma = c[(c.ref == "cvs_authors") & (c.method == "MAP (point world)")]["surgeries_with_false_confirm"]
    S["map"] = (ma.mean(), ma.min(), ma.max())
    panel(axes[2], S, "(c) Cholec80 (recalibrated): CVS", [0.05, 0.10, 0.20])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_operation_risk.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", OUT / "fig_operation_risk.pdf")


if __name__ == "__main__":
    main()
