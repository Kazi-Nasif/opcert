"""Main figure: fraction of operations with >= 1 false certificate vs nominal alpha.

Panels: (a) Endoscapes CVS confirmation, (b) Endoscapes duct/artery identification,
(c) Cholec80 CVS confirmation after recalibration on 20 Cholec80 videos.
Points: mean over random video-level splits (and seeds), placed at their actual alpha; whiskers:
2.5-97.5 % over splits in a lighter tint of the method colour.  White background, dotted target
line labelled "target", our method drawn heaviest.  Full CVPR text width (6.875 in).

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

import style as ST  # noqa: E402

ST.setup()
# key -> (label, colour, marker, linestyle, linewidth)
STYLE = {"surgery": ("Per operation (ours)", ST.OURS, "o", "-", 2.2),
         "pooled": ("Frame-level CRC", ST.CRC, "s", "-", 1.5),
         "ltt": ("Frame-level LTT", ST.LTT, "^", (0, (4, 2)), 1.5)}
ALPHAS = [0.05, 0.10, 0.20]


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


def panel(ax, S, title, subtitle):
    xmax = 0.225
    ax.plot([0, xmax], [0, xmax], ls=(0, (1.2, 1.4)), c=ST.INK, lw=0.8, zorder=1)
    ax.text(0.194, 0.228, "target", fontsize=6.3, color=ST.INK, ha="right", va="bottom", zorder=5)
    ax.axhline(S["map"][0], ls=(0, (5, 2.5)), c=ST.POINT, lw=0.9, label="Point world", zorder=2)
    for key in ("pooled", "ltt", "surgery"):
        m, lo, hi = S[key]
        m, lo, hi = m.loc[ALPHAS].values, lo.loc[ALPHAS].values, hi.loc[ALPHAS].values
        lab, c, mk, ls, lw = STYLE[key]
        x = np.array(ALPHAS)
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor=ST.tint(c, 0.45), elinewidth=0.8, capsize=1.8,
                    capthick=0.8, zorder=2)
        ax.plot(x, m, c=c, ls=ls, lw=lw, marker=mk, ms=4 if key == "surgery" else 3.6, mec="white", mew=0.6,
                label=lab, zorder=4 if key == "surgery" else 3)
    ax.set_title(f"{title}\n", pad=2)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=7, color=ST.MUTED, ha="left", va="bottom")
    ax.set_xlabel("nominal error level α"); ax.set_xticks(ALPHAS); ax.set_xticklabels(["5%", "10%", "20%"])
    ax.set_ylim(0, 1.02); ax.set_xlim(0.0, xmax); ax.grid(axis="x", visible=False)
    ax.set_yticks([0, .2, .4, .6, .8, 1]); ax.set_yticklabels(["0", "20", "40", "60", "80", "100"])


def main():
    fig, axes = plt.subplots(1, 3, figsize=(ST.DOUBLE, 2.15), sharey=True)
    # (a) Endoscapes CVS, DINOv2 seeds
    r = pd.read_csv(ROOT / "experiments" / "resplit" / "resplit.csv")
    r = r[r.run.str.startswith("dinov2l")]
    S = series(r, "video_false_confirm", {"surgery": "surgery-level (video max)",
                                          "pooled": "CRC-Mondrian (frames pooled)",
                                          "ltt": "LTT (frames pooled)", "map": "MAP (point world)"})
    panel(axes[0], S, "a  CVS confirmation", "Endoscapes")
    axes[0].set_ylabel("operations with ≥1 false certificate (%)")
    # (b) Endoscapes identification
    a = pd.read_csv(ROOT / "experiments" / "surgery_level" / "anatomy_surgery_level.csv")
    a = a[a.task == "identify"]
    S = series(a, "surgeries_with_error", {"surgery": "Surgery-level",
                                           "pooled": "Frame-level ours (frames pooled)",
                                           "ltt": "LTT (frames pooled)", "map": "MAP (point world)"})
    panel(axes[1], S, "b  Duct / artery identification", "Endoscapes")
    # (c) Cholec80 after local recalibration (20 videos), dataset's CVS definition
    c = pd.read_csv(ROOT / "experiments" / "surgery_level" / "cholec80_surgery_level.csv")
    cm = c[(c.ref == "cvs_authors") & (c.scenario == "B: recalibrate on 20 Cholec80 videos")]
    S = series(cm, "surgeries_with_false_confirm", {"surgery": "Surgery-level",
                                                    "pooled": "CRC-Mondrian (frames pooled)",
                                                    "ltt": "LTT (frames pooled)", "map": "MAP (point world)"})
    ma = c[(c.ref == "cvs_authors") & (c.method == "MAP (point world)")]["surgeries_with_false_confirm"]
    S["map"] = (ma.mean(), ma.min(), ma.max())
    panel(axes[2], S, "c  CVS confirmation", "Cholec80 · recalibrated")
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(x) for x in (STYLE["surgery"][0], STYLE["pooled"][0], STYLE["ltt"][0], "Point world")]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.09),
               columnspacing=1.6, handletextpad=0.5)
    fig.tight_layout(w_pad=0.8)
    ST.save(fig, OUT / "fig_operation_risk")
    print("wrote", OUT / "fig_operation_risk.pdf")


if __name__ == "__main__":
    main()
