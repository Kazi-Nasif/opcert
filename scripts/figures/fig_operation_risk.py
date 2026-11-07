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

import style as ST  # noqa: E402

ST.setup()
STYLE = {"surgery": ST.METHODS["ours"], "pooled": ST.METHODS["crc"], "ltt": ST.METHODS["ltt"],
         "map": ST.METHODS["point"]}


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
    ST.nominal(ax, 0.23)
    off = {"pooled": -0.004, "ltt": 0.0, "surgery": 0.004}
    for key in ("pooled", "ltt", "surgery"):
        m, lo, hi = S[key]
        m, lo, hi = m.loc[alphas], lo.loc[alphas], hi.loc[alphas]
        lab, c, mk = STYLE[key]
        x = np.array(alphas) + off[key]
        ax.errorbar(x, m.values, yerr=[m.values - lo.values, hi.values - m.values], c=c, marker=mk,
                    mec="white", mew=0.6, elinewidth=0.8, label=lab, zorder=3)
    ax.axhline(S["map"][0], ls=(0, (4, 2)), c=STYLE["map"][1], lw=1.0, label=STYLE["map"][0], zorder=2)
    ax.set_title(title); ax.set_xlabel("nominal level α"); ax.set_xticks(alphas)
    ax.set_xticklabels([f"{a:.2f}".lstrip("0") for a in alphas])
    ax.set_ylim(0, 1.02); ax.set_xlim(0.0, 0.23)
    ax.set_yticks([0, .2, .4, .6, .8, 1]); ax.set_yticklabels(["0", "20", "40", "60", "80", "100"])


def main():
    fig, axes = plt.subplots(1, 3, figsize=(ST.DOUBLE, 2.05), sharey=True)
    # (a) Endoscapes CVS, DINOv2 seeds
    r = pd.read_csv(ROOT / "experiments" / "resplit" / "resplit.csv")
    r = r[r.run.str.startswith("dinov2l")]
    S = series(r, "video_false_confirm", {"surgery": "surgery-level (video max)",
                                          "pooled": "CRC-Mondrian (frames pooled)",
                                          "ltt": "LTT (frames pooled)", "map": "MAP (point world)"})
    panel(axes[0], S, "a  Endoscapes · CVS confirmation", [0.05, 0.10, 0.20])
    axes[0].set_ylabel("operations with ≥1 false statement (%)")
    # (b) Endoscapes identification
    a = pd.read_csv(ROOT / "experiments" / "surgery_level" / "anatomy_surgery_level.csv")
    a = a[a.task == "identify"]
    S = series(a, "surgeries_with_error", {"surgery": "Surgery-level",
                                           "pooled": "Frame-level ours (frames pooled)",
                                           "ltt": "LTT (frames pooled)", "map": "MAP (point world)"})
    panel(axes[1], S, "b  Endoscapes · duct / artery labels", [0.05, 0.10, 0.20])
    # (c) Cholec80 after local recalibration (20 videos), dataset's CVS definition
    c = pd.read_csv(ROOT / "experiments" / "surgery_level" / "cholec80_surgery_level.csv")
    cm = c[(c.ref == "cvs_authors") & (c.scenario == "B: recalibrate on 20 Cholec80 videos")]
    S = series(cm, "surgeries_with_false_confirm", {"surgery": "Surgery-level",
                                                    "pooled": "CRC-Mondrian (frames pooled)",
                                                    "ltt": "LTT (frames pooled)", "map": "MAP (point world)"})
    ma = c[(c.ref == "cvs_authors") & (c.method == "MAP (point world)")]["surgeries_with_false_confirm"]
    S["map"] = (ma.mean(), ma.min(), ma.max())
    panel(axes[2], S, "c  Cholec80 (recalibrated) · CVS", [0.05, 0.10, 0.20])
    axes[0].errorbar([], [], yerr=[], fmt="none", ecolor=ST.MUTED, elinewidth=0.8, capsize=1.8,
                     label="whiskers: 2.5–97.5% across splits")
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(x) for x in (STYLE["surgery"][0], STYLE["pooled"][0], STYLE["ltt"][0], STYLE["map"][0], "nominal level",
                                  "whiskers: 2.5–97.5% across splits")]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.07),
               columnspacing=1.0, handletextpad=0.4)
    axes[0].text(0.012, 0.95, "violates the level", color="#b4442a", fontsize=6.5, va="top")
    fig.tight_layout(w_pad=0.8)
    ST.save(fig, OUT / "fig_operation_risk")
    print("wrote", OUT / "fig_operation_risk.pdf")


if __name__ == "__main__":
    main()
