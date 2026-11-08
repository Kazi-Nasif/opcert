"""Main figure: operation-level error against the nominal level, and what the guarantee costs.

Panels: (a) Endoscapes CVS confirmation and (b) Cholec80 CVS confirmation after recalibration on
20 Cholec80 videos: fraction of operations with >= 1 false certificate vs nominal alpha, means at
their actual alpha over random video-level splits x seeds (a: 200 x 3, b: 100 x 3), whiskers
2.5-97.5 % across split-seed evaluations.  (c) Endoscapes: sensitivity (share of CVS keyframes
confirmed) against the *observed* per-operation error for calibration over operations and for
frame-level CRC, each on raw (solid) and 10 s-smoothed (dashed) scores; filled markers are the
calibrated operating points (alpha = .05/.10/.20), hollow markers a descriptive threshold sweep
(alpha = .002-.02).  Full CVPR text width.

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
CAL = (0.05, 0.10, 0.20)        # calibrated operating points in panel (c); other alphas are a descriptive sweep
SMOOTH_LS = (0, (4, 2))         # dashed = 10 s smoothing in panel (c)


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


def panel_frontier(ax, arch="dinov2l_b8"):
    """Sensitivity vs observed per-operation error (Endoscapes); the essential curves of the supplementary frontier."""
    d = pd.read_csv(ROOT / "experiments" / "compare_all" / "compare_all.csv")
    d = d[d.run.str.startswith(arch)]
    # method -> (legend label, colour, marker, linestyle, linewidth, z)
    curves = {
        "Surgery-level (ours)": ("Per operation (ours)", ST.OURS, "o", "-", 2.2, 6),
        "Surgery-level + smoothing (ours)": ("Per operation + 10 s smoothing", ST.OURS, "o", SMOOTH_LS, 1.6, 5),
        "CRC (= frame-level)": ("Frame-level CRC", ST.CRC, "s", "-", 1.5, 4),
        "CRC + smoothing (10 s)": ("Frame-level CRC + 10 s smoothing", ST.CRC, "s", SMOOTH_LS, 1.2, 4),
    }
    curve_pts, handles = {}, []
    for m, (lab, c, mk, ls, lw, z) in curves.items():
        g = d[d.method == m].groupby("alpha")[["ops_false_ge1", "sens"]].mean().sort_index()
        curve_pts[m] = g
        handles += ax.plot(g.ops_false_ge1, g.sens, c=c, ls=ls, lw=lw, zorder=z, label=lab)
        cal = g[g.index.isin(CAL)]; swp = g[~g.index.isin(CAL)]
        ax.scatter(cal.ops_false_ge1, cal.sens, marker=mk, s=20, c=c, ec="white", lw=0.5, zorder=z + 1)
        ax.scatter(swp.ops_false_ge1, swp.sens, marker=mk, s=14, facecolors="white", edgecolors=c, lw=0.8, zorder=z + 1)
    mp = d[d.method == "Point world (MAP)"][["ops_false_ge1", "sens"]].mean()
    handles.append(ax.scatter([mp.ops_false_ge1], [mp.sens], marker="*", s=55, c=ST.POINT, zorder=7, label="Point world"))
    leg1 = ax.legend(handles=handles, loc="upper left", fontsize=6, frameon=False, handlelength=2.2, handletextpad=0.5,
                     borderaxespad=0.3, labelspacing=0.35)
    ax.add_artist(leg1)
    # marker semantics
    h1 = ax.scatter([], [], marker="o", s=20, c=ST.MUTED, ec="white", lw=0.5, label="calibrated, α = .05 / .10 / .20")
    h2 = ax.scatter([], [], marker="o", s=14, facecolors="white", edgecolors=ST.MUTED, lw=0.8, label="threshold sweep, α = .002–.02")
    ax.legend(handles=[h1, h2], loc="lower right", fontsize=6, frameon=False, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("c  Sensitivity against observed error\n", pad=2)
    ax.text(0.0, 1.02, "Endoscapes · observed, not nominal, error", transform=ax.transAxes, fontsize=7, color=ST.MUTED,
            ha="left", va="bottom")
    ax.set_xlabel("operations with ≥1 false certificate (%)")
    ax.set_ylabel("sensitivity (% of CVS frames confirmed)")
    ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.92)
    ax.set_xticks([0, .2, .4, .6, .8]); ax.set_xticklabels(["0", "20", "40", "60", "80"])
    ax.set_yticks([0, .2, .4, .6, .8]); ax.set_yticklabels(["0", "20", "40", "60", "80"])
    # caption support: where does frame-level CRC at its smallest sweep levels land relative to our curve?
    crc = curve_pts["CRC (= frame-level)"]; ours = curve_pts["Surgery-level (ours)"]
    for a_ in (0.002, 0.005, 0.01):
        print(f"CRC alpha={a_}: ops_false={crc.loc[a_].ops_false_ge1:.3f} sens={crc.loc[a_].sens:.3f}")
    for a_ in CAL:
        print(f"ours alpha={a_}: ops_false={ours.loc[a_].ops_false_ge1:.3f} sens={ours.loc[a_].sens:.3f}")
    print(f"point world: ops_false={mp.ops_false_ge1:.3f} sens={mp.sens:.3f}")


def main():
    fig = plt.figure(figsize=(ST.DOUBLE, 2.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
    ax_a = fig.add_subplot(gs[0]); ax_b = fig.add_subplot(gs[1], sharey=ax_a); ax_c = fig.add_subplot(gs[2])
    # (a) Endoscapes CVS, DINOv2 seeds (200 splits x 3 seeds)
    r = pd.read_csv(ROOT / "experiments" / "resplit" / "resplit.csv")
    r = r[r.run.str.startswith("dinov2l")]
    S = series(r, "video_false_confirm", {"surgery": "surgery-level (video max)",
                                          "pooled": "CRC-Mondrian (frames pooled)",
                                          "ltt": "LTT (frames pooled)", "map": "MAP (point world)"})
    panel(ax_a, S, "a  CVS confirmation", "Endoscapes")
    ax_a.set_ylabel("operations with ≥1 false certificate (%)")
    # (b) Cholec80 after local recalibration (20 videos; 100 splits x 3 seeds), dataset's CVS definition
    c = pd.read_csv(ROOT / "experiments" / "surgery_level" / "cholec80_surgery_level.csv")
    cm = c[(c.ref == "cvs_authors") & (c.scenario == "B: recalibrate on 20 Cholec80 videos")]
    S = series(cm, "surgeries_with_false_confirm", {"surgery": "Surgery-level",
                                                    "pooled": "CRC-Mondrian (frames pooled)",
                                                    "ltt": "LTT (frames pooled)", "map": "MAP (point world)"})
    ma = c[(c.ref == "cvs_authors") & (c.method == "MAP (point world)")]["surgeries_with_false_confirm"]
    S["map"] = (ma.mean(), ma.min(), ma.max())
    panel(ax_b, S, "b  CVS confirmation", "Cholec80 · recalibrated")
    ax_b.tick_params(labelleft=False)
    # (c) sensitivity vs observed per-operation error, Endoscapes
    panel_frontier(ax_c)
    h, l = ax_a.get_legend_handles_labels()
    order = [l.index(x) for x in (STYLE["surgery"][0], STYLE["pooled"][0], STYLE["ltt"][0], "Point world")]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.09),
               columnspacing=1.6, handletextpad=0.5)
    fig.tight_layout(w_pad=0.8)
    ST.save(fig, OUT / "fig_operation_risk")
    print("wrote", OUT / "fig_operation_risk.pdf")


if __name__ == "__main__":
    main()
