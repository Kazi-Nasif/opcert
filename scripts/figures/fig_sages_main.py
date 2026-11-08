"""Main-paper SAGES figure: the three policies side by side on the declared 700 -> 300 split (10 s
smoothing): (a) cases with >= 1 false CVS confirmation at an annotated timestamp of the pre-clipping
window, exact 95% intervals, nominal level; (b) sensitivity on CVS-positive rated frames, case-bootstrap
intervals.  The locked 100 -> 100 analysis stays in the supplementary figure.

    python scripts/figures/fig_sages_main.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ST  # noqa: E402

ST.setup()
import matplotlib.pyplot as plt  # noqa: E402

POL = {"A": (ST.POINT, "D", "A  Endoscapes threshold, unchanged"),
       "B": (ST.CRC, "s", "B  frame-level CRC, recalibrated on SAGES"),
       "C": (ST.OURS, "o", "C  per operation, recalibrated on SAGES (ours)")}
ALPHAS = (0.05, 0.10)


def main():
    d = pd.read_csv(ROOT / "experiments" / "external_sages" / "results_full.csv")
    d = d[(d.w == 10) & d.analysis.str.startswith("F1")]
    fig, ax = plt.subplots(1, 3, figsize=(ST.DOUBLE, 2.3), gridspec_kw=dict(width_ratios=[0.8, 0.8, 1.4]))
    for j, al in enumerate(ALPHAS):
        ax[0].plot([j - 0.45, j + 0.45], [100 * al] * 2, c=ST.INK, ls=(0, (1.2, 1.4)), lw=0.8, label="target (nominal level)" if j == 0 else None)
        for i, p in enumerate("ABC"):
            r = d[(d.policy == p) & np.isclose(d.alpha, al)].iloc[0]
            c, mk, lab = POL[p]; x = j + (i - 1) * 0.26
            ax[0].errorbar(x, 100 * r.case_error, yerr=[[100 * (r.case_error - r.case_error_lo)], [100 * (r.case_error_hi - r.case_error)]],
                           fmt=mk, c=c, mec="white", mew=0.5, label=lab if j == 0 else None, zorder=3)
            ax[0].annotate(f"{100 * r.case_error:.1f}", (x, 100 * r.case_error_hi + 1.2), ha="center", fontsize=6, color=c)
            ax[1].errorbar(x, 100 * r.sens, yerr=[[100 * (r.sens - r.sens_lo)], [100 * (r.sens_hi - r.sens)]],
                           fmt=mk, c=c, mec="white", mew=0.5, zorder=3)
            ax[1].annotate(f"{100 * r.sens:.1f}", (x, 100 * r.sens_hi + 2), ha="center", fontsize=6, color=c)
    for k in range(2):
        ax[k].set_xticks(range(len(ALPHAS))); ax[k].set_xticklabels([f"α = {al:.2f}".replace("0.", ".") for al in ALPHAS])
        ax[k].grid(axis="x", visible=False); ax[k].set_xlim(-0.6, len(ALPHAS) - 0.4)
    ax[0].set_ylim(0, 50); ax[0].set_ylabel("cases with ≥1 false confirmation (%)")
    ax[1].set_ylim(0, 100); ax[1].set_ylabel("sensitivity on CVS frames (%)")
    ax[0].set_title("a  case error, exact 95% CI")
    ax[1].set_title("b  sensitivity, bootstrap 95% CI")
    ax[0].text(0.02, 0.97, "300 test cases", transform=ax[0].transAxes, fontsize=6, color=ST.MUTED, va="top")
    # (c) leave-one-country-out at alpha = 0.10: one dot per held-out country, size = cases
    f2 = pd.read_csv(ROOT / "experiments" / "external_sages" / "results_full.csv")
    f2 = f2[(f2.w == 10) & f2.analysis.str.startswith("F2") & np.isclose(f2.alpha, 0.10)]
    f2["country"] = f2.analysis.str.split().str[-1].astype(int)
    cs = f2[f2.policy == "C"].sort_values("n_test", ascending=True)
    yy = np.arange(len(cs))
    ax[2].axvline(10, c=ST.INK, ls=(0, (1.2, 1.4)), lw=0.8)
    for k, (_, r) in enumerate(cs.iterrows()):
        b = f2[(f2.policy == "B") & (f2.country == r.country)].iloc[0]
        ax[2].plot([100 * r.case_error_lo, 100 * r.case_error_hi], [k, k], c=ST.OURS, lw=0.9, zorder=2)
        ax[2].scatter(100 * r.case_error, k, s=14, c=ST.OURS, marker="o", ec="white", lw=0.4, zorder=3)
        ax[2].scatter(100 * b.case_error, k, s=12, c=ST.CRC, marker="s", ec="white", lw=0.4, zorder=3)
    ax[2].set_yticks(yy); ax[2].set_yticklabels([f"{int(r.country)} (n={int(r.n_test)})" for _, r in cs.iterrows()], fontsize=6.5)
    ax[2].set_ylabel("held-out country (test cases)")
    ax[2].set_xlim(0, 70); ax[2].set_ylim(-0.7, len(cs) - 0.3); ax[2].grid(axis="y", visible=False)
    ax[2].set_xlabel("cases with ≥1 false confirmation (%)")
    ax[2].set_title("c  held-out countries, α = 0.10")
    # lay out first (so the long country tick labels of (c) get their space), then add the legend above the panels
    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(wspace=0.7)    # a visible gap between (b) and the labels of (c)
    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.09), columnspacing=1.2, handletextpad=0.4)
    ST.save(fig, ROOT / "paper" / "figures" / "fig_sages_main")
    print("wrote fig_sages_main")


if __name__ == "__main__":
    main()
