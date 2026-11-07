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
    fig, ax = plt.subplots(1, 2, figsize=(ST.DOUBLE, 2.15))
    for j, al in enumerate(ALPHAS):
        ax[0].fill_between([j - 0.45, j + 0.45], [100 * al] * 2, [60] * 2, color=ST.VIOLATE, lw=0, zorder=0)
        ax[0].plot([j - 0.45, j + 0.45], [100 * al] * 2, c=ST.INK, ls=(0, (2, 1.5)), lw=0.8, label="nominal level" if j == 0 else None)
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
        ax[k].set_xticks(range(len(ALPHAS))); ax[k].set_xticklabels([f"nominal α = {al:.2f}" for al in ALPHAS])
        ax[k].grid(axis="x", visible=False); ax[k].set_xlim(-0.6, len(ALPHAS) - 0.4)
    ax[0].set_ylim(0, 50); ax[0].set_ylabel("cases with ≥1 false confirmation (%)")
    ax[1].set_ylim(0, 100); ax[1].set_ylabel("sensitivity on CVS frames (%)")
    ax[0].set_title("a  case-level error, exact 95% CI (300 test cases)")
    ax[1].set_title("b  sensitivity, case-bootstrap 95% CI")
    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.1), columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout()
    ST.save(fig, ROOT / "paper" / "figures" / "fig_sages_main")
    print("wrote fig_sages_main")


if __name__ == "__main__":
    main()
