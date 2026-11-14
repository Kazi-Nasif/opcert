"""Supplementary SAGES figure: leave-one-country-out case error at alpha = 0.10 (10 s smoothing), one
row per held-out country (ten countries with >= 30 cases), ours with exact 95% intervals and frame-level
CRC recalibrated on the other countries.  Formerly panel (c) of the main SAGES figure.

    python scripts/figures/fig_sages_country.py
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

# seaborn "colorblind" palette (the usual NeurIPS look) for this figure and its supplementary companion
C_OURS, C_CRC, C_MUTED = "#0173B2", "#DE8F05", "#949494"


def main():
    f2 = pd.read_csv(ROOT / "experiments" / "external_sages" / "results_full.csv")
    f2 = f2[(f2.w == 10) & f2.analysis.str.startswith("F2") & np.isclose(f2.alpha, 0.10)]
    f2["country"] = f2.analysis.str.split().str[-1].astype(int)
    cs = f2[f2.policy == "C"].sort_values("n_test", ascending=True)
    fig, ax = plt.subplots(figsize=(ST.SINGLE, 2.75))
    ax.axvline(10, c=ST.INK, ls=(0, (1.2, 1.4)), lw=0.8, label="target (nominal level)")
    for k, (_, r) in enumerate(cs.iterrows()):
        b = f2[(f2.policy == "B") & (f2.country == r.country)].iloc[0]
        ax.plot([100 * r.case_error_lo, 100 * r.case_error_hi], [k, k], c=C_OURS, lw=0.9, zorder=2)
        ax.scatter(100 * r.case_error, k, s=16, c=C_OURS, marker="o", ec="white", lw=0.4, zorder=3,
                   label="per operation, recalibrated on the other countries (ours)" if k == 0 else None)
        ax.scatter(100 * b.case_error, k, s=14, c=C_CRC, marker="s", ec="white", lw=0.4, zorder=3,
                   label="frame-level CRC, recalibrated on the other countries" if k == 0 else None)
    ax.set_yticks(range(len(cs))); ax.set_yticklabels([f"{int(r.country)} (n={int(r.n_test)})" for _, r in cs.iterrows()], fontsize=6.5)
    ax.set_ylabel("held-out country (test cases)")
    ax.set_xlim(0, 70); ax.set_ylim(-0.7, len(cs) - 0.3); ax.grid(axis="y", visible=False)
    ax.set_xlabel("cases with ≥1 false confirmation (%)")
    ax.set_title("Leave one country out, α = 0.10", x=-0.2)
    fig.legend(loc="lower center", ncol=1, fontsize=6, frameon=False, handletextpad=0.3, bbox_to_anchor=(0.55, -0.01))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    ST.save(fig, ROOT / "paper" / "figures" / "fig_sages_country")
    print("wrote fig_sages_country; ours range %.1f–%.1f %%" % (100 * cs.case_error.min(), 100 * cs.case_error.max()))


if __name__ == "__main__":
    main()
