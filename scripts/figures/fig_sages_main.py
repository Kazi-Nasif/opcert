"""Main-paper SAGES figure: the three policies side by side on the declared 700 -> 300 split (10 s
smoothing): (a) cases with >= 1 false CVS confirmation at an annotated timestamp of the pre-clipping
window, exact 95% intervals, nominal level; (b) sensitivity on CVS-positive rated frames, case-bootstrap
intervals.  (c) Assumption stress test: the non-random Endoscapes splits of supplementary Table S10
(experiments/review3/fixed_splits.csv) as a paired-dot plot, alpha = 0.10, 10 s smoothing: each
partition in both directions, per-seed values behind the three-seed mean.  The locked 100 -> 100
analysis and the leave-one-country-out panel stay in the supplement (fig_sages.py, fig_sages_country.py).

    python scripts/figures/fig_sages_main.py [--suffix _v2]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ST  # noqa: E402

ST.setup()
import matplotlib.pyplot as plt  # noqa: E402

# seaborn "colorblind" palette (the usual NeurIPS look) for this figure and its supplementary companion
C_OURS, C_CRC, C_MUTED = "#0173B2", "#DE8F05", "#949494"

POL = {"A": (C_MUTED, "D", "A  Endoscapes threshold, unchanged"),
       "B": (C_CRC, "s", "B  frame-level CRC, recalibrated on SAGES"),
       "C": (C_OURS, "o", "C  per operation, recalibrated on SAGES (ours)")}
ALPHAS = (0.05, 0.10)


def panel_splits(ax):
    """Non-random Endoscapes splits, ours at alpha = 0.10 with 10 s smoothing (Table S10)."""
    d = pd.read_csv(ROOT / "experiments" / "review3" / "fixed_splits.csv")
    d = d[(d.w == 10) & np.isclose(d.alpha, 0.10) & (d.method == "ours")]
    pairs = [("official split\n31 ↔ 40", "official (val->test)", "official reversed"),
             ("video-ID halves\n35 ↔ 36", "video ID: lower->upper", "video ID: upper->lower")]
    crc = pd.read_csv(ROOT / "experiments" / "review3" / "fixed_splits.csv")
    crc = crc[(crc.w == 10) & np.isclose(crc.alpha, 0.10) & (crc.method == "CRC frame-level")].groupby("split").ops_err.mean()
    ax.axvline(10, c=ST.INK, ls=(0, (1.2, 1.4)), lw=0.8, zorder=1)
    ax.text(10.6, 0.5, "α = 0.10", fontsize=6, color=ST.INK, va="center")
    for k, (lab, fwd, rev) in enumerate(pairs):
        y = 1 - k
        mf, mr = 100 * d[d.split == fwd].ops_err.mean(), 100 * d[d.split == rev].ops_err.mean()
        ax.plot([mf, mr], [y, y], c=C_OURS, lw=1.0, zorder=2)
        for sp, m, fc in ((fwd, mf, "white"), (rev, mr, C_OURS)):
            seeds = 100 * d[d.split == sp].ops_err.values
            ax.scatter(seeds, [y - 0.22] * len(seeds), s=9, facecolors=fc, edgecolors=C_OURS, lw=0.6, alpha=0.75, zorder=3)
            ax.scatter([m], [y], s=30, facecolors=fc, edgecolors=C_OURS, lw=1.0, zorder=4)
            ax.annotate(f"{m:.1f}", (m, y + 0.14), ha="center", fontsize=6.2, color=C_OURS)
        print(f"{lab.splitlines()[0]}: forward {mf:.1f}  reversed {mr:.1f}  (CRC {100 * crc[fwd]:.1f} / {100 * crc[rev]:.1f})")
    ax.set_yticks([1, 0]); ax.set_yticklabels([p[0] for p in pairs], fontsize=6.3)
    ax.set_ylim(-0.7, 2.0); ax.set_xlim(0, 50); ax.grid(axis="y", visible=False)
    ax.set_xlabel("operations with ≥1 false confirmation (%)")
    # local legend from detached handles, so the figure-level legend does not pick them up
    mk = dict(marker="o", ls="none", mec=C_OURS, mew=1.0)
    h = [Line2D([], [], mfc="white", ms=5.5, label="calibrate earlier → test later", **mk),
         Line2D([], [], mfc=C_OURS, ms=5.5, label="reversed direction", **mk),
         Line2D([], [], marker="o", ls="none", mfc="white", mec=C_OURS, mew=0.6, ms=3, alpha=0.75, label="single seed")]
    ax.legend(handles=h, loc="upper right", fontsize=6, frameon=False, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.3)
    ax.text(0.02, 0.02, f"frame-level CRC: {100 * crc.min():.0f}–{100 * crc.max():.0f}% in all four directions",
            transform=ax.transAxes, fontsize=6, color=ST.MUTED, va="bottom")


def main(suffix=""):
    d = pd.read_csv(ROOT / "experiments" / "external_sages" / "results_full.csv")
    d = d[(d.w == 10) & d.analysis.str.startswith("F1")]
    fig, ax = plt.subplots(1, 3, figsize=(ST.DOUBLE, 2.3), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.15]))
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
    ax[0].set_title("a  SAGES case error\n", x=-0.16, pad=2)
    ax[1].set_title("b  SAGES sensitivity\n", x=-0.16, pad=2)
    ax[0].text(0.0, 1.02, "exact 95% CI · 700 cal. / 300 test cases", transform=ax[0].transAxes, fontsize=6,
               color=ST.MUTED, va="bottom")
    ax[1].text(0.0, 1.02, "case-bootstrap 95% CI", transform=ax[1].transAxes, fontsize=6, color=ST.MUTED, va="bottom")
    panel_splits(ax[2])
    ax[2].set_title("c  Endoscapes, non-random splits\n", x=-0.14, pad=2)
    ax[2].text(0.0, 1.02, "10 s smoothing · 3 seeds · fixed splits, no interval", transform=ax[2].transAxes,
               fontsize=6, color=ST.MUTED, va="bottom")
    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(wspace=0.42)
    hh, ll = ax[0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.09), columnspacing=1.2, handletextpad=0.4)
    ST.save(fig, ROOT / "paper" / "figures" / f"fig_sages_main{suffix}")
    print("wrote", f"fig_sages_main{suffix}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--suffix", default="")
    main(ap.parse_args().suffix)
