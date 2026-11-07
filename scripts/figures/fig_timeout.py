"""Time-out outcomes: at the first clip, what does each system say, split by whether the surgeons'
reference says CVS was achieved?  Stacked bars of confirm / STOP / defer for the point world and
our time-out certificate with CONFIRM at a = 0.10 (class-conditional over operations without CVS)
and a conditional STOP at a = 0.20 (class-conditional over operations with CVS; defers when too
few such operations exist), 5 s smoothing, Endoscapes and Cholec80 (recalibrated on 37 operations).
Means over random operation splits x 3 seeds (experiments/review/timeout_alarm.csv).

    python scripts/review2/timeout_alarm.py && python scripts/figures/fig_timeout.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ST  # noqa: E402

ST.setup()
COL = {"confirm": "#138a60", "STOP": "#d03b3b", "defer": "#d9d8d3"}   # status colours, labelled in the legend
OURS = "asymmetric (confirm .10, STOP .20)"


def main():
    d = pd.read_csv(ROOT / "experiments" / "review" / "timeout_alarm.csv")
    d = d[d.smooth_s == 5]
    groups = []
    for ds in ("Endoscapes", "Cholec80 (recal.)"):
        for rule, label in (("point world", "Point world"), (OURS, "Ours")):
            x = d[(d.dataset == ds) & (d.rule == rule)]
            wc = {"confirm": x.confirm_given_CVS.mean(), "STOP": x.alarm_given_CVS.mean()}
            wc["defer"] = max(0.0, 1 - wc["confirm"] - wc["STOP"])
            nc = {"confirm": x.confirm_given_noCVS.mean(), "STOP": x.alarm_given_noCVS.mean()}
            nc["defer"] = max(0.0, 1 - nc["confirm"] - nc["STOP"])
            groups.append((ds.replace(" (recal.)", ""), label, wc, nc))
    fig, axes = plt.subplots(1, 2, figsize=(ST.SINGLE, 1.5), sharey=True)
    names = [f"{g[1]}, {g[0]}" for g in groups]
    ys = np.array([3.3, 2.3, 1.0, 0.0])
    for ax, key, title in ((axes[0], 2, "CVS achieved at the clip"), (axes[1], 3, "CVS NOT achieved")):
        left = np.zeros(len(groups))
        for state in ("confirm", "STOP", "defer"):
            vals = np.array([g[key][state] for g in groups])
            ax.barh(ys, vals, left=left, color=COL[state], height=0.72, label=state, edgecolor="white", lw=1.0)
            for i, v in enumerate(vals):
                if v > 0.12:
                    ax.text(left[i] + v / 2, ys[i], f"{100 * v:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if state != "defer" else "k")
            left += vals
        ax.set_title(title, fontsize=7.5); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.grid(False)
        ax.set_xticklabels(["0", "50", "100%"], fontsize=6.5)
    axes[0].set_yticks(ys); axes[0].set_yticklabels(names, fontsize=6.5)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.6, -0.25), ncol=3)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    ST.save(fig, OUT / "fig_timeout")
    print("wrote fig_timeout")


if __name__ == "__main__":
    main()
