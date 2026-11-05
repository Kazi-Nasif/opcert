"""Marginal vs calibration-conditional validity of the surgery-level certificate (R1 W4, R2.3).

For V calibration operations and j = ceil((V+1)(1-a)), the per-operation risk of the deployed
threshold, conditional on the calibration set, is R ~ Beta(V+1-j, j) (continuous scores), with
mean (V+1-j)/(V+1) <= a (the marginal guarantee).  On an evaluation half of n operations the
observed error is ~ Binomial(n, R)/n (beta-binomial overall).  We compare the observed per-split
errors of Table 3 (200 halvings x 3 seeds, DINOv2-L, V = 35, n = 36) with this prediction, report
P(R > a), and the number of calibration operations a PAC certificate needs,
P(R > a) <= delta = 0.1 (training-conditional conformal; Vovk 2012).
Note: the 200 halvings of one pool overlap; their spread is a stability measure, not 200
independent replicates.

    python scripts/review2/calib_conditional.py
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, betabinom

ROOT = Path(__file__).resolve().parents[2]
V, N_EVAL, DELTA = 35, 36, 0.1


def pac_V(alpha, delta=DELTA):
    for v in range(1, 2000):
        for j in range(1, v + 1):
            if beta.sf(alpha, v + 1 - j, j) <= delta:
                return v
    return None


def main():
    d = pd.read_csv(ROOT / "experiments" / "compare_all" / "compare_all.csv")
    d = d[d.run.str.startswith("dinov2l_b8")]
    out, rows = {}, []
    fig, axes = plt.subplots(1, 3, figsize=(6.875, 1.9), sharey=False)
    for ax, a in zip(axes, (0.05, 0.10, 0.20)):
        j = math.ceil((V + 1) * (1 - a)); A_, B_ = V + 1 - j, j
        obs = d[(d.method == "Surgery-level (ours)") & (d.alpha == a)].ops_false_ge1.to_numpy()
        k = np.round(obs * N_EVAL).astype(int)
        pred = betabinom(N_EVAL, A_, B_)
        res = {"alpha": a, "j": j, "mean_risk_theory": A_ / (V + 1), "P(R>alpha)": float(beta.sf(a, A_, B_)),
               "obs_mean": float(obs.mean()), "obs_frac_above_alpha": float((obs > a + 1e-12).mean()),
               "pred_frac_above_alpha": float(pred.sf(math.floor(a * N_EVAL + 1e-9))),
               "obs_q95": float(np.quantile(obs, 0.95)), "pred_q95": float(pred.ppf(0.95) / N_EVAL),
               "V_for_PAC_delta0.1": pac_V(a)}
        rows.append(res)
        ks = np.arange(0, 16)
        emp = np.bincount(np.clip(k, 0, 15), minlength=16) / len(k)
        ax.bar(ks / N_EVAL, emp, width=0.8 / N_EVAL, color="#1f4e9c", alpha=0.6, label="observed (600 splits)")
        ax.plot(ks / N_EVAL, pred.pmf(ks), "o-", c="#e07b1a", ms=2.5, lw=1, label="theory (beta-binomial)")
        ax.axvline(a, c="k", ls=":", lw=0.8)
        ax.set_title(f"α = {a:.2f}", fontsize=7.5); ax.set_xlim(-0.01, 0.42); ax.tick_params(labelsize=6.5)
        ax.set_xlabel("operations with ≥1 false confirmation", fontsize=6.5)
    axes[0].set_ylabel("share of splits", fontsize=6.5); axes[0].legend(fontsize=5.8, frameon=False)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "paper" / "figures" / f"fig_conditional.{ext}", dpi=300, bbox_inches="tight")
    out["rows"] = rows
    (ROOT / "experiments" / "review" / "calib_conditional.json").write_text(json.dumps(out, indent=1))
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
