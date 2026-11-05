"""Risk-control baselines over a 1-D family of decision rules indexed by a threshold.

Both pick a threshold lam on a confidence score from a grid, using a calibration loss
matrix L[i, g] in {0, 1} that is non-increasing in the threshold (a higher threshold
acts on fewer frames, so it cannot err more).

  Conformal risk control (Angelopoulos, Bates, Fisch, Lei, Schuster; ICLR 2024):
      lam_hat = smallest lam with (n * Rhat(lam) + B) / (n + 1) <= alpha,  B = 1
      guarantee  E[L_test(lam_hat)] <= alpha
  Learn then Test (Angelopoulos, Bates, Candes, Jordan, Lei; 2021), fixed-sequence
  testing from the most conservative threshold, Hoeffding-Bentkus p-values:
      guarantee  P(R(lam_hat) <= alpha) >= 1 - delta
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom


def crc_index(L: np.ndarray, alpha: float, B: float = 1.0) -> int | None:
    """L: (n, G), grid ascending in threshold (risk non-increasing in g). Returns the
    smallest valid grid index, or None if even the most conservative one fails."""
    n = L.shape[0]
    ok = (n * L.mean(0) + B) / (n + 1) <= alpha
    return int(np.flatnonzero(ok)[0]) if ok.any() else None


def hb_pvalue(rhat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: R > alpha."""
    def h(a, b):
        a = min(max(a, 1e-12), 1 - 1e-12)
        return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))
    p_h = math.exp(-n * h(min(rhat, alpha), alpha)) if rhat < alpha else 1.0
    p_b = math.e * binom.cdf(math.ceil(n * rhat), n, alpha)
    return min(p_h, p_b, 1.0)


def ltt_index(L: np.ndarray, alpha: float, delta: float = 0.1) -> int | None:
    """Fixed-sequence test from the largest threshold downwards; returns the smallest
    index whose null (risk > alpha) was rejected, or None."""
    n, G = L.shape
    rhat = L.mean(0)
    best = None
    for g in range(G - 1, -1, -1):
        if hb_pvalue(float(rhat[g]), n, alpha) <= delta:
            best = g
        else:
            break
    return best


def threshold_grid(scores: np.ndarray, size: int = 400) -> np.ndarray:
    """Ascending candidate thresholds spanning the observed scores (plus 'act on nothing')."""
    qs = np.unique(np.quantile(scores, np.linspace(0, 1, size)))
    return np.concatenate([qs, [np.inf]])
