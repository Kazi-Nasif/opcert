"""Decision layer: may the CDSS tell the surgeon "CVS confirmed — clipping permitted"?

World z in {0,1}^3 (C1, C2, C3); rule: CONFIRM only if C1 and C2 and C3.
Configurations are coded c = 4*C1 + 2*C2 + C3, so CVS achieved <=> c == 7.

All methods output a boolean CONFIRM per frame; not confirming means the CDSS keeps
the surgeon in the dissection phase and reports which criteria are not established.
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-6
CODES = np.array([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)], bool)   # (8, 3)


def joint_from_marginals(P3: np.ndarray) -> np.ndarray:
    """(N, 8) product distribution from (N, 3) marginals (the misspecified ICASSP score)."""
    P3 = np.clip(P3, EPS, 1 - EPS)
    return np.prod(np.where(CODES[None], P3[:, None, :], 1 - P3[:, None, :]), axis=2)


def codes(Z: np.ndarray) -> np.ndarray:
    return (Z[:, 0].astype(int) << 2) | (Z[:, 1].astype(int) << 1) | Z[:, 2].astype(int)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores); k = math.ceil((n + 1) * (1 - alpha))
    return math.inf if k > n else float(np.sort(scores)[k - 1])


# ------------------------------------------------------------------ methods
def map_confirm(P8: np.ndarray) -> np.ndarray:
    """Point world: the most probable configuration is 'all three met'."""
    return P8.argmax(1) == 7


def threshold_confirm(P8: np.ndarray, tau: float) -> np.ndarray:
    return P8[:, 7] >= tau


def world_sets(P8: np.ndarray, q_hat: float) -> np.ndarray:
    """(N, 8) membership of each configuration in Gamma = {c : -log P(c) <= q_hat}."""
    return -np.log(np.clip(P8, EPS, 1)) <= q_hat


def world_confirm(P8: np.ndarray, q_hat: float) -> np.ndarray:
    """CONFIRM iff Gamma is non-empty and every plausible configuration is c == 7."""
    G = world_sets(P8, q_hat)
    return G[:, 7] & ~G[:, :7].any(1)


def conformal_selection(P8_cal: np.ndarray, cvs_cal: np.ndarray, P8_test: np.ndarray,
                        fdr: float) -> np.ndarray:
    """Conformal selection with BH (Jin & Candes, JMLR 2023): select test frames as CVS-
    achieved with FDR <= fdr, using calibration frames whose label is *not* CVS as nulls.
    Score V = P(CVS) (larger = more evidence); conformal p-value against the null scores."""
    v0 = P8_cal[~cvs_cal, 7]
    v = P8_test[:, 7]
    n0 = len(v0)
    p = (1 + (v0[None, :] >= v[:, None]).sum(1)) / (n0 + 1)
    m = len(p); order = np.argsort(p)
    passed = p[order] <= fdr * np.arange(1, m + 1) / m
    k = np.flatnonzero(passed).max() + 1 if passed.any() else 0
    sel = np.zeros(m, bool); sel[order[:k]] = True
    return sel


# ------------------------------------------------------------------ decision-sufficient projection
# The rule reads (C1, C2, C3) only through their conjunction, so the decision-sufficient
# statistic is b = 1[CVS]. Accepted-decision safety needs coverage of b only (any world
# set whose projection covers b_true gives P(confirm and not CVS) <= alpha), and covering
# a binary statistic is far cheaper than covering the full 8-way configuration.

def projected_scores(P8: np.ndarray, cvs: np.ndarray) -> np.ndarray:
    """-log P(b_true) under the model, b = CVS achieved."""
    p = np.clip(P8[:, 7], EPS, 1 - EPS)
    return -np.log(np.where(cvs, p, 1 - p))


def projected_confirm(P8: np.ndarray, q_hat: float) -> np.ndarray:
    """Gamma_b = {b : -log P(b) <= q_hat}; CONFIRM iff Gamma_b == {CVS}."""
    p = np.clip(P8[:, 7], EPS, 1 - EPS)
    return (-np.log(p) <= q_hat) & (-np.log(1 - p) > q_hat)


def projected_alarm(P8: np.ndarray, q_hat: float) -> np.ndarray:
    p = np.clip(P8[:, 7], EPS, 1 - EPS)
    return (-np.log(1 - p) <= q_hat) & (-np.log(p) > q_hat)


def mondrian_confirm(P8_cal: np.ndarray, cvs_cal: np.ndarray, P8_test: np.ndarray, alpha: float) -> np.ndarray:
    """Class-conditional: P(confirm | CVS not achieved) <= alpha. Calibrate the CVS
    probability on non-CVS calibration frames only; confirm when the test frame exceeds
    the ceil((n0+1)(1-alpha))-th smallest of them (infinite threshold -> never confirm)."""
    v0 = np.sort(P8_cal[~cvs_cal, 7])
    n0 = len(v0); k = math.ceil((n0 + 1) * (1 - alpha))
    if n0 == 0 or k > n0:
        return np.zeros(len(P8_test), bool)
    return P8_test[:, 7] > v0[k - 1]


def mondrian_alarm(P8_cal: np.ndarray, cvs_cal: np.ndarray, P8_test: np.ndarray, alpha: float) -> np.ndarray:
    """Class-conditional: P(alarm | CVS achieved) <= alpha."""
    v1 = np.sort(P8_cal[cvs_cal, 7])
    n1 = len(v1); k = math.ceil((n1 + 1) * alpha) - 1          # alarm below the alpha-quantile
    if n1 == 0 or k < 0 or k >= n1:                             # nothing to calibrate on: fail closed
        return np.zeros(len(P8_test), bool)
    return P8_test[:, 7] < v1[k]


# ------------------------------------------------------------------ surgery-level (video) guarantees
# Frames of one operation are strongly dependent, so frame-level exchangeability fails;
# pooling all frames is only asymptotically valid in the number of videos (Dunn,
# Wasserman & Ramdas, JASA 2023) and under-covers with ~30 videos. Videos ARE
# exchangeable. One score per calibration video -- the most confident non-CVS frame --
# gives an exact surgery-level guarantee:
#     P(a new operation receives ANY false CVS confirmation) <= alpha.

def video_max_confirm(p_cal: np.ndarray, cvs_cal: np.ndarray, vid_cal: np.ndarray,
                      p_test: np.ndarray, alpha: float) -> np.ndarray:
    m = [p_cal[(vid_cal == v) & ~cvs_cal].max() for v in np.unique(vid_cal)
         if ((vid_cal == v) & ~cvs_cal).any()]
    m = np.sort(m); k = math.ceil((len(m) + 1) * (1 - alpha))
    if len(m) == 0 or k > len(m):
        return np.zeros(len(p_test), bool)                    # too few videos for this alpha
    return p_test > m[k - 1]


def one_per_video_confirm(p_cal, cvs_cal, vid_cal, p_test, alpha, rng) -> np.ndarray:
    """Dunn et al. single subsampling: one random non-CVS frame per calibration video;
    exact frame-level guarantee P(confirm | not CVS) <= alpha for a frame of a new video."""
    m = []
    for v in np.unique(vid_cal):
        idx = np.flatnonzero((vid_cal == v) & ~cvs_cal)
        if len(idx):
            m.append(p_cal[rng.choice(idx)])
    m = np.sort(m); k = math.ceil((len(m) + 1) * (1 - alpha))
    if len(m) == 0 or k > len(m):
        return np.zeros(len(p_test), bool)
    return p_test > m[k - 1]


# ------------------------------------------------------------------ further baselines (video-aware)
def aci_confirm(p_cal_neg: np.ndarray, p_seq: np.ndarray, y_seq: np.ndarray, alpha: float,
                gamma: float = 0.005) -> np.ndarray:
    """Adaptive conformal inference (Gibbs & Candes, NeurIPS 2021) on a test stream.
    Threshold = (1 - alpha_t) quantile of the calibration non-CVS confidences; after each
    frame the true label is revealed (generous: unavailable in surgery) and
    alpha_{t+1} = alpha_t + gamma (alpha - err_t), err_t = 1[confirmed and not CVS].
    Guarantees the long-run average frame error -> alpha; nothing per operation."""
    v0 = np.sort(p_cal_neg); n0 = len(v0)
    a_t, out = alpha, np.zeros(len(p_seq), bool)
    for t, (p, y) in enumerate(zip(p_seq, y_seq)):
        if a_t <= 0:
            lam = np.inf
        elif a_t >= 1:
            lam = -np.inf
        else:
            k = math.ceil((n0 + 1) * (1 - a_t))
            lam = np.inf if k > n0 else v0[k - 1]
        out[t] = p > lam
        a_t += gamma * (alpha - float(out[t] and not y))
    return out


def repeated_subsampling_confirm(p_cal, cvs_cal, vid_cal, p_test, alpha, rng, B: int = 50) -> np.ndarray:
    """Dunn, Wasserman & Ramdas (JASA 2023), repeated subsampling: average over B subsamples
    of one non-CVS frame per calibration video of the conformal p-value; confirm iff the
    average is <= alpha. Guarantee P(confirm | not CVS) <= 2 alpha for a frame of a new video
    (run it at alpha/2 for a level-alpha guarantee)."""
    vids = [np.flatnonzero((vid_cal == v) & ~cvs_cal) for v in np.unique(vid_cal)]
    vids = [ix for ix in vids if len(ix)]
    V = len(vids)
    pv = np.zeros(len(p_test))
    for _ in range(B):
        s = np.sort([p_cal[rng.choice(ix)] for ix in vids])
        # p-value of a test confidence p: (1 + #{s >= p}) / (V + 1)
        pv += (1 + (V - np.searchsorted(s, p_test, side="left"))) / (V + 1)
    return pv / B <= alpha


def sgr_confirm(p_cal, cvs_cal, p_test, risk: float, delta: float = 0.1) -> np.ndarray:
    """Selection with guaranteed risk (Geifman & El-Yaniv, NeurIPS 2017): the lowest
    confidence threshold whose selective risk P(not CVS | confirmed) has a binomial upper
    confidence bound <= risk, with a Bonferroni correction over the thresholds tried
    (as in the original binary search). Assumes i.i.d. frames."""
    from scipy.stats import beta
    order = np.argsort(-p_cal); ps, ys = p_cal[order], cvs_cal[order]
    cands = np.unique(np.linspace(10, len(ps), 40).astype(int))
    best = np.inf
    for m in cands:                                   # accept the top-m calibration frames
        errs = int((~ys[:m]).sum())
        ub = beta.ppf(1 - delta / len(cands), errs + 1, m - errs) if errs < m else 1.0
        if ub <= risk:
            best = ps[m - 1]
    return p_test >= best if np.isfinite(best) else np.zeros(len(p_test), bool)


def video_kth_confirm(p_cal, cvs_cal, vid_cal, p_test, alpha, k: int = 0) -> np.ndarray:
    """k-tolerant surgery-level certificate:  P(more than k false confirmations in a new
    operation) <= alpha.  Score per calibration operation = its (k+1)-th largest confidence
    on a non-CVS frame (-inf if it has <= k such frames). k = 0 is video_max_confirm."""
    m = []
    for v in np.unique(vid_cal):
        s = np.sort(p_cal[(vid_cal == v) & ~cvs_cal])[::-1]
        m.append(s[k] if len(s) > k else -np.inf)
    m = np.sort(m); kk = math.ceil((len(m) + 1) * (1 - alpha))
    if len(m) == 0 or kk > len(m):
        return np.zeros(len(p_test), bool)
    return p_test > m[kk - 1]
