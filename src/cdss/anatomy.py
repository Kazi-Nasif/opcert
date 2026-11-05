"""Certified structure identification for clipping.

World for one frame with R candidate regions:
    y_r in {0 = background, 1..6 = Endoscapes classes}    identity of each region
    c in {0,1}^3                                           CVS criteria
Domain constraint K (Endoscapes annotation protocol, verified on the ground truth):
    at most one region per anatomical structure (cystic plate, hepatocystic triangle,
    cystic artery, cystic duct, gallbladder); tools are unconstrained.
Score (deliberately additive; split conformal only needs exchangeability):
    s(x, z) = sum_r -log P_r(y_r) + sum_j -log p_j(c_j)

Certifying region r as structure s ("this is the cystic duct, safe to clip") is violated
by a world z if y_r != s, or (when the clip rule is on) if CVS fails in z. A certificate is
issued only if NO world in Gamma ∩ K violates it. Because a certified *set* is violated
iff one of its members is, and "for all z" distributes over conjunction, the set of all
individually certified (r, s) is certified jointly: with probability >= 1 - alpha every
overlay shown in the frame is correct at once (frame-level family-wise guarantee).

Separation (does a violating plausible world exist?) is one CP-SAT model over the
7^R x 8 world space with the at-most-one constraints; it is never enumerated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model

EPS = 1e-6
SCALE = 10_000
CLASSES = ["background", "cystic_plate", "calot_triangle", "cystic_artery", "cystic_duct", "gallbladder", "tool"]
ARTERY, DUCT = 3, 4
UNIQUE = (1, 2, 3, 4, 5)          # at most one region each
CLIP_TARGETS = (DUCT, ARTERY)


# ------------------------------------------------------------------ ground truth
def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def region_truth(boxes: np.ndarray, gt_boxes: np.ndarray, gt_labels: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Greedy one-to-one matching (highest IoU first). Unmatched regions are background,
    so each ground-truth structure is held by at most one region (K holds for the truth)."""
    y = np.zeros(len(boxes), int)
    if len(boxes) == 0 or len(gt_boxes) == 0:
        return y
    M = iou(boxes, gt_boxes)
    used_r, used_g = set(), set()
    for flat in np.argsort(-M, axis=None):
        r, g = np.unravel_index(flat, M.shape)
        if M[r, g] < thr:
            break
        if r in used_r or g in used_g:
            continue
        y[r] = gt_labels[g]; used_r.add(r); used_g.add(g)
    return y


def temper(P: np.ndarray, T: float) -> np.ndarray:
    L = np.log(np.clip(P, EPS, 1)) / T
    L -= L.max(-1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(-1, keepdims=True)


# ------------------------------------------------------------------ score
def score(Preg: np.ndarray, y: np.ndarray, p3: np.ndarray | None = None, c: np.ndarray | None = None) -> float:
    s = float(-np.log(np.clip(Preg[np.arange(len(y)), y], EPS, 1)).sum()) if len(y) else 0.0
    if p3 is not None:
        p3 = np.clip(p3, EPS, 1 - EPS)
        s += float(-np.log(np.where(c, p3, 1 - p3)).sum())
    return s


def in_K(y: np.ndarray) -> bool:
    return all((y == k).sum() <= 1 for k in UNIQUE)


# ------------------------------------------------------------------ separation
@dataclass
class Sep:
    violating_world_exists: bool
    status: str
    y: np.ndarray | None = None
    c: np.ndarray | None = None
    world_score: float | None = None


def separate(Preg: np.ndarray, p3: np.ndarray | None, q_hat: float, region: int, target: int,
             require_cvs: bool, use_K: bool = True, time_limit: float = 5.0, max_blocks: int = 50) -> Sep:
    """Is there z in Gamma (∩ K) with y_region != target, or (require_cvs) CVS false?"""
    if not math.isfinite(q_hat):
        return Sep(True, "Q_INF")
    R = len(Preg)
    nll = -np.log(np.clip(Preg, EPS, 1))                       # (R, 7)
    m = cp_model.CpModel()
    Y = [[m.NewBoolVar(f"y{r}_{k}") for k in range(nll.shape[1])] for r in range(R)]
    for r in range(R):
        m.AddExactlyOne(Y[r])
    if use_K:
        for k in UNIQUE:
            m.Add(sum(Y[r][k] for r in range(R)) <= 1)
    base = nll.min(1)                                           # per-region best cost
    terms = [int(math.floor((nll[r, k] - base[r]) * SCALE)) * Y[r][k]
             for r in range(R) for k in range(nll.shape[1]) if nll[r, k] > base[r]]
    budget = q_hat - base.sum()
    C = []
    if p3 is not None:
        p = np.clip(p3, EPS, 1 - EPS)
        on, off = -np.log(p), -np.log(1 - p)
        cb = np.minimum(on, off)
        C = [m.NewBoolVar(f"c{j}") for j in range(3)]
        for j in range(3):
            if on[j] > cb[j]:
                terms.append(int(math.floor((on[j] - cb[j]) * SCALE)) * C[j])
            if off[j] > cb[j]:
                terms.append(int(math.floor((off[j] - cb[j]) * SCALE)) * C[j].Not())
        budget -= cb.sum()
    if budget < 0:
        return Sep(True, "GAMMA_EMPTY")                          # refuse: never force-include a world
    m.Add(sum(terms) <= int(math.floor(budget * SCALE)))
    viol = [Y[region][target].Not()]
    if require_cvs and C:
        viol += [C[0].Not(), C[1].Not(), C[2].Not()]
    m.AddBoolOr(viol)
    m.Minimize(sum(terms))
    for _ in range(max_blocks):
        sol = cp_model.CpSolver()
        sol.parameters.max_time_in_seconds = time_limit
        sol.parameters.num_workers = 1
        st = sol.Solve(m)
        if st == cp_model.INFEASIBLE:
            return Sep(False, "INFEASIBLE")
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return Sep(True, sol.StatusName(st))                # fail closed
        y = np.array([[sol.Value(v) for v in row].index(1) for row in Y])
        c = np.array([sol.Value(v) for v in C], bool) if C else None
        s = score(Preg, y, p3, c)
        if s <= q_hat + 1e-9:
            return Sep(True, "COUNTEREXAMPLE", y, c, s)
        lits = [Y[r][y[r]].Not() for r in range(R)] + ([C[j].Not() if c[j] else C[j] for j in range(3)] if C else [])
        m.AddBoolOr(lits)                                        # integerisation slack: block, retry
    return Sep(True, "BLOCK_LIMIT")


def certify_frame(Preg: np.ndarray, p3: np.ndarray | None, q_hat: float, require_cvs: bool,
                  use_K: bool = True) -> dict[int, int | None]:
    """For each clip target, the region certified as that structure (or None).
    Candidate = the region most likely to be that structure; others cannot all be certified
    for the same target under K anyway."""
    out = {}
    for t in CLIP_TARGETS:
        if len(Preg) == 0:
            out[t] = None; continue
        r = int(np.argmax(Preg[:, t]))
        s = separate(Preg, p3, q_hat, r, t, require_cvs, use_K)
        out[t] = None if s.violating_world_exists else r
    return out


def map_world(Preg: np.ndarray, use_K: bool = True) -> np.ndarray:
    """Most probable region labelling (under K if requested) — the point world."""
    R = len(Preg)
    if R == 0:
        return np.zeros(0, int)
    if not use_K:
        return Preg.argmax(1)
    nll = -np.log(np.clip(Preg, EPS, 1))
    m = cp_model.CpModel()
    Y = [[m.NewBoolVar("") for _ in range(nll.shape[1])] for _ in range(R)]
    for r in range(R):
        m.AddExactlyOne(Y[r])
    for k in UNIQUE:
        m.Add(sum(Y[r][k] for r in range(R)) <= 1)
    m.Minimize(sum(int(round(nll[r, k] * SCALE)) * Y[r][k] for r in range(R) for k in range(nll.shape[1])))
    sol = cp_model.CpSolver(); sol.parameters.num_workers = 1
    assert sol.Solve(m) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    return np.array([[sol.Value(v) for v in row].index(1) for row in Y])


# ------------------------------------------------------------------ decision-sufficient projection
# A certificate for target s reads only b_s = 1[y_{r_s} = s], where r_s = argmax_r P_r(s) is a
# fixed function of x; the clip rule additionally reads b_cvs = 1[CVS]. Covering the projection
# b = (b_duct, b_artery[, b_cvs]) is enough for the frame-level guarantee
#     P(some certificate shown in the frame is wrong) <= P(b_true not in Gamma_b) <= alpha,
# and it ignores the identities of regions no decision depends on (gallbladder, tools, ...),
# which is what made the full-world set uselessly large.
# Score: max over the projected bits of -log P(bit) (per-target budget, frame-level coverage).

def projected_bits(Preg: np.ndarray, p_cvs: float | None) -> tuple[dict, dict]:
    """Candidate region per target and P(candidate really is the target)."""
    cand, prob = {}, {}
    for t in CLIP_TARGETS:
        if len(Preg) == 0:
            cand[t], prob[t] = None, 0.0
        else:
            r = int(np.argmax(Preg[:, t])); cand[t], prob[t] = r, float(Preg[r, t])
    if p_cvs is not None:
        prob["cvs"] = float(p_cvs)
    return cand, prob


def projected_truth(y: np.ndarray, cand: dict, cvs_true: bool | None) -> dict:
    b = {t: (cand[t] is not None and y[cand[t]] == t) for t in CLIP_TARGETS}
    if cvs_true is not None:
        b["cvs"] = bool(cvs_true)
    return b


def projected_score(prob: dict, b: dict) -> float:
    return max(-math.log(min(max(prob[k] if b[k] else 1 - prob[k], EPS), 1.0)) for k in prob)


def projected_certify(cand: dict, prob: dict, q_hat: float) -> dict:
    """Certify target t iff every b in Gamma_b has b_t = 1 (and b_cvs = 1 when present).
    With the max score, Gamma_b is a product set: bit k may be 0 iff -log(1-p_k) <= q_hat."""
    if not math.isfinite(q_hat):
        return {t: None for t in CLIP_TARGETS}
    nonempty = all(min(-math.log(max(p, EPS)), -math.log(max(1 - p, EPS))) <= q_hat for p in prob.values())
    surely = {k: (-math.log(max(1 - p, EPS)) > q_hat and -math.log(max(p, EPS)) <= q_hat) for k, p in prob.items()}
    cvs_ok = surely.get("cvs", True)
    return {t: (cand[t] if nonempty and cand[t] is not None and surely[t] and cvs_ok else None)
            for t in CLIP_TARGETS}
