"""Structure-identification separation vs brute force on small scenes."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cdss import anatomy as A  # noqa: E402


def brute(Preg, p3, q, region, target, require_cvs, use_K):
    R, K = Preg.shape
    best = None
    cvs_opts = list(itertools.product([0, 1], repeat=3)) if p3 is not None else [None]
    for y in itertools.product(range(K), repeat=R):
        y = np.array(y)
        if use_K and not A.in_K(y):
            continue
        for c in cvs_opts:
            cc = None if c is None else np.array(c, bool)
            s = A.score(Preg, y, p3, cc)
            if s > q:
                continue
            viol = y[region] != target or (require_cvs and cc is not None and not cc.all())
            if viol:
                best = s if best is None else min(best, s)
    return best


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("use_K", [False, True])
@pytest.mark.parametrize("require_cvs", [False, True])
def test_separation_matches_bruteforce(seed, use_K, require_cvs):
    rng = np.random.default_rng(seed)
    R = 3
    Preg = rng.dirichlet(np.full(7, 0.3), size=R)
    Preg[0] = 0.02; Preg[0, A.DUCT] = 0.88                     # a confident duct candidate
    Preg[0] /= Preg[0].sum()
    if seed % 3 == 0:
        Preg[1] = 0.02; Preg[1, A.DUCT] = 0.6; Preg[1] /= Preg[1].sum()   # a rival duct (K matters)
    p3 = rng.uniform(0.05, 0.95, 3)
    s_min = min(A.score(Preg, np.array(y), p3, np.array(c, bool))
                for y in itertools.product(range(7), repeat=R) if not use_K or A.in_K(np.array(y))
                for c in itertools.product([0, 1], repeat=3))
    for dq in (0.1, 1.0, 3.0):
        q = s_min + dq
        exp = brute(Preg, p3, q, 0, A.DUCT, require_cvs, use_K)
        got = A.separate(Preg, p3, q, 0, A.DUCT, require_cvs, use_K)
        assert got.violating_world_exists == (exp is not None), (dq, exp, got.status)
        if exp is not None and got.status == "COUNTEREXAMPLE":
            assert abs(got.world_score - exp) < 1e-3


def test_region_truth_one_to_one():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], float)
    gt = np.array([[0, 0, 10, 10]], float)
    y = A.region_truth(boxes, gt, np.array([A.DUCT]))
    assert y.tolist() == [A.DUCT, 0, 0]


def test_projected_certify_product_set():
    P = np.zeros((2, 7))
    P[0] = 0.05 / 6; P[0, A.DUCT] = 0.95                   # duct candidate: P(not duct) = 0.05
    P[1] = 0.45 / 6; P[1, A.ARTERY] = 0.55                 # artery candidate: P(not artery) = 0.45
    cand, prob = A.projected_bits(P, None)
    q = -np.log(0.2)                          # bit may be 0 iff P(bit=0) >= 0.2
    cert = A.projected_certify(cand, prob, q)
    assert cert[A.DUCT] == 0 and cert[A.ARTERY] is None


def test_projected_frame_guarantee_simulation():
    """Frame-level error <= alpha when bits are drawn from the model (exchangeable)."""
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(200):
        def draw(n):
            out = []
            for _ in range(n):
                P = rng.dirichlet(np.full(7, 0.4), size=3)
                cand, prob = A.projected_bits(P, None)
                y = np.array([rng.choice(7, p=p) for p in P])
                out.append((cand, prob, A.projected_truth(y, cand, None)))
            return out
        cal, test = draw(300), draw(300)
        s = np.sort([A.projected_score(p, b) for _, p, b in cal])
        q = s[int(np.ceil(301 * 0.9)) - 1]
        errs.append(np.mean([any(r is not None and not b[t] for t, r in A.projected_certify(c, p, q).items())
                             for c, p, b in test]))
    assert np.mean(errs) <= 0.10
