"""Checks of the conventions the paper states (review round 3): order-statistic rank, CRC floor,
equality of CRC over operations with the order statistic, ties, infinite thresholds, operations
without unsafe frames, PAC sample sizes, and the conditional STOP rule.

    python -m pytest tests/test_certificates.py -q
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "scripts", "scripts/review", "scripts/review2", "scripts/review3"):
    sys.path.insert(0, str(ROOT / p))

from cdss import riskcontrol as RC          # noqa: E402
from common import order_stat               # noqa: E402
from rebuttal import mmax, op_crc, pac_index  # noqa: E402


def test_rank_and_infinite_threshold():
    m = np.arange(1, 36, dtype=float)                     # V = 35
    assert order_stat(m, 0.10) == 33.0                    # ceil(36 * 0.9) = 33rd smallest
    assert order_stat(m, 0.05) == 35.0                    # ceil(36 * 0.95) = 35
    assert math.isinf(order_stat(m[:18], 0.05))           # ceil(19 * 0.95) = 19 > 18: never certify
    assert math.isinf(order_stat([], 0.1))


def test_crc_floor():
    # n0 = 35, alpha = 0.1: at most floor(36 * 0.1) - 1 = 2 negatives strictly above lambda
    v = np.arange(1, 36, dtype=float)
    lam = order_stat(v, 0.10)
    assert (v > lam).sum() == 2


def test_crc_over_operations_equals_order_statistic():
    rng = np.random.default_rng(0)
    for _ in range(200):
        V = int(rng.integers(5, 60)); a = float(rng.choice([0.05, 0.1, 0.2]))
        m = rng.random(V)
        assert op_crc(m, a) == order_stat(m, a)


def test_ties_are_conservative():
    m = np.array([0.1] * 10 + [0.5] * 25)
    lam = order_stat(m, 0.10)
    assert (m > lam).sum() <= math.floor(36 * 0.10) - 1


def test_operations_without_unsafe_frames_left_out():
    o = {"p": {1: np.array([0.9, 0.95])}, "cvs": np.array([True, True])}
    assert mmax(o, 1) == -np.inf                          # cannot err: excluded from calibration


def test_pac_sample_sizes():
    assert pac_index(21, 0.10) is None and pac_index(22, 0.10) == 22
    assert pac_index(44, 0.05) is None and pac_index(45, 0.05) == 45
    assert pac_index(10, 0.20) is None and pac_index(11, 0.20) == 11
    assert pac_index(35, 0.10) == 35                       # PAC at 0.1 equals the marginal rank at 0.05


def test_conditional_stop_rule():
    from timeout_final import policies
    mk = lambda p, b: {"p": np.array([p]), "ref": {"majority": np.array([b])}}
    # four operations with CVS: r = floor(0.2 * 5) = 1 -> stop below the smallest positive score
    cal = [mk(x, False) for x in np.linspace(0.0, 0.3, 30)] + [mk(x, True) for x in (0.5, 0.6, 0.7, 0.8)]
    tst = [mk(0.45, True), mk(0.2, False)]
    out = {name: (conf, stop) for name, conf, stop, _ in policies(cal, tst, "majority", 0.10)}
    conf, stop = out["ours"]
    assert conf[0] and not conf[1] and stop[1] and not stop[0]   # confirm-first precedence
    # three operations with CVS: r = floor(0.2 * 4) = 0 -> never stop
    cal = cal[:30] + [mk(x, True) for x in (0.5, 0.6, 0.7)]
    conf, stop = {n: (c, s) for n, c, s, _ in policies(cal, tst, "majority", 0.10)}["ours"]
    assert not stop.any()


def test_ltt_returns_none_when_infeasible():
    L = np.ones((5, 3))
    assert RC.ltt_index(L, 0.1, 0.1) is None
