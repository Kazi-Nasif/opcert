import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cdss import riskcontrol as RC  # noqa: E402


def _sim(fn, alpha=0.1, reps=300, seed=0):
    rng = np.random.default_rng(seed)
    risks = []
    for _ in range(reps):
        def draw(n):
            y = rng.random(n) < 0.3
            s = np.where(y, rng.beta(5, 2, n), rng.beta(2, 5, n))
            return s, y
        sc, yc = draw(400); st, yt = draw(4000)
        grid = RC.threshold_grid(sc)
        L = (sc[:, None] > grid[None]) & ~yc[:, None]
        g = fn(L, alpha); lam = np.inf if g is None else grid[g]
        risks.append(((st > lam) & ~yt).mean())
    return np.array(risks)


def test_crc_expected_risk():
    assert _sim(RC.crc_index).mean() <= 0.1 + 0.005


def test_ltt_high_probability():
    r = _sim(RC.ltt_index)
    assert (r <= 0.1).mean() >= 0.9 - 0.03
