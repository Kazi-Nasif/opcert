import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cdss import decide  # noqa: E402


def test_codes_roundtrip():
    assert (decide.codes(decide.CODES) == np.arange(8)).all()
    assert decide.codes(np.array([[1, 1, 1]], bool))[0] == 7


def test_joint_from_marginals_sums_to_one():
    P = np.random.default_rng(0).uniform(0, 1, (50, 3))
    assert np.allclose(decide.joint_from_marginals(P).sum(1), 1)


def test_world_confirm_requires_every_plausible_world_to_be_cvs():
    P8 = np.full((3, 8), 1e-4)
    P8[0, 7] = 0.99                       # only CVS plausible -> confirm
    P8[1, 7], P8[1, 3] = 0.6, 0.39        # C1 might be missing -> refuse
    P8[2, 0] = 0.99                       # clearly not CVS -> refuse
    P8 /= P8.sum(1, keepdims=True)
    q = -np.log(0.3)                      # plausible = probability >= 0.3
    assert decide.world_confirm(P8, q).tolist() == [True, False, False]
    assert decide.world_confirm(P8, -np.log(0.99995)).tolist() == [False, False, False]  # empty set refuses


def test_vera_bound_on_synthetic_exchangeable_data():
    """P(confirm and not CVS) <= alpha (+ finite-sample slack) when scores are exchangeable."""
    rng = np.random.default_rng(1)
    viol = []
    for _ in range(200):
        n = 400
        logits = rng.normal(0, 2, (2 * n, 8))
        P8 = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        c = np.array([rng.choice(8, p=p) for p in P8])           # labels drawn from the model
        s = -np.log(P8[np.arange(n), c[:n]])
        q = decide.conformal_quantile(s, 0.1)
        conf = decide.world_confirm(P8[n:], q)
        viol.append((conf & (c[n:] != 7)).mean())
    assert np.mean(viol) <= 0.1


def test_conformal_selection_empty_when_no_evidence():
    P8c = np.full((100, 8), 1 / 8); P8t = np.full((10, 8), 1 / 8)
    assert not decide.conformal_selection(P8c, np.zeros(100, bool), P8t, 0.1).any()


def test_mondrian_bounds_false_reassurance_given_not_cvs():
    rng = np.random.default_rng(3)
    rates = []
    for _ in range(200):
        n = 600
        y = rng.random(2 * n) < 0.1
        score = np.where(y, rng.beta(5, 2, 2 * n), rng.beta(2, 5, 2 * n))
        P8 = np.zeros((2 * n, 8)); P8[:, 7] = score; P8[:, 0] = 1 - score
        conf = decide.mondrian_confirm(P8[:n], y[:n], P8[n:], 0.05)
        rates.append(conf[~y[n:]].mean())
    assert np.mean(rates) <= 0.05 + 0.005


def test_projected_sets_partition():
    P8 = np.zeros((3, 8)); P8[:, 7] = [0.99, 0.5, 0.01]; P8[:, 0] = 1 - P8[:, 7]
    q = -np.log(0.2)
    assert decide.projected_confirm(P8, q).tolist() == [True, False, False]
    assert decide.projected_alarm(P8, q).tolist() == [False, False, True]


def test_mondrian_fails_closed_without_calibration_examples():
    P8c = np.full((50, 8), 1 / 8); P8t = np.full((5, 8), 1 / 8)
    assert not decide.mondrian_alarm(P8c, np.zeros(50, bool), P8t, 0.1).any()     # no CVS examples
    assert not decide.mondrian_confirm(P8c, np.ones(50, bool), P8t, 0.1).any()    # no non-CVS examples


def test_video_max_surgery_level_guarantee():
    """Videos with strongly dependent frames: P(any false confirm in a new video) <= alpha."""
    rng = np.random.default_rng(5)
    rates = []
    for _ in range(300):
        def videos(n):
            p, y, v = [], [], []
            for i in range(n):
                level = rng.normal(0, 1)                         # shared per-video effect
                f = 60
                s = 1 / (1 + np.exp(-(level + 0.3 * rng.normal(0, 1, f) - 2)))
                yy = rng.random(f) < 0.05
                p.append(s); y.append(yy); v.append(np.full(f, i))
            return np.concatenate(p), np.concatenate(y), np.concatenate(v)
        pc, yc, vc = videos(40); pt, yt, vt = videos(40)
        conf = decide.video_max_confirm(pc, yc, vc, pt, 0.1)
        rates.append(np.mean([((conf & ~yt)[vt == u]).any() for u in np.unique(vt)]))
    assert np.mean(rates) <= 0.1 + 0.01


def test_repeated_subsampling_2alpha():
    rng = np.random.default_rng(11)
    rates = []
    for _ in range(150):
        def vids(n):
            p, y, v = [], [], []
            for i in range(n):
                lev = rng.normal(0, 1); f = 40
                p.append(1 / (1 + np.exp(-(lev + 0.4 * rng.normal(0, 1, f) - 2)))); y.append(rng.random(f) < 0.05)
                v.append(np.full(f, i))
            return np.concatenate(p), np.concatenate(y), np.concatenate(v)
        pc, yc, vc = vids(40); pt, yt, vt = vids(40)
        conf = decide.repeated_subsampling_confirm(pc, yc, vc, pt, 0.05, rng, B=20)
        rates.append(conf[~yt].mean())
    assert np.mean(rates) <= 0.10 + 0.01            # 2 alpha guarantee


def test_aci_tracks_alpha_on_iid_stream():
    rng = np.random.default_rng(12)
    pc = rng.beta(2, 5, 3000)
    y = rng.random(20000) < 0.1
    p = np.where(y, rng.beta(5, 2, 20000), rng.beta(2, 5, 20000))
    conf = decide.aci_confirm(pc, p, y, 0.05, gamma=0.01)
    assert abs((conf & ~y).mean() - 0.05) < 0.01


def test_sgr_selective_risk_iid():
    rng = np.random.default_rng(13)
    viol = []
    for _ in range(100):
        def draw(n):
            y = rng.random(n) < 0.3
            return np.where(y, rng.beta(6, 2, n), rng.beta(2, 4, n)), y
        pc, yc = draw(2000); pt, yt = draw(5000)
        c = decide.sgr_confirm(pc, yc, pt, risk=0.1, delta=0.1)
        if c.any():
            viol.append((~yt[c]).mean() > 0.1)
    assert np.mean(viol) <= 0.1 + 0.05


def test_video_kth_guarantee():
    rng = np.random.default_rng(14)
    for k in (0, 2):
        rates = []
        for _ in range(200):
            def vids(n):
                p, y, v = [], [], []
                for i in range(n):
                    lev = rng.normal(0, 1); f = 50
                    p.append(1 / (1 + np.exp(-(lev + 0.4 * rng.normal(0, 1, f) - 2)))); y.append(rng.random(f) < 0.05)
                    v.append(np.full(f, i))
                return np.concatenate(p), np.concatenate(y), np.concatenate(v)
            pc, yc, vc = vids(40); pt, yt, vt = vids(40)
            conf = decide.video_kth_confirm(pc, yc, vc, pt, 0.1, k)
            rates.append(np.mean([((conf & ~yt)[vt == u]).sum() > k for u in np.unique(vt)]))
        assert np.mean(rates) <= 0.1 + 0.01
