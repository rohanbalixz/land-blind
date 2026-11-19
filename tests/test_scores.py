import numpy as np
import pytest

from landblind.scores import crps_fair, ensemble_moments, exceedance_probability


def brute_force_fair_crps(x: np.ndarray, y: float) -> float:
    m = x.size
    term1 = np.abs(x - y).mean()
    term2 = np.abs(x[:, None] - x[None, :]).sum() / (2 * m * (m - 1))
    return term1 - term2


def test_fair_crps_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 23))
    y = rng.normal(size=7)
    got = crps_fair(x, y, axis=1)
    want = [brute_force_fair_crps(x[i], y[i]) for i in range(7)]
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_fair_crps_member_axis_first():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 4))
    y = rng.normal(size=4)
    np.testing.assert_allclose(crps_fair(x, y, axis=0), crps_fair(x.T, y, axis=-1))


def test_fair_crps_is_unbiased_for_calibrated_ensemble():
    # For y, x_i iid N(0, 1), E[CRPS_fair] equals the CRPS of N(0, 1) at N(0, 1): 1/sqrt(pi).
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200_000, 5))
    y = rng.normal(size=200_000)
    assert crps_fair(x, y).mean() == pytest.approx(1 / np.sqrt(np.pi), abs=3e-3)


def test_fair_crps_needs_two_members():
    with pytest.raises(ValueError):
        crps_fair(np.zeros((3, 1)), np.zeros(3))


def test_moments_and_exceedance():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    mean, var = ensemble_moments(x)
    assert mean[0] == 2.5
    assert var[0] == pytest.approx(np.var([1, 2, 3, 4], ddof=1))
    assert exceedance_probability(x, np.array([2.0]))[0] == 0.5
