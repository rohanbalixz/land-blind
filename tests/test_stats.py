import numpy as np
import pytest

from landblind.stats import (
    Sample,
    block_counts,
    demean,
    driscoll_kraay_se,
    fe_groups,
    holm,
    slope_sums,
    summarize,
    verdict,
)


def synthetic_panel(seed=0, n_dates=60, n_cells=40, beta=-0.3, missing=0.25):
    rng = np.random.default_rng(seed)
    mask = rng.random((n_dates, n_cells)) > missing
    weight = rng.uniform(0.2, 1.0, n_cells)
    alpha = rng.normal(size=n_cells)
    tau = rng.normal(size=n_dates)
    a = rng.normal(size=(n_dates, n_cells))
    s = 0.6 * a + rng.normal(size=(n_dates, n_cells)) + alpha[None, :] + 0.5 * tau[:, None]
    y = alpha[None, :] + tau[:, None] + beta * s + 0.8 * a + rng.normal(size=(n_dates, n_cells))
    return mask, weight, s, a, y


def dummy_wls(mask, weight, s, a, y):
    t, c = np.nonzero(mask)
    n_dates, n_cells = mask.shape
    design = np.column_stack([s[t, c], a[t, c], np.eye(n_cells)[c], np.eye(n_dates)[t][:, 1:]])
    w = np.sqrt(weight[c])
    coef, *_ = np.linalg.lstsq(design * w[:, None], y[t, c] * w, rcond=None)
    return coef[0]


def test_twoway_fwl_slope_equals_dummy_variable_regression():
    mask, weight, s, a, y = synthetic_panel()
    sample = Sample.from_mask(mask, weight)
    groups = fe_groups(sample, "cell+date")
    sums = slope_sums(sample, sample.gather(y)[:, None], sample.gather(s),
                      sample.gather(a)[:, None], groups)
    assert sums.beta[0] == pytest.approx(dummy_wls(mask, weight, s, a, y), rel=1e-7)


def test_demean_removes_group_means():
    mask, weight, s, *_ = synthetic_panel(seed=1)
    sample = Sample.from_mask(mask, weight)
    x = demean(sample.gather(s), fe_groups(sample, "cell+date"), sample.w)
    for g in (sample.c, sample.t):
        means = np.bincount(g, weights=sample.w * x) / np.bincount(g, weights=sample.w)
        assert np.abs(means).max() < 1e-8


def test_slope_recovers_truth_on_large_panel():
    mask, weight, s, a, y = synthetic_panel(seed=3, n_dates=300, n_cells=200, beta=-0.25)
    sample = Sample.from_mask(mask, weight)
    sums = slope_sums(sample, sample.gather(y)[:, None], sample.gather(s),
                      sample.gather(a)[:, None], fe_groups(sample, "cell+date"))
    assert sums.beta[0] == pytest.approx(-0.25, abs=0.01)


def test_bootstrap_counts_shape_total_and_reproducibility():
    c1 = block_counts(365, 7, 50, seed=11)
    c2 = block_counts(365, 7, 50, seed=11)
    assert c1.shape == (50, 365)
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(c1.sum(axis=1), np.full(50, 365))


def test_bootstrap_with_unit_counts_reproduces_estimate():
    mask, weight, s, _, y = synthetic_panel(seed=4)
    sample = Sample.from_mask(mask, weight)
    sums = slope_sums(sample, sample.gather(y)[:, None], sample.gather(s), None,
                      fe_groups(sample, "cell+date"))
    ones = np.ones((1, mask.shape[0]))
    assert sums.boot(ones)[0, 0] == pytest.approx(sums.beta[0])


def test_bootstrap_se_is_close_to_sampling_sd():
    # Independent dates: the block bootstrap SE should track the Monte Carlo SD of the estimator.
    estimates, ses = [], []
    for seed in range(40):
        mask, weight, s, a, y = synthetic_panel(seed=100 + seed, n_dates=120, n_cells=30)
        sample = Sample.from_mask(mask, weight)
        sums = slope_sums(sample, sample.gather(y)[:, None], sample.gather(s),
                          sample.gather(a)[:, None], fe_groups(sample, "cell+date"))
        estimates.append(sums.beta[0])
        ses.append(sums.boot(block_counts(120, 1, 400, seed))[:, 0].std(ddof=1))
    assert np.mean(ses) == pytest.approx(np.std(estimates, ddof=1), rel=0.25)


def test_driscoll_kraay_without_lags_is_sum_of_squares():
    u = np.array([1.0, -2.0, 0.5, 3.0])
    assert driscoll_kraay_se(u, 2.0, 0) == pytest.approx(np.sqrt((u**2).sum()) / 2.0)


def test_holm_matches_hand_computation():
    assert holm([0.01, 0.04, 0.03, 0.2]) == pytest.approx([0.04, 0.09, 0.09, 0.2])


def test_summarize_and_verdict():
    rng = np.random.default_rng(5)
    res = summarize(-0.3, rng.normal(-0.3, 0.05, 5000), sesoi=0.1)
    assert res["p_one_sided"] < 1e-6
    assert not res["equivalent_within_sesoi"]
    assert verdict(res["p_one_sided"], res["equivalent_within_sesoi"], 0.05) == "supported"
    null = summarize(0.0, rng.normal(0.0, 0.02, 5000), sesoi=0.1)
    assert null["equivalent_within_sesoi"]
    assert verdict(0.5, null["equivalent_within_sesoi"], 0.05) == "equivalent to zero"
    wide = summarize(0.0, rng.normal(0.0, 0.2, 5000), sesoi=0.1)
    assert verdict(0.5, wide["equivalent_within_sesoi"], 0.05) == "inconclusive"
    assert wide["mde80"] == pytest.approx(2.486 * wide["se_boot"], rel=1e-3)
