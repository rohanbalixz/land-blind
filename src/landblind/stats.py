"""Estimators and inference (preregistration §5).

Slopes come from area-weighted least squares with fixed effects, computed by the
Frisch–Waugh–Lovell theorem: the regressor of interest and the outcomes are demeaned by
alternating projections and the controls are partialled out once on the full sample.
Every statistic is a ratio of date-level sums, so a moving-block bootstrap over dates reduces
to multiplying a (replicate x date) count matrix with those sums. This keeps all cells and lead
days of a date together and preserves their spatial and cross-lead dependence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.stats import norm

Z80 = norm.ppf(0.80)


# --------------------------------------------------------------------------------------------
# Sample and fixed effects
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    """Observations (date t, cell c) selected by a boolean mask of shape (n_dates, n_cells)."""

    t: np.ndarray
    c: np.ndarray
    w: np.ndarray
    n_dates: int
    n_cells: int

    @classmethod
    def from_mask(cls, mask: np.ndarray, cell_weight: np.ndarray) -> Sample:
        t, c = np.nonzero(mask)
        return cls(t, c, cell_weight[c].astype(np.float64), mask.shape[0], mask.shape[1])

    @property
    def n(self) -> int:
        return int(self.t.size)

    def gather(self, a: np.ndarray) -> np.ndarray:
        """(n_dates, n_cells) -> (n,), or (n_dates, L, n_cells) -> (n, L)."""
        if a.ndim == 2:
            return a[self.t, self.c].astype(np.float64)
        return a[self.t, :, self.c].astype(np.float64)

    def date_sum(self, v: np.ndarray) -> np.ndarray:
        """Weighted sums by date: (n,) -> (n_dates,), (n, L) -> (n_dates, L)."""
        if v.ndim == 1:
            return np.bincount(self.t, weights=self.w * v, minlength=self.n_dates)
        return _group_matrix(self.t, self.w, self.n_dates) @ v


def _group_matrix(g: np.ndarray, w: np.ndarray, n_groups: int | None = None) -> sparse.csr_matrix:
    """Sparse (group x observation) matrix of weights, so that ``G @ x`` sums ``w * x`` by group."""
    n_groups = int(g.max()) + 1 if n_groups is None else n_groups
    return sparse.csr_matrix((w, (g, np.arange(g.size))), shape=(n_groups, g.size))


def demean(x: np.ndarray, groups: Sequence[np.ndarray], w: np.ndarray, *, tol: float = 1e-10,
           max_iter: int = 5000) -> np.ndarray:
    """Weighted projection of ``x`` (n,) or (n, k) off the fixed effects in ``groups``.

    Uses alternating projections (the method of Gaure, 2013, used in two-way fixed-effects
    software); exact in one pass for a single set of fixed effects. Group means of all columns are
    computed at once with sparse group matrices.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    projections = []
    for g in groups:
        G = _group_matrix(g, w)
        sw = np.asarray(G.sum(axis=1)).ravel()
        projections.append((G, np.where(sw > 0, sw, 1.0), g))
    scale = max(float(np.abs(x).max()), 1.0)
    for _ in range(max_iter):
        largest = 0.0
        for G, sw, g in projections:
            means = (G @ x) / sw[:, None]
            x -= means[g]
            largest = max(largest, float(np.abs(means).max()))
        if len(groups) == 1 or largest < tol * scale:
            break
    else:
        raise RuntimeError("fixed-effects demeaning did not converge")
    return x[:, 0] if squeeze else x


def fe_groups(sample: Sample, kind: str, month_of_date: np.ndarray | None = None
              ) -> list[np.ndarray]:
    """Fixed-effect group indices for the registered specifications."""
    if kind == "cell+date":
        return [sample.c, sample.t]
    if kind == "cellmonth+date":
        if month_of_date is None:
            raise ValueError("month_of_date required")
        cm = sample.c * 12 + (month_of_date[sample.t] - 1)
        _, cm = np.unique(cm, return_inverse=True)
        return [cm, sample.t]
    if kind == "date":
        return [sample.t]
    raise ValueError(kind)


# --------------------------------------------------------------------------------------------
# Slopes as ratios of date-level sums
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SlopeSums:
    """Date-level numerator N (n_dates, L) and denominator D (n_dates,) of an FWL slope."""

    N: np.ndarray
    D: np.ndarray

    @property
    def beta(self) -> np.ndarray:
        return self.N.sum(axis=0) / self.D.sum()

    def boot(self, counts: np.ndarray) -> np.ndarray:
        return (counts @ self.N) / (counts @ self.D)[:, None]


def slope_sums(sample: Sample, y: np.ndarray, x: np.ndarray, controls: np.ndarray | None,
               groups: Sequence[np.ndarray]) -> SlopeSums:
    """FWL pieces for the slope of each column of ``y`` (n, L) on ``x`` (n,) given fixed effects
    and ``controls`` (n, k).

    Returns N[t, l] = sum over date t of w * x_res * y_res[:, l] and D[t] = sum of w * x_res**2,
    where x_res is x net of fixed effects and controls and y_res is the full-model residual plus
    the x term (so that sum_t N / sum_t D is the regression coefficient).
    """
    w = sample.w
    yy = demean(y, groups, w)
    xx = demean(x, groups, w)
    if controls is not None and controls.size:
        aa = demean(controls, groups, w)
        design = np.column_stack([xx, aa])
        wd = design * w[:, None]
        coef = np.linalg.solve(design.T @ wd, wd.T @ yy)  # (1 + k, L)
        yy = yy - aa @ coef[1:]
        gamma = np.linalg.solve(aa.T @ (aa * w[:, None]), (aa * w[:, None]).T @ xx)
        xx = xx - aa @ gamma
    N = sample.date_sum(xx[:, None] * yy)
    D = sample.date_sum(xx * xx)
    return SlopeSums(N=N, D=D)


# --------------------------------------------------------------------------------------------
# Resampling and standard errors
# --------------------------------------------------------------------------------------------

def block_counts(n_dates: int, block: int, n_boot: int, seed: int) -> np.ndarray:
    """Circular moving-block bootstrap counts, shape (n_boot, n_dates)."""
    rng = np.random.default_rng(seed)
    n_blocks = -(-n_dates // block)
    starts = rng.integers(0, n_dates, size=(n_boot, n_blocks))
    idx = ((starts[..., None] + np.arange(block)) % n_dates).reshape(n_boot, -1)[:, :n_dates]
    counts = np.zeros((n_boot, n_dates), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(n_boot), n_dates), idx.ravel()), 1.0)
    return counts


def driscoll_kraay_se(scores: np.ndarray, denominator: float, lags: int) -> float:
    """Driscoll–Kraay standard error from time-ordered date-level scores (Bartlett kernel)."""
    u = np.asarray(scores, dtype=np.float64)
    gamma0 = float(u @ u)
    total = gamma0
    for j in range(1, lags + 1):
        total += 2.0 * (1.0 - j / (lags + 1.0)) * float(u[j:] @ u[:-j])
    return float(np.sqrt(max(total, 0.0)) / abs(denominator))


def dk_se_mean_slope(sums: SlopeSums, cols: np.ndarray, lags: int) -> float:
    """Driscoll–Kraay SE of the mean of the slopes in ``cols``."""
    beta = sums.beta
    u = (sums.N[:, cols] - sums.D[:, None] * beta[cols]).mean(axis=1)
    return driscoll_kraay_se(u, sums.D.sum(), lags)


# --------------------------------------------------------------------------------------------
# Summaries and decisions
# --------------------------------------------------------------------------------------------

def summarize(value: float, boot: np.ndarray, *, sesoi: float | None = None,
              se_dk: float | None = None, alternative: str = "less") -> dict:
    """Point estimate, bootstrap SE and percentile intervals, z-test p-value and MDE."""
    boot = np.asarray(boot, dtype=np.float64)
    boot = boot[np.isfinite(boot)]
    se = float(boot.std(ddof=1))
    z = value / se if se > 0 else np.nan
    if alternative == "less":
        p = float(norm.cdf(z))
    elif alternative == "greater":
        p = float(norm.sf(z))
    else:
        p = float(2 * norm.sf(abs(z)))
    out: dict[str, Any] = {
        "estimate": float(value),
        "se_boot": se,
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "ci90": [float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))],
        "z": float(z),
        "p_one_sided" if alternative != "two-sided" else "p_two_sided": p,
        "mde80": float((norm.ppf(0.95) + Z80) * se),
        "n_boot": int(boot.size),
    }
    if se_dk is not None:
        out["se_driscoll_kraay"] = float(se_dk)
    if sesoi is not None:
        lo, hi = out["ci90"]
        out["equivalent_within_sesoi"] = bool(-sesoi < lo and hi < sesoi)
    return out


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()


def verdict(adjusted_p: float, equivalent: bool, alpha: float) -> str:
    """Registered three-way reading of a confirmatory test (preregistration §5.4)."""
    if adjusted_p < alpha:
        return "supported"
    if equivalent:
        return "equivalent to zero"
    return "inconclusive"
