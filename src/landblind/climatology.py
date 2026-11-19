"""Day-of-year climatologies with circular windows.

The calendar has 365 days; 29 February is treated as 28 February.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_DOY = 365


def noleap_doy(times: pd.DatetimeIndex) -> np.ndarray:
    """Day of year in 1..365, with 29 February mapped to 28 February."""
    doy = np.asarray(times.day_of_year, dtype=np.int64).copy()
    leap_after = np.asarray(times.is_leap_year) & (doy >= 60)
    doy[leap_after] -= 1
    return doy


def _doy_sums(x: np.ndarray, doy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-day-of-year count, sum and sum of squares of ``x`` (time first), NaNs ignored."""
    shape = (N_DOY, *x.shape[1:])
    count = np.zeros(shape)
    total = np.zeros(shape)
    total_sq = np.zeros(shape)
    for d in range(1, N_DOY + 1):
        rows = x[doy == d].astype(np.float64)
        finite = np.isfinite(rows)
        rows = np.where(finite, rows, 0.0)
        count[d - 1] = finite.sum(axis=0)
        total[d - 1] = rows.sum(axis=0)
        total_sq[d - 1] = (rows * rows).sum(axis=0)
    return count, total, total_sq


def _circular_window_sum(a: np.ndarray, half_window: int) -> np.ndarray:
    out = np.zeros_like(a)
    for k in range(-half_window, half_window + 1):
        out += np.roll(a, k, axis=0)
    return out


def doy_mean_sd(x: np.ndarray, doy: np.ndarray, half_window: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of anomalies for each day of year, ±``half_window`` window.

    The mean is the window mean. The standard deviation is computed from anomalies about each
    day's own window mean, so the slope of the seasonal cycle inside the window does not inflate
    it. Returns arrays of shape (365, *x.shape[1:]); index ``doy - 1``.
    """
    count, total, _ = (_circular_window_sum(a, half_window) for a in _doy_sums(x, doy))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    anom = x - mean[doy - 1]
    count_a, _, total_sq = (_circular_window_sum(a, half_window) for a in _doy_sums(anom, doy))
    with np.errstate(invalid="ignore", divide="ignore"):
        var = total_sq / (count_a - 1)
    return mean, np.sqrt(var)


def doy_quantile(x: np.ndarray, doy: np.ndarray, q: float, half_window: int) -> np.ndarray:
    """Quantile ``q`` for each day of year over a ±``half_window`` window, shape (365, ...)."""
    out = np.empty((N_DOY, *x.shape[1:]), dtype=np.float32)
    for d in range(1, N_DOY + 1):
        dist = np.abs(((doy - d) + N_DOY // 2) % N_DOY - N_DOY // 2)
        out[d - 1] = np.nanquantile(x[dist <= half_window], q, axis=0)
    return out


def warmest_months(monthly_mean: np.ndarray, n_months: int) -> np.ndarray:
    """Start month (1..12) of the warmest run of ``n_months`` consecutive months, per column.

    ``monthly_mean`` has shape (12, ...); the result has shape ``monthly_mean.shape[1:]``.
    """
    run = sum(np.roll(monthly_mean, -k, axis=0) for k in range(n_months)) / n_months
    return np.argmax(run, axis=0) + 1


def in_season(month: np.ndarray, start_month: np.ndarray, n_months: int) -> np.ndarray:
    """Boolean array broadcasting calendar ``month`` against each cell's season start."""
    return ((np.asarray(month)[..., None] - start_month) % 12) < n_months
