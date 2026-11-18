"""Ensemble scores used by the secondary analyses S7–S9."""

from __future__ import annotations

import numpy as np


def crps_fair(members: np.ndarray, obs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Fair (unbiased) CRPS of an ensemble (Ferro, 2014).

    CRPS = mean_i |x_i - y| - sum_{i,j} |x_i - x_j| / (2 m (m - 1)),
    computed in O(m log m) with the sorted-member identity
    sum_{i,j} |x_i - x_j| = 2 sum_k (2k - m - 1) x_(k).
    """
    x = np.moveaxis(np.asarray(members, dtype=np.float64), axis, -1)
    y = np.asarray(obs, dtype=np.float64)
    m = x.shape[-1]
    if m < 2:
        raise ValueError("fair CRPS needs at least two members")
    spread_term = np.sort(x, axis=-1) @ (2.0 * np.arange(1, m + 1) - m - 1.0) / (m * (m - 1))
    return np.abs(x - y[..., None]).mean(axis=-1) - spread_term


def ensemble_moments(members: np.ndarray, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble mean and unbiased ensemble variance."""
    x = np.asarray(members, dtype=np.float64)
    return x.mean(axis=axis), x.var(axis=axis, ddof=1)


def exceedance_probability(members: np.ndarray, threshold: np.ndarray, axis: int = -1
                           ) -> np.ndarray:
    """Fraction of members strictly above ``threshold``."""
    x = np.moveaxis(np.asarray(members), axis, -1)
    return (x > np.asarray(threshold)[..., None]).mean(axis=-1)
