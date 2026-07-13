"""``seasonal_residual`` composite transform.

``T = y - seasonal_mean(phase)`` with ``phase = row_index % period``: the fit learns per-phase means of the train target and the forward subtracts each
row's phase mean; the inverse adds it back. ``period`` is either supplied via a fit kwarg or selected on train by minimising the residual variance over a
small candidate grid (``_SEASONAL_PERIOD_CANDIDATES`` capped at ``n // 3`` so every candidate sees at least ~3 full cycles).

Index-position phase assumption. Like ``ewma_residual``, the transform is defined on the ROW SEQUENCE: phase is the row's position modulo ``period``
within the batch it is evaluated on, NOT a calendar field. The caller is responsible for chronological, gap-free row order at fit and predict, and a
predict batch is assumed to start at phase 0 (the same stateless-batch convention the EWMA anchor uses). Given the phase, the transform is pointwise
(no neighbour reads), so ``recurrent=False`` is correct: dropping a row shifts later phases exactly as it would shift them in the caller's own frame.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_SEASONAL_PERIOD_CANDIDATES: tuple[int, ...] = (4, 5, 7, 12, 24, 52)
"""Candidate seasonal periods for the fit-time grid search: intra-month week (4/5), week of daily data (7), months (12), hours (24), weeks of year (52)."""


def _seasonal_phase_means(y_f: np.ndarray, period: int) -> tuple[np.ndarray, float]:
    """Per-phase means of ``y_f`` (NaN-aware) + the residual variance after subtracting them. Empty phases fall back to the global mean."""
    n = y_f.size
    phase = np.arange(n, dtype=np.int64) % period
    finite = np.isfinite(y_f)
    global_mean = float(np.mean(y_f[finite])) if finite.any() else 0.0
    sums = np.bincount(phase[finite], weights=y_f[finite], minlength=period)
    counts = np.bincount(phase[finite], minlength=period)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), global_mean)
    resid = y_f[finite] - means[phase[finite]]
    var = float(np.var(resid)) if resid.size else float("inf")
    return np.asarray(means, dtype=np.float64), var


def _seasonal_residual_fit(
    y: np.ndarray, base: np.ndarray | None,
    period: int | None = None,
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Learn per-phase means. ``period`` may be supplied explicitly (like other transforms accept ``k`` / ``d`` fit kwargs); otherwise it is chosen from ``_SEASONAL_PERIOD_CANDIDATES`` (capped at n//3) by minimum residual variance, ties to the smaller period."""
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y_f.size
    if period is not None:
        period = max(1, int(period))
        means, _ = _seasonal_phase_means(y_f, period)
        return {"period": period, "phase_means": means, "y_train_mean": float(means.mean())}
    candidates = [p for p in _SEASONAL_PERIOD_CANDIDATES if p <= max(n // 3, 1)]
    if not candidates:
        # Too few rows for any seasonal cycle: degenerate to a global mean-centering (period 1).
        candidates = [1]
    best_period = candidates[0]
    best_var = float("inf")
    for p in candidates:
        means, var = _seasonal_phase_means(y_f, p)
        if var < best_var - 1e-15:
            best_var = var
            best_period = p
    means, _ = _seasonal_phase_means(y_f, best_period)
    return {"period": int(best_period), "phase_means": means, "y_train_mean": float(means.mean())}


def _seasonal_residual_forward(
    y: np.ndarray, base: np.ndarray | None, params: dict[str, Any],
) -> np.ndarray:
    """Apply ``T = y - phase_means[row_index % period]`` (phase 0 = first row of the batch)."""
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    means = np.asarray(params["phase_means"], dtype=np.float64)
    period = int(params["period"])
    phase = np.arange(y_f.size, dtype=np.int64) % period
    return np.asarray(y_f - means[phase])


def _seasonal_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray | None, params: dict[str, Any],
) -> np.ndarray:
    """Undo the transform: ``y = T_hat + phase_means[row_index % period]``."""
    t_f = np.asarray(t_hat, dtype=np.float64).reshape(-1)
    means = np.asarray(params["phase_means"], dtype=np.float64)
    period = int(params["period"])
    phase = np.arange(t_f.size, dtype=np.int64) % period
    return np.asarray(t_f + means[phase])


def _seasonal_residual_domain(
    y: np.ndarray | None, base: np.ndarray | None,
) -> np.ndarray:
    """Unary y-only domain: finite ``y`` at fit time; all-True (sized off whichever array is present) at predict time."""
    if y is None:
        n = len(base) if base is not None and hasattr(base, "__len__") else 1
        return np.ones(n, dtype=bool)
    return np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1))
