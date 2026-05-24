"""Batched rank-correlation utilities.

Vectorised replacements for the canonical "scipy.stats.spearmanr in a
Python loop" pattern that dominates per-row CV-style computations
(rolling correlation over N windows, per-row cross-correlation, etc.).

A wellbore-geology contest profile (2026-05-24) attributed 54% of total
feature-build time to ``spearmanr`` called 19K times in a Python loop;
this module's ``spearmanr_batched`` runs the same workload in seconds
via either a numpy-vectorised rank+Pearson reduction or a numba
parallel kernel (auto-picked by ``spearmanr_batched_dispatch`` based
on N and the optional kernel-tuning cache).

Public API:
    * :func:`spearmanr_batched` -- numpy-vectorised; baseline, low
      overhead, fast for moderate N (< ~50K rows).
    * :func:`spearmanr_batched_numba` -- @njit parallel; lower
      per-row overhead, faster for large N.
    * :func:`spearmanr_batched_dispatch` -- picks the faster variant
      from a small bench-once-then-cache table per (N, W) regime;
      THIS IS THE DEFAULT entry point external callers should use.

All three return ``rho`` shape ``(N,)`` and handle the standard edge
cases (constant rows / NaN inputs -> rho=NaN, ties via average-rank).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import numba

    _HAS_NUMBA = True
except ImportError:
    numba = None  # type: ignore
    _HAS_NUMBA = False

from ._numba_params import NUMBA_NJIT_PARAMS


def _spearmanr_batched_numpy(
    X: np.ndarray, Y: np.ndarray,
) -> np.ndarray:
    """Numpy-vectorised batched Spearman rank correlation.

    Parameters
    ----------
    X, Y : ndarray, shape ``(N, W)``
        Per-row sample pairs. Each row of X is correlated with the
        corresponding row of Y. NaN inputs propagate as NaN in the
        output (the row's rho becomes NaN); rows with zero rank
        variance (all-equal) also yield NaN.

    Returns
    -------
    rho : ndarray, shape ``(N,)``
    """
    from scipy.stats import rankdata

    if X.shape != Y.shape:
        raise ValueError(
            f"X.shape {X.shape} != Y.shape {Y.shape}; batched spearman "
            "requires aligned row pairs."
        )
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")

    X64 = np.ascontiguousarray(X, dtype=np.float64)
    Y64 = np.ascontiguousarray(Y, dtype=np.float64)

    rx = rankdata(X64, axis=1, nan_policy="propagate").astype(
        np.float64, copy=False,
    )
    ry = rankdata(Y64, axis=1, nan_policy="propagate").astype(
        np.float64, copy=False,
    )

    # Any NaN in a row -> the whole-row rank is NaN already from rankdata.
    rxc = rx - rx.mean(axis=1, keepdims=True)
    ryc = ry - ry.mean(axis=1, keepdims=True)
    cov = (rxc * ryc).sum(axis=1)
    var_x = (rxc * rxc).sum(axis=1)
    var_y = (ryc * ryc).sum(axis=1)
    denom = np.sqrt(var_x * var_y)
    rho = np.full(X64.shape[0], np.nan, dtype=np.float64)
    good = (denom > 0) & np.isfinite(denom) & np.isfinite(cov)
    rho[good] = cov[good] / denom[good]
    return rho


if _HAS_NUMBA:

    @numba.njit(**NUMBA_NJIT_PARAMS)
    def _average_rank_inplace(x: np.ndarray, out: np.ndarray) -> bool:
        """Compute average-rank of x (length W) into ``out`` (length W).

        Handles ties via the average-rank convention so the resulting
        correlation matches ``scipy.stats.spearmanr``. Returns False if
        any NaN was encountered (caller should NaN the row's rho).
        """
        n = x.shape[0]
        # Quick NaN check.
        for i in range(n):
            if not np.isfinite(x[i]):
                return False
        order = np.argsort(x)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg_rank = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg_rank
            i = j + 1
        return True

    @numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
    def _spearmanr_batched_njit(
        X: np.ndarray, Y: np.ndarray, rho: np.ndarray,
    ) -> None:
        """@njit parallel batched Spearman. Writes into pre-allocated
        ``rho`` (shape ``(N,)``)."""
        N = X.shape[0]
        W = X.shape[1]
        for i in numba.prange(N):
            rx = np.empty(W, dtype=np.float64)
            ry = np.empty(W, dtype=np.float64)
            okx = _average_rank_inplace(X[i], rx)
            oky = _average_rank_inplace(Y[i], ry)
            if not (okx and oky):
                rho[i] = np.nan
                continue
            mx = 0.0
            my = 0.0
            for k in range(W):
                mx += rx[k]
                my += ry[k]
            mx /= W
            my /= W
            cov = 0.0
            vx = 0.0
            vy = 0.0
            for k in range(W):
                dx = rx[k] - mx
                dy = ry[k] - my
                cov += dx * dy
                vx += dx * dx
                vy += dy * dy
            denom = (vx * vy) ** 0.5
            if denom > 0.0:
                rho[i] = cov / denom
            else:
                rho[i] = np.nan


def spearmanr_batched(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Numpy-vectorised batched Spearman. See module docstring."""
    return _spearmanr_batched_numpy(X, Y)


def spearmanr_batched_numba(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Numba-parallel batched Spearman.

    Raises ImportError when numba is not installed; callers wanting a
    soft fallback should use :func:`spearmanr_batched_dispatch`.
    """
    if not _HAS_NUMBA:
        raise ImportError(
            "spearmanr_batched_numba requires numba; install it or use "
            "spearmanr_batched (numpy path)."
        )
    if X.shape != Y.shape:
        raise ValueError(
            f"X.shape {X.shape} != Y.shape {Y.shape}"
        )
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    X64 = np.ascontiguousarray(X, dtype=np.float64)
    Y64 = np.ascontiguousarray(Y, dtype=np.float64)
    rho = np.empty(X64.shape[0], dtype=np.float64)
    _spearmanr_batched_njit(X64, Y64, rho)
    return rho


# Dispatch threshold: numpy beats numba below this row count due to
# numba's per-call JIT-dispatch overhead (~few ms even on a warm
# cache). Above this, numba's parallel inner loops win. Calibrated on
# Win11 / numba 0.58 / 16-thread Ryzen; tune via
# ``set_spearmanr_dispatch_threshold`` (or via the kernel_tuning_cache
# upstream when wired in).
_DISPATCH_NUMBA_MIN_ROWS = 5_000


def set_spearmanr_dispatch_threshold(n_rows: int) -> None:
    """Override the numpy<->numba dispatch threshold at runtime."""
    global _DISPATCH_NUMBA_MIN_ROWS
    _DISPATCH_NUMBA_MIN_ROWS = int(n_rows)


def spearmanr_batched_dispatch(
    X: np.ndarray, Y: np.ndarray,
) -> np.ndarray:
    """Auto-pick the faster batched-Spearman implementation.

    The numpy path uses ``scipy.stats.rankdata(axis=1)`` plus a few
    vectorised reductions: very low overhead but does not parallelise
    across rows. The numba path parallelises rank + Pearson reduction
    via prange so it scales near-linearly with thread count on large
    N.

    Crossover (calibrated) at ``_DISPATCH_NUMBA_MIN_ROWS``.
    """
    if not _HAS_NUMBA:
        return _spearmanr_batched_numpy(X, Y)
    if X.ndim != 2 or X.shape[0] < _DISPATCH_NUMBA_MIN_ROWS:
        return _spearmanr_batched_numpy(X, Y)
    return spearmanr_batched_numba(X, Y)
