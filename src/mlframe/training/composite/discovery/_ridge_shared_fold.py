"""One Ridge factorisation per (feature matrix, fold, thread), reused by every spec the tiny CV scores on it.

The ``"linear"`` screening family is ``SimpleImputer(mean) + Ridge(alpha=1)``, refit for every spec although only the
target differs: each fit re-imputes the fold and re-factorises the same ``X^T X + alpha I``. Ridge with an intercept is
a closed form on centred data, so the fold's column means and the Cholesky factor of its regularised Gram matrix are
computed once and each spec is one right-hand side: 32 targets on 13.3k x 60 took 83 ms against 1836 ms per-spec.

The solve runs in float64. The sklearn path inherits the matrix dtype, which is float32 for the screening block, so its
predictions carry float32 rounding (max abs 1.9e-5 against this path on the same data); against a float64 sklearn fit
the two agree to 2e-8. The cache holds only the column means and the F x F factor, never the fold's rows.
"""

from __future__ import annotations

import threading
from typing import Any, cast

import numpy as np

from ._matrix_keyed_cache import MatrixKeyedCache

_MAX_ENTRIES = 64
"""Factorisations kept across all threads; the oldest go first. Each is two length-F vectors and one F x F factor."""

_ALPHA = 1.0
"""The Ridge penalty ``_build_tiny_model('linear', ...)`` uses."""

_CACHE = MatrixKeyedCache(_MAX_ENTRIES)


class _RidgeFoldModel:
    """A fitted linear-family model: mean-impute with the fold's train means, then ``X @ w + b``."""

    def __init__(self, fill: np.ndarray, coef: np.ndarray, intercept: float) -> None:
        self.fill = fill
        self.coef = coef
        self.intercept = intercept

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict on ``x``, filling each non-finite cell with its column's train mean."""
        xi = _impute(x, self.fill)
        return cast(np.ndarray, xi @ self.coef + self.intercept)


def _impute(x: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """``x`` as float64 with non-finite cells replaced by ``fill`` column-wise (what ``SimpleImputer(mean)`` does)."""
    return _impute_in_place(np.array(x, dtype=np.float64, copy=True), fill)


def _impute_in_place(xi: np.ndarray, fill: np.ndarray, bad: np.ndarray | None = None) -> np.ndarray:
    """``_impute`` on a float64 array the caller owns: fills it in place and returns it."""
    bad = ~np.isfinite(xi) if bad is None else bad
    if bad.any():
        xi[bad] = np.broadcast_to(fill, xi.shape)[bad]
    return xi


def prune_dead() -> None:
    """Drop the entries whose matrix has been freed, so a finished phase leaves nothing of its folds resident."""
    _CACHE.prune_dead()


def _fold_factor(x: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, Any]:
    """This thread's ``(fill, column_mean, cholesky_factor)`` for the fold ``x[rows]``, computed on first use."""
    from scipy.linalg import cho_factor

    key = (threading.get_ident(), id(x), x.shape, hash(np.ascontiguousarray(rows).tobytes()))
    hit = _CACHE.get(key, x)
    if hit is not None:
        return cast(tuple, hit)
    # One float64 copy of the fold, imputed and centred in place: the where-copy, the imputed copy and the centred copy were
    # three more fold-sized float64 arrays at once (29 MB of a 122 MB rerank peak at 4k x 240), for the same numbers.
    xc = np.asarray(x[rows], dtype=np.float64)  # x[rows] is already a fresh copy
    finite = np.isfinite(xc)
    all_finite = bool(finite.all())
    fill = np.nanmean(xc if all_finite else np.where(finite, xc, np.nan), axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)  # an all-missing column imputes to 0, contributing nothing
    if not all_finite:
        _impute_in_place(xc, fill, ~finite)
    del finite
    mu = xc.mean(axis=0)
    xc -= mu
    gram = xc.T @ xc
    del xc
    gram[np.diag_indices_from(gram)] += _ALPHA
    entry = (fill, mu, cho_factor(gram))
    _CACHE.put(key, x, entry)
    return entry


def fit_ridge_on_shared_fold(x: np.ndarray, rows: np.ndarray, target: np.ndarray) -> _RidgeFoldModel:
    """Fit the linear-family model on every row of ``x[rows]`` against ``target``, reusing the fold's factorisation."""
    from scipy.linalg import cho_solve

    fill, mu, factor = _fold_factor(x, rows)
    t = np.asarray(target, dtype=np.float64)
    t_mean = float(t.mean())
    xi = _impute_in_place(np.asarray(x[rows], dtype=np.float64), fill)
    # Centring X is folded into the right-hand side: (X - mu)^T (t - t_mean) == X^T (t - t_mean), as the residual sums to 0.
    coef = cho_solve(factor, xi.T @ (t - t_mean))
    return _RidgeFoldModel(fill, coef, t_mean - float(mu @ coef))
