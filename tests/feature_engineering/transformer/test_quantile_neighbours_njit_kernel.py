"""Regression sensors for the fused ``_weighted_quantiles_njit`` kernel in quantile_neighbours.

The kernel replaces the numpy reference (per-q ``(cdf >= q).argmax(axis=1)`` sweeps over an (n_rows, k) boolean temporary, on top of a full (n_rows, k) argsort +
cumsum) with a single prange pass per row: one argsort, one float32 cumsum, and a first-cdf>=q scan per quantile, with no (n_rows, k) temporaries and no n_qs full
array sweeps. These sensors pin (a) the kernel exists and is importable (the symbol is the sensor that fails on pre-fix code), and (b) the kernel is bit-identical
to the numpy reference on continuous, tied-y, and tied-cdf (equal-weight) inputs -- the exact tie cases where any reordering / scan-direction change would diverge.
"""
from __future__ import annotations

import numpy as np


def _numpy_reference(y_neighbors: np.ndarray, weights: np.ndarray, qs: np.ndarray) -> np.ndarray:
    n_rows, k = y_neighbors.shape
    sort_idx = np.argsort(y_neighbors, axis=1)
    rows_arange = np.arange(n_rows)[:, None]
    y_sorted = y_neighbors[rows_arange, sort_idx]
    w_sorted = weights[rows_arange, sort_idx]
    cdf = np.cumsum(w_sorted, axis=1)
    n_qs = qs.shape[0]
    out = np.zeros((n_rows, n_qs), dtype=np.float32)
    for j, q in enumerate(qs):
        idx = (cdf >= q).argmax(axis=1)
        out[:, j] = y_sorted[rows_arange.ravel(), idx]
    return out


def test_weighted_quantiles_njit_symbol_exists():
    # Fails on pre-fix code: the fused njit kernel did not exist (ImportError), proving this sensor catches a missing optimization.
    from mlframe.feature_engineering.transformer.quantile_neighbours import _weighted_quantiles_njit

    assert callable(_weighted_quantiles_njit)


def test_weighted_quantiles_njit_bit_identical_continuous():
    from mlframe.feature_engineering.transformer.quantile_neighbours import _weighted_quantiles

    rng = np.random.default_rng(7)
    n, k = 50_000, 32
    y = rng.standard_normal((n, k)).astype(np.float32)
    w = rng.random((n, k)).astype(np.float32)
    w /= w.sum(axis=1, keepdims=True)
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float32)
    np.testing.assert_array_equal(_weighted_quantiles(y, w, qs), _numpy_reference(y, w, qs))


def test_weighted_quantiles_njit_bit_identical_tied_y_and_tied_cdf():
    from mlframe.feature_engineering.transformer.quantile_neighbours import _weighted_quantiles

    rng = np.random.default_rng(1)
    n, k = 50_000, 32
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float32)
    # Heavy y ties (integer labels) -> argsort tie-ordering must match numpy.
    y = rng.integers(0, 3, size=(n, k)).astype(np.float32)
    w = rng.random((n, k)).astype(np.float32)
    w /= w.sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(_weighted_quantiles(y, w, qs), _numpy_reference(y, w, qs))
    # Equal weights -> cdf has exact ties at quantile boundaries; first-cdf>=q scan direction must match argmax.
    w_eq = np.full((n, k), 1.0 / k, dtype=np.float32)
    np.testing.assert_array_equal(_weighted_quantiles(y, w_eq, qs), _numpy_reference(y, w_eq, qs))
