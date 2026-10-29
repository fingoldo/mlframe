"""The external-validation materialise kernel writes row-major; its output must equal the numpy op for every (external factor, op) column."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_materialise import _materialise_extval_njit


def _reference(a: np.ndarray, b: np.ndarray, op: int) -> np.ndarray:
    """Numpy form of registry op ``op`` on float64 operands."""
    a = a.astype(np.float64)
    if op == 0:
        return a * b
    if op == 1:
        return a + b
    if op == 2:
        return a - b
    if op == 3:
        return a / np.where(b != 0.0, b, 1e-9)
    if op == 4:
        return np.maximum(a, b)
    if op == 5:
        return np.minimum(a, b)
    if op == 6:
        return np.abs(a - b)
    if op == 7:
        return np.sign(a) * np.abs(b) + np.where(np.isnan(a) | np.isnan(b), np.nan, 0.0)
    return a / (np.abs(b) + 1.0)


def test_extval_kernel_matches_numpy_ops_with_nan_and_zero():
    """Every column e * n_ops + o equals op o of the float32 operand against external column e, NaN and exact zeros included."""
    rng = np.random.default_rng(3)
    n, n_ext = 500, 5
    a = rng.standard_normal(n).astype(np.float32)
    a[::17] = 0.0
    a[::31] = np.nan
    b = rng.standard_normal((n, n_ext))
    b[::7, 0] = 0.0
    b[::11, 2] = np.nan
    ops = np.arange(9, dtype=np.int64)
    out = np.empty((n, n_ext * ops.size))
    _materialise_extval_njit(a, b, ops, out)
    for e in range(n_ext):
        for o in range(ops.size):
            np.testing.assert_array_equal(out[:, e * ops.size + o], _reference(a, b[:, e], o), err_msg=f"ext={e} op={o}")
