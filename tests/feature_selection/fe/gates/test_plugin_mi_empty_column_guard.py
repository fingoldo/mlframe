"""Regression: the njit plug-in-MI kernels must return 0.0 (not crash) on an empty column.

Found by the edge-case critique agent (C1, 2026-06-22): a fully-filtered subsample / empty finite mask can
hand a 0-row column to the batch/scalar MI kernels, whose ``y_min = y[0]`` (or ``log(n)``) hit a numba
native access violation. The cuda twins already guarded n==0; the njit ones did not.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.hermite_fe import (
    _plugin_mi_classif_njit,
    _plugin_mi_classif_batch_njit,
    _plugin_mi_regression_njit,
    _plugin_mi_from_binned_njit,
)


def test_scalar_classif_mi_empty():
    """Scalar classif mi empty."""
    assert _plugin_mi_classif_njit(np.empty(0), np.empty(0, dtype=np.int64), 20) == 0.0


def test_regression_mi_empty():
    """Regression mi empty."""
    assert _plugin_mi_regression_njit(np.empty(0), np.empty(0), 20) == 0.0


def test_from_binned_mi_empty():
    """From binned mi empty."""
    assert _plugin_mi_from_binned_njit(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 20) == 0.0


def test_batch_classif_mi_empty_columns():
    # (0, k) matrix: every column is empty -> all-zero MI vector, no crash.
    """Batch classif mi empty columns."""
    out = _plugin_mi_classif_batch_njit(np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int64), 20)
    assert out.shape == (3,)
    assert np.all(out == 0.0)
