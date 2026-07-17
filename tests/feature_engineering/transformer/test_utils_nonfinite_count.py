"""Regression test for the fused non-finite cell counter in transformer/_utils.

Pins _count_nonfinite_cells to the exact numpy reference
``int(np.count_nonzero(~np.isfinite(X)))`` it replaced (NaN AND +/-Inf both counted),
across f32/f64, the serial/parallel dispatch boundary, and edge cases (all-NaN, all-Inf,
clean, mixed). A regression that broke +/-Inf detection or the dispatch would change the
count and fail here.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer._utils import (
    _NONFINITE_PAR_THRESHOLD,
    _count_nonfinite_cells,
    validate_numeric_input,
)


def _ref(X: np.ndarray) -> int:
    return int(np.count_nonzero(~np.isfinite(X)))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(1, 1), (10, 10), (50_000, 7), (2003, 5)])
def test_count_matches_numpy_reference_clean(dtype, shape):
    rng = np.random.default_rng(0)
    X = rng.standard_normal(shape).astype(dtype)
    assert _count_nonfinite_cells(X) == _ref(X) == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_count_matches_numpy_reference_mixed(dtype):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((500, 8)).astype(dtype)
    X[3, 4] = np.nan
    X[7, 1] = np.inf
    X[100, 2] = -np.inf
    X[200, 0] = np.nan
    assert _count_nonfinite_cells(X) == _ref(X) == 4


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_count_all_nan_and_all_inf(dtype):
    nan_arr = np.full((4, 5), np.nan, dtype=dtype)
    inf_arr = np.full((4, 5), np.inf, dtype=dtype)
    neg_inf_arr = np.full((4, 5), -np.inf, dtype=dtype)
    assert _count_nonfinite_cells(nan_arr) == _ref(nan_arr) == 20
    assert _count_nonfinite_cells(inf_arr) == _ref(inf_arr) == 20
    assert _count_nonfinite_cells(neg_inf_arr) == _ref(neg_inf_arr) == 20


def test_count_across_dispatch_boundary():
    # Just over the parallel threshold: exercises the prange path; result must match serial/numpy.
    n = (_NONFINITE_PAR_THRESHOLD // 2) + 10
    X = np.ones((n, 2), dtype=np.float32)
    X[5, 0] = np.nan
    X[6, 1] = -np.inf
    assert X.size >= _NONFINITE_PAR_THRESHOLD
    assert _count_nonfinite_cells(X) == _ref(X) == 2


def test_validate_numeric_input_still_rejects_nonfinite():
    X = np.ones((10, 3), dtype=np.float32)
    X[2, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_numeric_input(X, name="X")
    # +/-Inf path
    X2 = np.ones((10, 3), dtype=np.float32)
    X2[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_numeric_input(X2, name="X")
    # clean float passes
    validate_numeric_input(np.ones((10, 3), dtype=np.float64), name="X")
