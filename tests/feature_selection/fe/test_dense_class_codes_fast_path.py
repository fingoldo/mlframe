"""The integer fast path of ``dense_class_codes`` equals the ``np.unique`` densification on every shape of label vector."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_fe_step._step_class_codes import dense_class_codes


def _reference(y):
    """The original implementation."""
    _, dense = np.unique(np.asarray(y).ravel(), return_inverse=True)
    return dense.astype(np.int64).ravel()


@pytest.mark.parametrize(
    "y",
    [
        np.array([3, 3, 7, 1, 7, 1, 1]),
        np.array([-5, 0, 5, -5, 5]),
        np.array([10**9, 10**9 + 3, 10**9 + 1]),
        np.array([2**40, 5, 5]),  # span too wide for the counting path
        np.array([True, False, True]),
        np.arange(20, dtype=np.int8).reshape(-1, 1),
        np.array([0.5, 1.5, 0.5]),  # fractional labels stay distinct via the np.unique path
        np.random.default_rng(0).integers(0, 20, 100_000),
        np.array([], dtype=np.int64),
    ],
)
def test_matches_the_unique_densification(y):
    """Same codes, same dtype, 1-D."""
    got = dense_class_codes(y)
    np.testing.assert_array_equal(got, _reference(y))
    assert got.dtype == np.int64 and got.ndim == 1
