"""``_occupied_levels`` counts the distinct codes exactly like ``np.unique`` on every dtype and range it may meet."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._screen_predictors_gate import _occupied_levels


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.float32])
def test_matches_unique(dtype):
    """Same count as np.unique for dense, sparse, constant, empty and strided columns."""
    rng = np.random.default_rng(0)
    for col in (rng.integers(0, 10, 5000), rng.integers(0, 120, 5000) * 7 % 127, np.zeros(100, dtype=np.int64), np.array([], dtype=np.int64)):
        col = col.astype(dtype)
        assert _occupied_levels(col) == np.unique(col).shape[0]
    wide = rng.integers(0, 12, (3000, 5)).astype(dtype)
    assert _occupied_levels(wide[:, 2]) == np.unique(wide[:, 2]).shape[0]


def test_negative_and_huge_codes_take_the_unique_path():
    """Negative codes (a sentinel) and codes beyond the bincount bound still count correctly."""
    assert _occupied_levels(np.array([-1, 0, 0, 3, -1])) == 3
    assert _occupied_levels(np.array([0, 5_000_000, 5_000_000])) == 2
