"""``bur_term`` refuses columns of different lengths instead of letting the njit kernel read past the shorter one."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._bur_term import bur_term


def test_bur_term_rejects_selected_column_of_different_length():
    """A shorter selected column would be read out of bounds by the compiled kernel."""
    x = np.zeros(10, dtype=np.int64)
    y = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError, match="same number of rows"):
        bur_term(x, [np.zeros(4, dtype=np.int64)], y, 2, [2], 2)


def test_bur_term_rejects_target_of_different_length():
    """A target shorter than the candidate is rejected too."""
    with pytest.raises(ValueError, match="same number of rows"):
        bur_term(np.zeros(10, dtype=np.int64), [], np.zeros(5, dtype=np.int64), 2, [], 2)
