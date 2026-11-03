"""The fused entropy kernels refuse out-of-range class codes instead of writing past their frequency array."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory._entropy_kernels import _entropy_x_onto_classes, _merge_vars_freqs


def test_last_variable_code_equal_to_nbins_raises():
    """Code 3 in a column declared with 3 bins used to index freqs[3] on a 3-slot array (memory corruption, or a silently wrong H)."""
    data = np.array([[0, 0, 0], [1, 1, 3], [2, 2, 1], [0, 1, 2]], dtype=np.int32)
    with pytest.raises(ValueError):
        _merge_vars_freqs(data, np.array([0, 1, 2]), np.array([3, 3, 3]))


def test_negative_sentinel_in_the_candidate_column_raises():
    """A -1 null sentinel in the melted column must raise like merge_vars does."""
    data = np.array([[0, -1], [1, 0], [0, 1], [1, 1]], dtype=np.int32)
    with pytest.raises(ValueError):
        _entropy_x_onto_classes(data, 1, np.array([0, 1, 0, 1], dtype=np.int32), 2, 2)


def test_valid_codes_still_work():
    """In-range input is unchanged (negative control)."""
    data = np.array([[0, 0, 0], [1, 1, 2], [2, 2, 1], [0, 1, 2]], dtype=np.int32)
    assert _merge_vars_freqs(data, np.array([0, 1, 2]), np.array([3, 3, 3])).sum() > 0
