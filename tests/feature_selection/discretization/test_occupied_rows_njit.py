"""Regression: the occupied-row count in the honest-null fast path must be a standalone njit helper.

The inline nested loop with a branch hit ``AssertionError: unexpected cycle in lookup()`` in numba 0.68.1.
"""

import numpy as np

from mlframe.feature_selection.filters._mdlp_honest_bycol_njit import _occupied_rows


def test_occupied_rows_counts_only_rows_with_mass():
    """Only x-rows holding at least one count are occupied."""
    joint = np.zeros((5, 3), dtype=np.int64)
    joint[0, 1] = 4
    joint[2, 0] = 1
    joint[4, 2] = 7
    assert _occupied_rows(joint) == 3


def test_occupied_rows_all_empty_is_zero():
    """A joint with no counts has no occupied rows."""
    assert _occupied_rows(np.zeros((4, 2), dtype=np.int64)) == 0
