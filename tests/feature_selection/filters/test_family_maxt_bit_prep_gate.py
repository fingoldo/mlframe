"""The bit-set gate of ``_family_maxt_gate`` declines a family with no columns, and the dispatch above it stays consistent for one."""

import numpy as np

import mlframe.feature_selection.filters._family_maxt_gate as gate


def test_family_bit_prep_declines_a_family_with_no_columns():
    """With no columns there is nothing to pack, so neither the host bit sets nor a device copy is built."""
    cols = np.zeros((0, 12), dtype=np.int64)
    n_x = np.zeros(0, dtype=np.int64)
    y = np.zeros(12, dtype=np.int64)
    assert gate._family_bit_prep(cols, n_x, y, 2, False, None) == (None, None)
