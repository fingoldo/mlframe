"""Grouped causal target bases (expanding / trailing mean of strictly earlier rows) skip rows without a label."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery._grouped_causal_bases import _grouped_expanding_kernel, _grouped_trailing_kernel

Y = np.array([1.0, np.nan, 3.0, np.nan, 5.0, 7.0, 2.0, np.nan, 4.0])
OFFSETS = np.array([0, 6, 9], dtype=np.int64)  # two groups: rows 0..5 and 6..8


def test_the_expanding_mean_averages_the_earlier_labelled_rows_only():
    """Summing a NaN made every later row of its group NaN, so the base lost all coverage after the first gap."""
    out = _grouped_expanding_kernel(Y, OFFSETS, False)
    np.testing.assert_allclose(out, [np.nan, 1.0, 1.0, 2.0, 2.0, 3.0, np.nan, 2.0, 2.0])


def test_the_trailing_mean_takes_its_window_from_the_labelled_rows():
    out = _grouped_trailing_kernel(Y, OFFSETS, 2, False)
    np.testing.assert_allclose(out, [np.nan, 1.0, 1.0, 2.0, 2.0, 4.0, np.nan, 2.0, 2.0])


@pytest.mark.parametrize("kernel", ["expanding", "trailing"])
def test_a_filled_first_row_uses_the_first_label_of_its_group(kernel):
    y = np.array([np.nan, 4.0, 6.0])
    offsets = np.array([0, 3], dtype=np.int64)
    out = _grouped_expanding_kernel(y, offsets, True) if kernel == "expanding" else _grouped_trailing_kernel(y, offsets, 5, True)
    np.testing.assert_allclose(out, [4.0, 4.0, 4.0])  # row 2 sees only row 1 (4.0); rows 0 and 1 have no earlier label
