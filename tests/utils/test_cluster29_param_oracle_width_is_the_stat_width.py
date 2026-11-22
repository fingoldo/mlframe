"""Audit 2026-11-20 finding 29-F26: the fingerprint width matches the matrix the statistics are computed over, for a 3-D input.

A (n, 4, 5) array was fingerprinted with p=4 but its statistics ran over 20 flattened columns. The width is now the reshaped width.
"""

import numpy as np

from mlframe.utils import _param_oracle as po


def test_three_d_input_width_is_the_flattened_width():
    """The reported width for a (n, 4, 5) array is 20, the number of columns the matrix has."""
    arr = np.zeros((10, 4, 5))
    arr2d, _n, p, _kind = po._as_2d_numeric(arr)
    assert arr2d.shape[1] == 20
    assert p == 20
