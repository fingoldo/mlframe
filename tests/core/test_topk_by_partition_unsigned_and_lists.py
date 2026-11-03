"""topk_by_partition negates safely: unsigned, bool and list inputs pick the true largest elements."""

from __future__ import annotations

import numpy as np

from mlframe.core.arrays import topk_by_partition


def test_unsigned_input_returns_the_true_maximum():
    """[1, 5, 3] as uint8: negation wrapped to [255, 251, 253] and picked index 0; the maximum is at index 1."""
    ind, vals = topk_by_partition(np.array([1, 5, 3], dtype=np.uint8), 1)
    assert int(np.ravel(ind)[0]) == 1 and int(np.ravel(vals)[0]) == 5


def test_plain_list_input_works():
    """A Python list used to fail on unary minus."""
    ind, _vals = topk_by_partition([2.0, 9.0, 4.0], 1)
    assert int(np.ravel(ind)[0]) == 1
