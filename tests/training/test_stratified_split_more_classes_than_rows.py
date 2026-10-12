"""A stratified split whose target has more classes than one side has rows falls back to a random split instead of failing.

sklearn needs at least one row per class on each side of the cut. A 36-class target carved into a 24-row test side made the
splitter raise and failed whole fuzz combos; the singleton-class case already degrades the same way.
"""

from __future__ import annotations

import numpy as np

from mlframe.training._split_helpers import _stratified_split


def test_more_classes_than_test_rows_falls_back_to_random_split():
    """36 classes of 2 rows each, carved to a 24-row test side, still split into disjoint train and test index sets."""
    stratify_y = np.repeat(np.arange(36), 2)
    indices = np.arange(stratify_y.size)

    left, right = _stratified_split(indices, test_size=1 / 3, stratify_y=stratify_y, random_state=0)

    assert right.size == 24
    assert left.size == 48
    assert np.intersect1d(left, right).size == 0
