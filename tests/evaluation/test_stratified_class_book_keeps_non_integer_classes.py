"""The stratified resampling book keeps every class for fractional and string labels."""

from __future__ import annotations

import numpy as np

from mlframe.evaluation._bootstrap_single_helpers import _stratified_class_book


def test_fractional_labels_keep_both_classes():
    """Labels 0.2 and 0.7 used to share the key int(c) == 0 and lose a class: total_n came out smaller than n."""
    y = np.array([0.2] * 30 + [0.7] * 10)
    groups, sizes, _offsets, total = _stratified_class_book(y, 40, "bootstrap_metric")
    assert len(groups) == 2 and total == 40 and sorted(sizes.tolist()) == [10, 30]


def test_string_labels_do_not_raise():
    """String classes raised ValueError from int()."""
    y = np.array(["a"] * 5 + ["b"] * 5)
    assert _stratified_class_book(y, 10, "bootstrap_metric")[3] == 10
