"""The stratified resampler keeps one group per class for string and fractional labels."""

from __future__ import annotations

import numpy as np

from mlframe.calibration.policy import _build_resample_indices


def test_string_labels_do_not_crash_and_resample_within_each_class():
    """Two string classes give a resample whose class counts equal the original ones."""
    y = np.array(["neg"] * 30 + ["pos"] * 10)
    idx = _build_resample_indices(40, 5, y, 0)
    assert idx.shape == (5, 40)
    for row in idx:
        assert (y[row] == "pos").sum() == 10


def test_float_labels_keep_both_classes():
    """Labels 0.2 and 0.7 used to collapse into one integer key and lose a class."""
    y = np.array([0.2] * 25 + [0.7] * 15)
    idx = _build_resample_indices(40, 4, y, 0)
    assert len(idx) == 4
    for row in idx:
        assert (y[row] == 0.7).sum() == 15
