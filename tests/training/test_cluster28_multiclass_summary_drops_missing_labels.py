"""Audit 2026-11-20 finding 28-F31: the multiclass split summary leaves missing labels out of its denominator.

A NaN label used to count in n, so every class rate fell and a split with many unlabelled rows drew a false drift warning. The test gives a
split with one NaN label and checks the rates are over the labelled rows only, with the missing count reported.
"""

import numpy as np

from mlframe.training.drift_report import _multiclass_split_summary


def test_missing_labels_are_outside_the_rate_denominator():
    """Three labelled rows across three classes give rates of one third; the NaN row is counted as missing, not as a row."""
    out = _multiclass_split_summary(np.array([0.0, 1.0, np.nan, 2.0]), [0, 1, 2])
    assert out["n"] == 3
    assert out["n_missing"] == 1
    assert all(abs(rate - 1 / 3) < 1e-12 for rate in out["rates"].values())
