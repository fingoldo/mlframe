"""Target binning for the MI-floor operators: NaN targets and fractional class labels keep distinct, valid classes."""

import numpy as np

from mlframe.feature_selection.filters._fe_accuracy_gate import bin_y_for_class_mi


def test_continuous_target_with_nan_rows_gets_a_class_after_the_last_bin_not_int64_min():
    """qcut leaves NaN rows unbinned; the int64 cast used to turn them into -2**63."""
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.random(200), [np.nan, np.nan]])
    out = bin_y_for_class_mi(y, nbins=10)
    assert out.min() == 0
    assert out[-1] == out[-2] == out[:-2].max() + 1


def test_fractional_class_labels_are_not_merged_by_the_integer_cast():
    """Two classes 0.2 and 0.7 stay two classes (the cast sent both to 0)."""
    y = np.array([0.2, 0.7] * 20)
    out = bin_y_for_class_mi(y)
    assert sorted(set(out.tolist())) == [0, 1]
    np.testing.assert_array_equal(out, np.array([0, 1] * 20))


def test_integer_valued_float_labels_keep_the_plain_cast():
    """Integer-valued float labels stay bit-identical to the direct cast (including a negative class)."""
    y = np.array([-1.0, 1.0, 3.0] * 10)
    np.testing.assert_array_equal(bin_y_for_class_mi(y), y.astype(np.int64))
