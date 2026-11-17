"""Regression tests for NaN targets in MDLP binning and the many-class MDL threshold overflow."""
import numpy as np

from mlframe.feature_selection.filters.supervised_binning import _encode_mdlp_target_uncached, _drop_nonfinite_xy, _mdlp_pass_threshold_njit


def test_nan_rows_of_a_quantised_continuous_target_are_dropped_not_made_a_class():
    """A NaN in a high-cardinality float y must be dropped with its row, not searchsorted into the top pseudo-class."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    y[:7] = np.nan
    x = rng.normal(size=500)
    y_enc, _ = _encode_mdlp_target_uncached(y, 64, False, 16, 8)
    x2, y2 = _drop_nonfinite_xy(x, y_enc)
    assert x2.size == 493 and y2.size == 493


def test_missing_labels_of_an_object_target_are_dropped_not_minus_one_class():
    """None in an object y factorizes to -1; those rows must be dropped, not kept as class -1."""
    y = np.array(["a", "b", None, "a", "b", None], dtype=object)
    x = np.arange(6, dtype=float)
    y_enc, _ = _encode_mdlp_target_uncached(y, 64, False, 16, 8)
    x2, y2 = _drop_nonfinite_xy(x, y_enc)
    assert x2.size == 4 and (y2 >= 0).all()


def test_mdl_threshold_does_not_overflow_for_many_classes():
    """3**K overflows float64 for K > 650, which made every split fail; a clearly informative split must pass."""
    ok = _mdlp_pass_threshold_njit.py_func(6.9, 6.9, 100000, 1000, 500, 500, 0.0, 0.0)
    assert ok is True
