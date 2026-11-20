"""Cluster 39 regressions: zero smoothing has a finite first encoding, null categories are not NaN sums, and an object target with fractional values is refused (audit 2026-11-20, findings 39-F1, 39-F33, 39-F34)."""

import numpy as np
import pytest

from mlframe.training.extractors._extractors_dtype_helpers import _safe_int_cast_numpy
from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode


def test_zero_smoothing_first_occurrence_is_finite():
    """With smoothing=0 the first row of a category is 0/0; the global prior is the documented fallback (39-F34)."""
    cats = np.array(["a", "b", "a", "b"], dtype=object)
    y = np.array([1.0, 0.0, 0.0, 1.0])
    out = ordered_target_encode(cats, y, smoothing=0.0)
    assert np.isfinite(out).all()


def test_null_category_gets_a_finite_encoding():
    """A null category used to produce NaN running sums (39-F33)."""
    cats = np.array(["a", None, "a", None], dtype=object)
    y = np.array([1.0, 0.0, 0.0, 1.0])
    out = ordered_target_encode(cats, y, smoothing=1.0)
    assert np.isfinite(out).all()


def test_object_target_with_fractional_values_is_refused():
    """An object target of 0.5 used to be truncated to 0 by the int8 cast (39-F1)."""
    arr = np.array([0.5, 1.0], dtype=object)
    with pytest.raises(ValueError, match="non-integer"):
        _safe_int_cast_numpy(arr, "y")


def test_object_target_of_whole_numbers_still_casts():
    """Whole-number object targets keep working."""
    out = _safe_int_cast_numpy(np.array([0, 1, 1], dtype=object), "y")
    assert out.tolist() == [0, 1, 1]


def test_multilabel_scores_use_per_label_sigmoid_not_softmax():
    """A multilabel head is per-label logits: each column maps through its own sigmoid (audit 39-F53)."""
    from mlframe.training.neural._history_recorder import IterationMetricsRecorder

    rec = IterationMetricsRecorder(target_type="multilabel_classification", n_classes=2)
    logits = np.array([[0.0, 0.0], [10.0, -10.0]])
    out = rec._logits_to_score(logits)
    np.testing.assert_allclose(out, [[0.5, 0.5], [1.0, 0.0]], atol=1e-4)
