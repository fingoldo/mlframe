"""Degenerate inputs: identical single-class raters, empty multilabel input and empty ordinal input."""
import numpy as np
import pytest

from mlframe.metrics._multilabel_metrics import hamming_loss, jaccard_score_multilabel, subset_accuracy
from mlframe.metrics.classification._ordinal_cutpoints import optimal_ordinal_cutpoints
from mlframe.metrics.classification._weighted_kappa import weighted_kappa


def test_identical_single_class_raters_with_n_classes_two_agree_perfectly():
    """The expected-disagreement denominator is 0 here and the function returned 0.0; without n_classes the same input gave 1.0."""
    assert weighted_kappa(np.array([0, 0, 0]), np.array([0, 0, 0]), n_classes=2) == 1.0


def test_empty_multilabel_input_is_nan_not_a_division_error():
    """The kernels ended with total / N and N = 0."""
    empty = np.zeros((0, 3), dtype=np.int64)
    for fn in (hamming_loss, subset_accuracy, jaccard_score_multilabel):
        assert np.isnan(fn(empty, empty))


def test_empty_ordinal_input_raises_a_clear_error():
    """np.quantile on a zero-length array raised an opaque IndexError."""
    with pytest.raises(ValueError, match="empty"):
        optimal_ordinal_cutpoints(np.array([], dtype=int), np.array([]), n_classes=3)
