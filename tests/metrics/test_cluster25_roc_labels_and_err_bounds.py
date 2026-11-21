"""Audit 2026-11-20 findings 25-F33 and 25-F46 in the metrics.

25-F33: fast_roc_curve picks the positive class the same way fast_roc_auc does (the larger of two labels), so {1, 2} labels
give the curve sklearn gives, and a NaN score is refused.
25-F46: expected_reciprocal_rank rejects a negative grade, which would push the remain probability out of [0, 1].
"""

import numpy as np
import pytest
from sklearn.metrics import roc_curve

from mlframe.metrics._core_auc_brier import fast_roc_curve
from mlframe.metrics._ranking_extras import expected_reciprocal_rank


def test_roc_curve_takes_the_larger_label_as_positive_like_sklearn():
    """Labels {1, 2}: the curve's TPR and FPR match sklearn's, with 2 as the positive class."""
    y = np.array([1, 2, 2, 1, 2, 1, 2, 2])
    s = np.array([0.1, 0.8, 0.6, 0.3, 0.9, 0.2, 0.7, 0.5])
    fpr, tpr, _ = fast_roc_curve(y, s)
    sk_fpr, sk_tpr, _ = roc_curve(y, s, pos_label=2)
    np.testing.assert_allclose(tpr, sk_tpr)
    np.testing.assert_allclose(fpr, sk_fpr)


def test_roc_curve_refuses_a_nan_score():
    """A NaN score is refused, as sklearn refuses it, rather than sorted to the top."""
    with pytest.raises(ValueError, match="NaN"):
        fast_roc_curve(np.array([0, 1, 1, 0]), np.array([0.1, np.nan, 0.6, 0.2]))


def test_err_refuses_a_negative_grade():
    """A label of -4 is refused: a negative grade would make the stop probability, and the remain product, leave [0, 1]."""
    with pytest.raises(ValueError, match="below 0"):
        expected_reciprocal_rank(np.array([-4.0, 1.0]), np.array([0.9, 0.2]), np.array([0, 0]))
