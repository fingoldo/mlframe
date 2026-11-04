"""AUC entry points treat the larger of two labels as positive, like sklearn, whatever the label values."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from mlframe.metrics import fast_roc_auc


@pytest.mark.parametrize("labels", [(1, 2), (0, 2), (-1, 1), (0, 1)])
def test_fast_roc_auc_matches_sklearn_for_any_two_labels(labels):
    """Labels {1, 2} gave 1 - AUC and {0, 2} gave NaN."""
    lo, hi = labels
    y = np.array([lo, lo, hi, hi, lo, hi])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.6])
    assert abs(float(fast_roc_auc(y, p)) - float(roc_auc_score(y, p))) < 1e-9
