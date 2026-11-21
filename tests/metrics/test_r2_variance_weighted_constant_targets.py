"""variance_weighted R2 on constant targets follows sklearn: a perfect fit scores 1.0 (audit 2026-11-20, finding 17-F2)."""
import numpy as np
from sklearn.metrics import r2_score

from mlframe.metrics.regression._regression_metrics import fast_r2_score


def test_perfect_fit_on_constant_outputs_is_one():
    """All outputs constant and predicted exactly: sklearn gives 1.0, the earlier code gave 0.0."""
    y = np.array([[2.0, 5.0], [2.0, 5.0], [2.0, 5.0]])
    assert abs(fast_r2_score(y, y.copy(), multioutput="variance_weighted") - r2_score(y, y.copy(), multioutput="variance_weighted")) < 1e-9
    assert abs(fast_r2_score(y, y.copy(), multioutput="variance_weighted") - 1.0) < 1e-9


def test_imperfect_fit_on_constant_outputs_is_zero():
    """Constant targets with a wrong prediction: sklearn gives 0.0 for a nonzero residual."""
    y = np.array([[2.0, 5.0], [2.0, 5.0], [2.0, 5.0]])
    p = np.array([[3.0, 5.0], [3.0, 5.0], [3.0, 5.0]])
    assert abs(fast_r2_score(y, p, multioutput="variance_weighted") - r2_score(y, p, multioutput="variance_weighted")) < 1e-9
