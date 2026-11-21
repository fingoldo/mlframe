"""MaxError is NaN when a residual is NaN, the same as MAE (audit 2026-11-20, finding 17-F1)."""
import numpy as np

from mlframe.metrics.regression import _regression_metrics as rm


def test_max_error_and_mae_agree_on_nan_residuals():
    """With one NaN prediction MAE is NaN; the max error must be NaN too, not the largest finite residual."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, np.nan, 5.0])
    assert np.isnan(rm.fast_mean_absolute_error(y_true, y_pred))
    assert np.isnan(rm.fast_max_error(y_true, y_pred))


def test_max_error_is_unchanged_on_finite_residuals():
    """Finite residuals keep the maximum."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.0, 1.0])
    assert abs(rm.fast_max_error(y_true, y_pred) - 2.0) < 1e-9
