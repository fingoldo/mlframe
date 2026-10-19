"""Stages of ``audit_residuals``."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
from mlframe.training.targets.regression_residual_audit import _AUDIT_MOMENTS_PAR_THRESHOLD
from mlframe.training.targets.regression_residual_audit import _HAS_NUMBA_MOMENTS
from mlframe.training.targets.regression_residual_audit import _audit_moments_njit_par
from mlframe.training.targets.regression_residual_audit import _audit_moments_njit_seq
from mlframe.training.targets.regression_residual_audit import logger
# --- end imports ---


def _residual_moments(n, y_true, y_pred, warnings, residuals, excess_kurt, mean, pct_outliers_3sigma, skew, std):
    """Stage of ``audit_residuals``."""
    if _HAS_NUMBA_MOMENTS and n >= 2:
        try:
            if n >= _AUDIT_MOMENTS_PAR_THRESHOLD:
                mean, std, skew, excess_kurt, pct_outliers_3sigma = _audit_moments_njit_par(y_true, y_pred)
            else:
                mean, std, skew, excess_kurt, pct_outliers_3sigma = _audit_moments_njit_seq(y_true, y_pred)
            mean, std = float(mean), float(std)
            skew, excess_kurt = float(skew), float(excess_kurt)
            pct_outliers_3sigma = float(pct_outliers_3sigma)
        except Exception as _moments_err:
            logger.warning(
                "audit_residuals: numba fused-moments kernel failed (%s); " "falling back to numpy path. n=%d.",
                _moments_err,
                n,
            )
            warnings.append(f"numba fused-moments kernel failed ({_moments_err}); fell back to the numpy moments path.")
            mean = float(residuals.mean())
            std = float(residuals.std(ddof=1)) if n > 1 else 0.0
            if std > 0:
                z = (residuals - mean) / std
                z2 = z * z
                skew = float(np.mean(z * z2))
                excess_kurt = float(np.mean(z2 * z2) - 3.0)
                pct_outliers_3sigma = float((np.abs(z) > 3.0).mean())
            else:
                skew, excess_kurt, pct_outliers_3sigma = 0.0, 0.0, 0.0
    else:
        # numba-less environment: same numpy-only path as the pre-iter129 code.
        mean = float(residuals.mean())
        std = float(residuals.std(ddof=1)) if n > 1 else 0.0
        if std > 0:
            z = (residuals - mean) / std
            z2 = z * z
            skew = float(np.mean(z * z2))
            excess_kurt = float(np.mean(z2 * z2) - 3.0)
            pct_outliers_3sigma = float((np.abs(z) > 3.0).mean())
        else:
            skew, excess_kurt, pct_outliers_3sigma = 0.0, 0.0, 0.0
    return excess_kurt, mean, pct_outliers_3sigma, skew, std
