"""Stages of ``audit_residuals``."""

from __future__ import annotations

# --- imports (managed) ---
# --- end imports ---


def _residual_flags(std, mean, warnings, y_true_all_nonneg, y_pred):
    """Stage of ``audit_residuals``."""
    if std > 0 and abs(mean) > 0.5 * std:
        warnings.append(
            f"residual mean is large ({mean:+.4g}) relative to std ({std:.4g}); "
            "the model is biased - consider re-centering predictions or "
            "checking for a regression-target offset bug."
        )
    if y_true_all_nonneg and y_pred.min() < 0:
        warnings.append(
            "y_true is non-negative but y_pred has negative values - model is "
            "violating a domain constraint. Add a non-negativity constraint "
            "(e.g. log-link, exp output, or clip)."
        )
