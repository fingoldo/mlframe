"""Cluster 41 regression: strong skew is not masked by a mild-kurtosis verdict (audit 2026-11-20, finding 41-F4)."""

from mlframe.training.targets.regression_residual_audit import _diagnose


def test_mild_kurtosis_with_strong_skew_is_reported_as_skewed():
    """Skew +1.5 with excess kurtosis +1.0 must not be labelled Near-Gaussian (41-F4)."""
    hypothesis, _loss, _rationale = _diagnose(
        skew=1.5,
        excess_kurt=1.0,
        hetero_spearman=0.0,
        pct_outliers_3sigma=0.01,
        y_true_all_nonneg=False,
        y_pred_all_nonneg=False,
    )
    assert "Skewed" in hypothesis, hypothesis


def test_mild_kurtosis_without_skew_still_near_gaussian():
    """Control: with symmetric residuals the mild-peak verdict is unchanged."""
    hypothesis, _loss, _rationale = _diagnose(
        skew=0.0,
        excess_kurt=1.0,
        hetero_spearman=0.0,
        pct_outliers_3sigma=0.01,
        y_true_all_nonneg=False,
        y_pred_all_nonneg=False,
    )
    assert hypothesis == "Near-Gaussian (mildly peaky)", hypothesis
