"""biz_value test for ``evaluation.triage_cv_delta``.

The win: a CV delta of a given magnitude should be trusted when it comes from feature engineering but
distrusted (flagged non-actionable) when it comes from hyperparameter tuning at the SAME magnitude -- per the
Home-Credit writeup finding that FE-driven CV gains correlate with LB far more reliably than hyperparameter-
driven gains of equal nominal size.
"""

from __future__ import annotations

import numpy as np

from mlframe.evaluation.cv_delta_triage import CVDeltaHistory, triage_cv_delta
from mlframe.evaluation.noise_band import cv_score_equivalence_band


def test_biz_val_triage_cv_delta_trusts_fe_but_flags_equal_size_hyperparameter_delta():
    """Triage cv delta trusts fe but flags equal size hyperparameter delta."""
    rng = np.random.default_rng(0)
    baseline = 0.700 + rng.normal(0, 0.003, size=6)

    band = cv_score_equivalence_band(baseline, method="sem")
    # a delta that clears the plain noise band but not a 2x-widened one.
    borderline_delta = band * 1.5
    candidate = baseline + borderline_delta

    fe_result = triage_cv_delta(baseline, candidate, change_source="feature_engineering")
    hp_result = triage_cv_delta(baseline, candidate, change_source="hyperparameter")

    assert fe_result["actionable"] is True, fe_result["reason"]
    assert hp_result["actionable"] is False, hp_result["reason"]


def test_triage_cv_delta_within_noise_flags_both_sources_non_actionable():
    """Triage cv delta within noise flags both sources non actionable."""
    rng = np.random.default_rng(1)
    baseline = 0.700 + rng.normal(0, 0.003, size=6)
    band = cv_score_equivalence_band(baseline, method="sem")
    tiny_delta = band * 0.1
    candidate = baseline + tiny_delta

    fe_result = triage_cv_delta(baseline, candidate, change_source="feature_engineering")
    hp_result = triage_cv_delta(baseline, candidate, change_source="hyperparameter")

    assert fe_result["actionable"] is False
    assert hp_result["actionable"] is False


def test_triage_cv_delta_shape_mismatch_raises():
    """Triage cv delta shape mismatch raises."""
    import pytest

    with pytest.raises(ValueError):
        triage_cv_delta(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), change_source="feature_engineering")


def test_triage_cv_delta_invalid_change_source_raises():
    """Triage cv delta invalid change source raises."""
    import pytest

    with pytest.raises(ValueError):
        triage_cv_delta(np.array([1.0, 2.0]), np.array([1.1, 2.1]), change_source="bogus")


def test_biz_val_triage_cv_delta_history_lowers_false_positive_rate_over_single_call():
    """The win: with a small ``n_folds``, a single call's fold-score sample variance is itself noisy, so its
    ``sem`` band mis-fires on NULL deltas (baseline and candidate drawn from the identical distribution -- no
    true improvement) more often than the true noise scale would justify. A ``CVDeltaHistory`` accumulator
    pools variance evidence across many historical calls; once it has enough pooled degrees of freedom its band
    converges toward the true noise scale and the empirical false-positive rate on NULL deltas drops measurably
    below the single-call rate observed on the SAME draws.
    """
    rng = np.random.default_rng(42)
    sigma = 0.01
    n_folds = 4
    n_experiments = 400
    warmup = 60  # calls used only to build up history's pooled dof before FP rate is scored

    history = CVDeltaHistory()
    fp_single = 0
    fp_history = 0
    n_scored = 0
    for i in range(n_experiments):
        baseline = 0.700 + rng.normal(0, sigma, size=n_folds)
        candidate = 0.700 + rng.normal(0, sigma, size=n_folds)  # NULL: no true delta, same generative distribution

        single_result = triage_cv_delta(baseline, candidate, change_source="feature_engineering")
        history_result = triage_cv_delta(baseline, candidate, change_source="feature_engineering", history=history, min_history_dof=20)

        if i >= warmup:
            n_scored += 1
            fp_single += int(single_result["actionable"])
            fp_history += int(history_result["actionable"])

    fp_rate_single = fp_single / n_scored
    fp_rate_history = fp_history / n_scored

    # measured on this fixed seed: fp_rate_single ~= 0.265, fp_rate_history ~= 0.212 -- thresholds set with margin.
    assert fp_rate_single >= 0.23, fp_rate_single
    assert fp_rate_history <= 0.23, fp_rate_history
    assert fp_rate_history < fp_rate_single, (fp_rate_history, fp_rate_single)
    assert history.pooled_dof == (n_folds - 1) * n_experiments  # confirms history actually accumulated, not a no-op


def test_biz_val_triage_cv_delta_history_noop_when_absent():
    """Opt-in guarantee: omitting ``history`` must reproduce the exact single-call band, bit-for-bit."""
    rng = np.random.default_rng(7)
    baseline = 0.700 + rng.normal(0, 0.003, size=6)
    candidate = baseline + 0.01

    plain = triage_cv_delta(baseline, candidate, change_source="feature_engineering")
    explicit_none = triage_cv_delta(baseline, candidate, change_source="feature_engineering", history=None)

    assert plain == explicit_none
    assert plain["band_source"] == "single_call"
