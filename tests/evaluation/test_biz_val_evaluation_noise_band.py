"""biz_value + unit tests for ``evaluation.noise_band``.

The win: gating a greedy candidate-acceptance loop on ``cv_score_equivalence_band`` should reject far fewer
statistically-tied-but-numerically-different candidates as "improvements" than a naive any-positive-delta
acceptance rule, when the two candidates are drawn from the SAME underlying score distribution (true tie).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.evaluation.noise_band import cv_score_equivalence_band, is_within_noise_band


def test_cv_score_equivalence_band_zero_for_single_score():
    assert cv_score_equivalence_band(np.array([0.87])) == 0.0


def test_cv_score_equivalence_band_positive_for_noisy_folds():
    rng = np.random.default_rng(0)
    folds = 0.85 + 0.01 * rng.standard_normal(5)
    band = cv_score_equivalence_band(folds)
    assert band > 0.0 and np.isfinite(band)


def test_cv_score_equivalence_band_std_wider_than_sem():
    rng = np.random.default_rng(1)
    folds = 0.80 + 0.02 * rng.standard_normal(8)
    sem_band = cv_score_equivalence_band(folds, method="sem")
    std_band = cv_score_equivalence_band(folds, method="std")
    assert std_band > sem_band


def test_cv_score_equivalence_band_invalid_method_raises():
    with pytest.raises(ValueError):
        cv_score_equivalence_band(np.array([0.1, 0.2, 0.3]), method="bogus")


def test_is_within_noise_band_true_for_tiny_delta():
    folds = np.array([0.900, 0.905, 0.895, 0.902, 0.898])
    assert is_within_noise_band(0.9000, 0.9005, folds) is True


def test_is_within_noise_band_false_for_large_delta():
    folds = np.array([0.900, 0.901, 0.899, 0.900, 0.900])
    assert is_within_noise_band(0.90, 0.98, folds) is False


def test_biz_val_noise_band_reduces_false_accepts_vs_naive_greedy_selection():
    """Simulate a greedy selection loop comparing a "candidate" against a "current best" drawn from the
    SAME score distribution (a true tie: neither is actually better). A naive rule that accepts any positive
    delta over-accepts noise as signal; gating on the noise band should cut the false-accept rate sharply.

    Ground truth: both candidate and best are 5-fold CV scores with mean 0.850 and per-fold std 0.01 (a
    realistic small-n CV noise level). Repeated over many independent trials, "candidate beats best" should
    happen ~50% of the time by chance under a naive rule (it's a coin flip), while the noise-band-gated rule
    should call it a tie almost every time.
    """
    rng = np.random.default_rng(42)
    n_trials = 400
    n_folds = 5
    fold_std = 0.01

    naive_accepts = 0
    band_accepts = 0
    for _ in range(n_trials):
        best_folds = 0.850 + fold_std * rng.standard_normal(n_folds)
        cand_folds = 0.850 + fold_std * rng.standard_normal(n_folds)
        best_mean = float(best_folds.mean())
        cand_mean = float(cand_folds.mean())

        if cand_mean > best_mean:
            naive_accepts += 1

        if not is_within_noise_band(cand_mean, best_mean, best_folds) and cand_mean > best_mean:
            band_accepts += 1

    naive_rate = naive_accepts / n_trials
    band_rate = band_accepts / n_trials

    # Naive rule is a coin flip on a true tie (~50%); the noise-band rule should suppress
    # the overwhelming majority of those false accepts. Floor set well below the measured
    # value (~0.02-0.05) to absorb seed variance while still catching a regression to ~naive.
    assert naive_rate > 0.35, f"sanity: naive accept rate should be near 50% on a true tie, got {naive_rate}"
    assert band_rate < 0.15, f"noise-band accept rate should be far below naive on a true tie, got {band_rate}"
    assert band_rate < naive_rate / 2.0
