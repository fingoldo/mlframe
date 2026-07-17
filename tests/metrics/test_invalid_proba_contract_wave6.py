"""Regression (wave6 P2): invalid-probability handling must be consistent across the standalone metrics, the
brier_and_precision model-selection scorer, and the fused report block.

- brier_and_precision_score: fast_brier_score_loss returns NaN on out-of-[0,1] probs; `NaN > threshold` is False,
  so the invalid model slipped through the gate and the scorer returned NaN (or precision - NaN), silently
  changing which model an sklearn search selects. It must return 0.0 (hard fail) instead.
- fast_binary_probability_metrics_block: previously CLAMPED out-of-range scores to [1e-15, 1-1e-15] and reported a
  plausible Brier/log_loss, diverging from the standalone kernels that return NaN. It must now return an all-NaN
  block on invalid scores.
"""

import numpy as np

from mlframe.metrics._core_auc_brier import brier_and_precision_score
from mlframe.metrics.classification._classification_extras_blocks import fast_binary_probability_metrics_block


def test_brier_precision_scorer_fails_closed_on_out_of_range_proba():
    yt = np.array([0, 1, 0, 1])
    yp_bad = np.array([0.3, 1.4, 0.2, -0.1])  # out of [0,1] -> Brier NaN
    score = brier_and_precision_score(yt, yp_bad)
    assert score == 0.0, "invalid probabilities must fail the gate (0.0), not return NaN"
    assert np.isfinite(score)


def test_brier_precision_scorer_fails_closed_on_nan_proba():
    yt = np.array([0, 1, 1, 0])
    yp = np.array([0.2, np.nan, 0.9, 0.1])
    score = brier_and_precision_score(yt, yp)
    assert score == 0.0 and np.isfinite(score)


def test_brier_precision_scorer_still_rewards_good_model():
    yt = np.array([0, 0, 1, 1, 1])
    yp = np.array([0.05, 0.1, 0.9, 0.95, 0.99])
    score = brier_and_precision_score(yt, yp)
    assert np.isfinite(score) and score > 0.0


def test_probability_block_returns_nan_on_out_of_range():
    yt = np.array([0, 1, 0, 1])
    yp = np.array([0.3, 1.5, 0.2, 0.8])
    block = fast_binary_probability_metrics_block(yt, yp)
    assert np.isnan(block["Brier"]), "out-of-range scores must yield NaN Brier (match standalone), not a clamped value"
    assert np.isnan(block["log_loss"]) and np.isnan(block["BSS"])


def test_probability_block_returns_nan_on_nan_scores():
    yt = np.array([0, 1, 1])
    yp = np.array([0.2, np.nan, 0.8])
    block = fast_binary_probability_metrics_block(yt, yp)
    assert np.isnan(block["Brier"]) and np.isnan(block["log_loss"])


def test_probability_block_valid_scores_unchanged():
    yt = np.array([0, 0, 1, 1])
    yp = np.array([0.1, 0.2, 0.8, 0.9])
    block = fast_binary_probability_metrics_block(yt, yp)
    # in-range (incl. exact 0/1) still produces finite metrics
    assert np.isfinite(block["Brier"]) and 0.0 <= block["Brier"] <= 1.0
    block01 = fast_binary_probability_metrics_block(np.array([0, 1]), np.array([0.0, 1.0]))
    assert np.isfinite(block01["Brier"]), "exact 0.0/1.0 are in-range and must stay finite"
