"""biz_value test for ``calibration.smoothed_override_backtest.backtest_override``.

The win: an override source can be genuinely good at high confidence (a tight, near-exact match) but
noisy/wrong at low confidence (a loose fuzzy match) -- a caller who blends it in everywhere loses money
in the low-confidence tail even though the source looks great on average. ``backtest_override`` must
recover a safe confidence threshold from history such that blending ONLY at/above that threshold beats
both blending everywhere and the model alone.
"""

from __future__ import annotations

import numpy as np

from mlframe.calibration.smoothed_override import apply_smoothed_override
from mlframe.calibration.smoothed_override_backtest import backtest_override


def _make_confidence_gated_scenario(n: int, seed: int):
    """Builds seeded synthetic test data; returns ``(y_true, model_pred, override_pred, confidence)``."""
    rng = np.random.default_rng(seed)
    y_true = rng.uniform(0, 1, n)
    model_pred = y_true + rng.normal(scale=0.15, size=n)

    confidence = rng.random(n)
    # override is near-exact at high confidence, badly noisy at low confidence -- a realistic
    # fuzzy-match-degrades-gracefully shape.
    is_high_conf = confidence >= 0.5
    override_pred = np.where(is_high_conf, y_true + rng.normal(scale=0.01, size=n), y_true + rng.normal(scale=0.8, size=n))

    return y_true, model_pred, override_pred, confidence


def test_biz_val_backtest_override_recovers_safe_confidence_threshold():
    """Backtest override recovers safe confidence threshold."""
    y_true, model_pred, override_pred, confidence = _make_confidence_gated_scenario(n=6000, seed=0)

    result = backtest_override(y_true, model_pred, override_pred, confidence, a=0.9, n_buckets=10)

    # the source degrades exactly at confidence=0.5 by construction -- the recovered threshold must
    # land close to it, not at the extremes (0.0 = "trust everything", 1.0 = "trust nothing").
    assert 0.35 <= result.safe_threshold <= 0.65, f"expected safe_threshold near 0.5, got {result.safe_threshold:.4f}"

    # thresholding on the recovered value must beat both blending everywhere and the model alone.
    assert (
        result.mae_blend_safe < result.mae_blend_all
    ), f"expected thresholded blend to beat blend-everywhere, got safe={result.mae_blend_safe:.4f} all={result.mae_blend_all:.4f}"
    assert (
        result.mae_blend_safe < result.mae_model_overall
    ), f"expected thresholded blend to beat model alone, got safe={result.mae_blend_safe:.4f} model={result.mae_model_overall:.4f}"

    # applying the recovered threshold through the real production API must reproduce mae_blend_safe.
    override_mask = confidence >= result.safe_threshold
    production_blend = apply_smoothed_override(model_pred, override_pred, override_mask, a=0.9)
    mae_production = float(np.mean(np.abs(y_true - production_blend)))
    assert abs(mae_production - result.mae_blend_safe) < 1e-9


def test_biz_val_backtest_override_flags_uniformly_bad_source_as_unsafe():
    """Backtest override flags uniformly bad source as unsafe."""
    rng = np.random.default_rng(1)
    n = 3000
    y_true = rng.uniform(0, 1, n)
    model_pred = y_true + rng.normal(scale=0.1, size=n)
    confidence = rng.random(n)
    # a source that's noisy at EVERY confidence level (confidence score is uninformative junk).
    override_pred = y_true + rng.normal(scale=1.0, size=n)

    result = backtest_override(y_true, model_pred, override_pred, confidence, a=0.9, n_buckets=5)

    assert (
        result.safe_threshold >= 0.99
    ), f"expected an uninformative override source to be flagged unsafe (threshold near 1.0), got {result.safe_threshold:.4f}"
    assert result.mae_blend_safe <= result.mae_model_overall + 1e-9


def test_backtest_override_default_a_matches_apply_smoothed_override_bit_identical():
    """Regression: introducing backtest_override must not change apply_smoothed_override's own output."""
    prediction = np.array([1.0, 2.0, 3.0, 4.0])
    known_label = np.array([10.0, 20.0, 30.0, 40.0])
    mask = np.array([True, False, True, False])

    out = apply_smoothed_override(prediction, known_label, mask, a=0.9)
    np.testing.assert_array_equal(out, np.array([0.9 * 10.0 + 0.1 * 1.0, 2.0, 0.9 * 30.0 + 0.1 * 3.0, 4.0]))
