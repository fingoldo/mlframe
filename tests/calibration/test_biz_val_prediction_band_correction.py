"""biz_value test for ``calibration.prediction_band_correction``.

Source: 4th_home-credit-default-risk.md -- "if you correct your prediction for revolving loan that is over
0.4 by 0.8, it will boost your auc" -- a targeted multiplicative correction for predictions above a threshold,
driven by a discovered subpopulation miscalibration. This tests the CALIBRATION improvement (Brier score,
the metric that actually captures whether corrected probabilities match observed rates) on a held-out test
split, fitting the correction factor only from a separate OOF split (no leakage).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss

from mlframe.calibration.prediction_band_correction import apply_prediction_band_correction, find_prediction_band_shift


def _make_biased_band_predictions(n: int, seed: int):
    """Builds seeded synthetic test data; returns ``(y_true, y_pred_biased)``."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n).astype(float)
    y_pred = np.clip(y_true * 0.6 + rng.normal(scale=0.25, size=n) + 0.2, 0, 1)
    # inflate predictions above 0.4 systematically (a discovered subpopulation over-prediction).
    mask_high = y_pred > 0.4
    y_pred_biased = y_pred.copy()
    y_pred_biased[mask_high] = np.clip(y_pred_biased[mask_high] * 1.5, 0, 1)
    return y_true, y_pred_biased


def test_biz_val_band_correction_improves_calibration_on_held_out_split():
    """Band correction improves calibration on held out split."""
    y_true, y_pred_biased = _make_biased_band_predictions(n=4000, seed=0)
    oof_idx, test_idx = np.arange(0, 2500), np.arange(2500, 4000)

    factor = find_prediction_band_shift(y_true[oof_idx], y_pred_biased[oof_idx], lo=0.4, hi=1.0)
    corrected = apply_prediction_band_correction(y_pred_biased, lo=0.4, hi=1.0, factor=factor)

    brier_before = float(brier_score_loss(y_true[test_idx], y_pred_biased[test_idx]))
    brier_after = float(brier_score_loss(y_true[test_idx], corrected[test_idx]))

    assert (
        brier_after < brier_before * 0.97
    ), f"expected the band correction to improve Brier score on held-out data by >=3%, got after={brier_after:.4f} before={brier_before:.4f}"


def test_band_correction_does_not_hurt_already_calibrated_predictions():
    # a band selected purely by thresholding y_pred has a genuine selection effect even when the underlying
    # model is well-calibrated (conditioning on y_pred > lo already selects more positives) -- what matters
    # is that fitting the correction on one split and applying it to a DISJOINT split doesn't materially hurt
    # Brier score, i.e. the fitted factor generalizes rather than overfitting split-specific noise.
    """Band correction does not hurt already calibrated predictions."""
    rng = np.random.default_rng(1)
    n = 4000
    y_true = rng.integers(0, 2, n).astype(float)
    y_pred = np.clip(y_true * 0.5 + rng.normal(scale=0.1, size=n) + 0.25, 0, 1)
    oof_idx, test_idx = np.arange(0, 2500), np.arange(2500, 4000)

    factor = find_prediction_band_shift(y_true[oof_idx], y_pred[oof_idx], lo=0.4, hi=1.0)
    corrected = apply_prediction_band_correction(y_pred, lo=0.4, hi=1.0, factor=factor)

    brier_before = brier_score_loss(y_true[test_idx], y_pred[test_idx])
    brier_after = brier_score_loss(y_true[test_idx], corrected[test_idx])
    assert brier_after < brier_before * 1.1  # generalizes -- no material degradation on held-out data.


def test_apply_prediction_band_correction_leaves_out_of_band_predictions_unchanged():
    """Apply prediction band correction leaves out of band predictions unchanged."""
    preds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    corrected = apply_prediction_band_correction(preds, lo=0.4, hi=1.0, factor=0.5, clip=None)
    np.testing.assert_allclose(corrected, [0.1, 0.3, 0.25, 0.35, 0.45])
