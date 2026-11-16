"""Regression: the MAD pad in ``predictor_outlier_signature`` must scale with the predictors, not be an absolute 1e-12."""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.ensemble_features import predictor_outlier_signature


def test_mad_pad_relative_scale_flags_dissenter_on_tiny_scale_predictors():
    """Identical predictors at 1e-15 scale with one dissenter: the dissenter must be flagged (the absolute pad hid it)."""
    base = 1e-15
    preds = np.full((4, 5), base)
    preds[:, 2] = 6 * base
    n_out, idx = predictor_outlier_signature(preds)
    assert n_out.tolist() == [1.0] * 4
    assert idx.tolist() == [2.0] * 4


def test_mad_pad_relative_scale_ordinary_scale_results_unchanged():
    """On ordinary-scale data the result equals the old absolute-pad formula, including the MAD=0 dissenter case."""
    rng = np.random.default_rng(0)
    arr = rng.normal(0.5, 0.2, size=(2000, 7))
    arr[:10] = 0.5
    arr[:10, 3] = 0.9
    med = np.median(arr, axis=1, keepdims=True)
    dev = np.abs(arr - med)
    old = (dev > 2.5 * (np.median(dev, axis=1, keepdims=True) + 1e-12) * 1.4826).sum(axis=1).astype(float)
    n_out, _ = predictor_outlier_signature(arr)
    assert np.array_equal(n_out, old)
    assert n_out[:10].tolist() == [1.0] * 10
