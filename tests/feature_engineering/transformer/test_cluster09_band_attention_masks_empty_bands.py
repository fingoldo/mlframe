"""Cluster 09 regression: empty bands are masked out of the softmax in every band-attention module (audit 2026-11-20, finding 09-F41).

With a constant baseline prediction every residual (or prediction) ties, so one band holds all the rows and the others are empty. They used to keep the
global centroid as a live anchor and draw attention mass; now, as in quantile_band_attention, they get exactly zero weight.
"""

import numpy as np
import pytest

import mlframe.feature_engineering.transformer.prediction_band_attention as pba
import mlframe.feature_engineering.transformer.residual_band_attention as rba
import mlframe.feature_engineering.transformer.signed_residual_band as srb


@pytest.mark.parametrize(
    "mod, fn_name",
    [
        (rba, "compute_residual_band_attention_features"),
        (srb, "compute_signed_residual_band_features"),
        (pba, "compute_prediction_band_attention_features"),
    ],
)
def test_empty_bands_get_zero_weight(monkeypatch, mod, fn_name):
    """A constant baseline leaves a single populated band: it takes weight 1.0 and the empty bands exactly 0.0."""
    monkeypatch.setattr(mod, "_fit_baseline_predict", lambda Xt, y, **kw: np.full(len(y), 0.5, dtype=np.float32))
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3)).astype(np.float32)
    y = np.ones(200, dtype=np.float32)  # constant target + constant baseline: every residual ties
    df = getattr(mod, fn_name)(X, y, rng.standard_normal((10, 3)).astype(np.float32), seed=1, task="binary", n_bands=4)
    w = df.select([c for c in df.columns if "_w_" in c]).to_numpy()
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)
    populated = (w > 0).sum(axis=1)
    assert (populated == 1).all(), f"only one band holds rows, got {populated}"
