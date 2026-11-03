"""shapley_blend with coalition_blend='rank_mean' applies its weights to ranks, the scale they were learned on."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from mlframe.votenrank.shapley_blend import shapley_blend


def _bed():
    """Three models on wildly different scales that rank the same binary target."""
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.5).astype(float)
    base = y + rng.normal(0, 0.8, n)
    preds = np.stack([base, 1000.0 * (base + rng.normal(0, 0.3, n)), 0.001 * (base + rng.normal(0, 0.5, n))])
    return preds, y


def test_rank_mean_ensemble_is_a_weighted_mean_of_ranks():
    """The returned prediction equals sum(w_m * rank_m / n), so it lies in [0, 1] regardless of each model's scale."""
    preds, y = _bed()
    out = shapley_blend(preds, y, coalition_blend="rank_mean", n_permutations=30, rng=np.random.default_rng(0))
    ranks = np.stack([rankdata(r) / preds.shape[1] for r in preds])
    expected = (out["weights"][:, None] * ranks).sum(axis=0)
    np.testing.assert_allclose(out["ensemble_pred"], expected)
    assert float(out["ensemble_pred"].max()) <= 1.0 + 1e-9


def test_mean_blend_still_uses_raw_predictions():
    """The default mean coalition keeps blending raw predictions."""
    preds, y = _bed()
    out = shapley_blend(preds, y, n_permutations=30, rng=np.random.default_rng(0))
    expected = (out["weights"][:, None] * preds).sum(axis=0)
    np.testing.assert_allclose(out["ensemble_pred"], expected)
