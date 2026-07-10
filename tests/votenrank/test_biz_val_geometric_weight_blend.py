"""biz_value test for ``votenrank.geometric_weight_blend``.

The win: when two probability-forecasting models each observe a DIFFERENT, conditionally-independent slice
of evidence (a classic "combine independent expert opinions" setup), the mathematically correct combiner
multiplies the models' ODDS (equivalently, sums their LOGITS) -- an arithmetic mean of raw probabilities is a
well-known suboptimal combiner here (it systematically under-confidently blends toward 0.5 relative to what
the combined evidence actually supports). A geometric mean of probabilities with fitted per-model exponents
approximates the log-odds-summing combiner far better, giving materially lower log-loss than the best
achievable arithmetic-mean blend.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss

from mlframe.votenrank.constrained_weight_blend import constrained_weight_blend
from mlframe.votenrank.geometric_weight_blend import geometric_weight_blend


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-z)))


def _make_independent_evidence_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    coef = 4.0
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    true_logit = coef * (x1 + x2 + x3)
    y = (rng.uniform(size=n) < _sigmoid(true_logit)).astype(int)

    # Each model only observes its own slice of evidence and is individually well-calibrated for that slice.
    preds = [_sigmoid(coef * x1), _sigmoid(coef * x2), _sigmoid(coef * x3)]
    return y, preds


def _log_loss(y_true, y_pred):
    return float(log_loss(y_true, np.clip(y_pred, 1e-7, 1 - 1e-7)))


def test_biz_val_geometric_blend_beats_arithmetic_blend_on_independent_evidence():
    y, preds = _make_independent_evidence_dataset(n=3000, seed=0)

    geo_result = geometric_weight_blend(preds, y, _log_loss, n_restarts=5, random_state=0)
    arith_result = constrained_weight_blend(preds, y, _log_loss, n_restarts=5, random_state=0)

    improvement = (arith_result["loss"] - geo_result["loss"]) / arith_result["loss"]
    assert improvement > 0.025, f"expected geometric blend to beat the best arithmetic blend by >2.5% log-loss, got geo={geo_result['loss']:.4f} arith={arith_result['loss']:.4f} (improvement={improvement:.4f})"


def test_geometric_weight_blend_single_model_pool():
    y = np.array([0, 1, 1, 0, 1])
    preds = [np.array([0.2, 0.8, 0.7, 0.3, 0.9])]
    result = geometric_weight_blend(preds, y, _log_loss, n_restarts=2)
    assert result["exponents"].shape == (1,)
    assert result["exponents"][0] > 0


def test_geometric_weight_blend_empty_pool_raises():
    import pytest

    with pytest.raises(ValueError):
        geometric_weight_blend([], np.array([1.0]), _log_loss)
