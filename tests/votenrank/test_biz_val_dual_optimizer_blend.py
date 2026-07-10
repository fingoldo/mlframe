"""biz_value test for ``votenrank.dual_optimizer_blend.dual_optimizer_weight_blend``.

The win (1st_mechanisms-of-action-moa-prediction.md): two independent optimizers (SLSQP, Optuna TPE) run on
the SAME OOF objective should converge to nearly identical weights when the objective has a clear, real
optimum -- corroborating that the found weights aren't an artifact of one optimizer's search bias. And a
model pool with genuinely useless (pure-noise) candidates should have both optimizers independently drive
those candidates' weights toward zero, correctly flagging them as pruning candidates.
"""
from __future__ import annotations

import numpy as np

from mlframe.votenrank.dual_optimizer_blend import dual_optimizer_weight_blend


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def test_biz_val_dual_optimizer_blend_low_divergence_on_clear_optimum():
    rng = np.random.default_rng(0)
    n = 1500
    y_true = rng.standard_normal(n) * 3.0
    good_preds = [y_true + 0.3 * rng.standard_normal(n) for _ in range(3)]

    result = dual_optimizer_weight_blend(good_preds, y_true, _rmse, n_restarts=5, n_optuna_trials=150, random_state=0)

    assert result["max_weight_divergence"] < 0.15, f"expected the two independent optimizers to corroborate (low divergence) on a clear-optimum objective, got {result['max_weight_divergence']:.4f}"
    assert np.isclose(result["slsqp_weights"].sum(), 1.0)


def test_biz_val_dual_optimizer_blend_prunes_pure_noise_models():
    rng = np.random.default_rng(1)
    n = 1500
    y_true = rng.standard_normal(n) * 3.0
    good_preds = [y_true + 0.3 * rng.standard_normal(n) for _ in range(3)]
    noise_preds = [rng.standard_normal(n) * 3.0 for _ in range(2)]  # uncorrelated with y_true -- pure noise
    preds = good_preds + noise_preds

    result = dual_optimizer_weight_blend(preds, y_true, _rmse, n_restarts=5, n_optuna_trials=150, random_state=1, zero_weight_threshold=0.05)

    assert set(result["prune_candidates"].tolist()).issuperset({3, 4}), f"expected the two pure-noise models (indices 3, 4) to be flagged for pruning, got {result['prune_candidates']}"


def test_dual_optimizer_blend_both_optimizers_beat_equal_weight():
    rng = np.random.default_rng(2)
    n = 1000
    y_true = rng.standard_normal(n) * 3.0
    preds = [y_true + 0.3 * rng.standard_normal(n) for _ in range(2)] + [y_true + 2.0 * rng.standard_normal(n) for _ in range(3)]
    equal_weight_rmse = _rmse(y_true, np.mean(preds, axis=0))

    result = dual_optimizer_weight_blend(preds, y_true, _rmse, n_restarts=5, n_optuna_trials=150, random_state=2)

    assert result["slsqp_loss"] < equal_weight_rmse
    assert result["optuna_loss"] < equal_weight_rmse
