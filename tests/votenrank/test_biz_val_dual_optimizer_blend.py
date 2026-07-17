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

    assert result["max_weight_divergence"] < 0.15, (
        f"expected the two independent optimizers to corroborate (low divergence) on a clear-optimum objective, got {result['max_weight_divergence']:.4f}"
    )
    assert np.isclose(result["slsqp_weights"].sum(), 1.0)


def test_biz_val_dual_optimizer_blend_prunes_pure_noise_models():
    rng = np.random.default_rng(1)
    n = 1500
    y_true = rng.standard_normal(n) * 3.0
    good_preds = [y_true + 0.3 * rng.standard_normal(n) for _ in range(3)]
    noise_preds = [rng.standard_normal(n) * 3.0 for _ in range(2)]  # uncorrelated with y_true -- pure noise
    preds = good_preds + noise_preds

    result = dual_optimizer_weight_blend(preds, y_true, _rmse, n_restarts=5, n_optuna_trials=150, random_state=1, zero_weight_threshold=0.05)

    assert set(result["prune_candidates"].tolist()).issuperset({3, 4}), (
        f"expected the two pure-noise models (indices 3, 4) to be flagged for pruning, got {result['prune_candidates']}"
    )


def test_biz_val_dual_optimizer_blend_coord_descent_catches_correlated_blind_spot():
    """SLSQP (local gradient descent) and Optuna TPE (Bayesian surrogate sampling) are both attracted to the
    SAME broad, gentle decoy basin near equal weights on this adversarial landscape and report LOW divergence
    between each other -- which the 2-way check alone would read as "corroborated". A third, mechanically
    distinct optimizer (gradient-free pairwise-coordinate descent) independently finds a narrow, much deeper
    true optimum near a single-model vertex that both other optimizers miss, proving 3-way triangulation
    catches a correlated blind spot invisible to the pairwise check.

    The landscape is engineered (via an identity-matrix ``oof_preds`` trick so the blended prediction IS the
    raw weight vector) with a flat/gentle decoy bowl near uniform weights and a narrow, much lower true optimum
    near a single-model vertex -- deterministic and reproducible for the fixed ``random_state=2``.
    """
    n_models = 5
    w_true = np.array([0.02, 0.02, 0.02, 0.02, 0.92])
    dip_radius = 0.12
    decoy_loss = 1.0
    dip_depth = 0.6

    def _adversarial_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        w = y_pred  # identity-trick: the "blended prediction" IS the weight vector itself
        d_true = float(np.linalg.norm(w - w_true))
        if d_true < dip_radius:
            return decoy_loss - dip_depth * (1 - d_true / dip_radius)
        decoy_center = np.full(len(w), 1.0 / len(w))
        d_decoy = float(np.linalg.norm(w - decoy_center))
        return decoy_loss - 0.05 * np.exp(-d_decoy * 2)

    preds = [np.eye(n_models)[i] for i in range(n_models)]
    y_true = np.zeros(n_models)

    result = dual_optimizer_weight_blend(
        preds,
        y_true,
        _adversarial_loss,
        n_restarts=5,
        n_optuna_trials=150,
        random_state=2,
        include_coord_descent=True,
        n_coord_descent_iters=400,
    )

    # SLSQP and Optuna land on the SAME decoy basin -- low divergence between them, reading as "corroborated".
    assert result["max_weight_divergence"] < 0.05, (
        f"expected SLSQP/Optuna to agree (correlated blind spot precondition), got {result['max_weight_divergence']:.4f}"
    )
    # Yet coordinate descent independently finds a meaningfully better (lower) loss than either of them.
    assert result["coord_descent_loss"] < 0.6 * min(result["slsqp_loss"], result["optuna_loss"]), (
        f"expected the third optimizer to find a meaningfully better optimum than the two agreeing ones, "
        f"got coord_descent_loss={result['coord_descent_loss']:.4f} vs slsqp={result['slsqp_loss']:.4f} optuna={result['optuna_loss']:.4f}"
    )
    # The triangulated flag must fire: this is exactly the correlated-blind-spot case it exists to catch.
    assert result["correlated_blind_spot_detected"] is True
    # And the 3-way divergence must be much larger than the 2-way (SLSQP vs Optuna) divergence caught it missed.
    assert result["triangulated_max_divergence"] > 5 * result["max_weight_divergence"]


def test_dual_optimizer_blend_both_optimizers_beat_equal_weight():
    rng = np.random.default_rng(2)
    n = 1000
    y_true = rng.standard_normal(n) * 3.0
    preds = [y_true + 0.3 * rng.standard_normal(n) for _ in range(2)] + [y_true + 2.0 * rng.standard_normal(n) for _ in range(3)]
    equal_weight_rmse = _rmse(y_true, np.mean(preds, axis=0))

    result = dual_optimizer_weight_blend(preds, y_true, _rmse, n_restarts=5, n_optuna_trials=150, random_state=2)

    assert result["slsqp_loss"] < equal_weight_rmse
    assert result["optuna_loss"] < equal_weight_rmse
