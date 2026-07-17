"""biz_value test for ``votenrank.constrained_weight_blend``.

The win: on a pool of 20 OOF prediction arrays where 5 are genuinely good (low-noise) and 15 are mediocre
(high-noise), the simplex-constrained weight optimizer should converge to weights that overwhelmingly favor
the good models and beat a naive equal-weight average over the full noisy pool.
"""

from __future__ import annotations

import numpy as np

from mlframe.votenrank.constrained_weight_blend import constrained_weight_blend


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _make_model_pool(n_samples: int, n_good: int, n_bad: int, seed: int):
    rng = np.random.default_rng(seed)
    y_true = rng.standard_normal(n_samples) * 3.0
    good_preds = [y_true + 0.3 * rng.standard_normal(n_samples) for _ in range(n_good)]
    bad_preds = [y_true + 2.5 * rng.standard_normal(n_samples) for _ in range(n_bad)]
    return y_true, good_preds + bad_preds


def test_biz_val_constrained_weight_blend_beats_equal_weight_average():
    y_true, preds = _make_model_pool(n_samples=2000, n_good=5, n_bad=15, seed=42)

    result = constrained_weight_blend(preds, y_true, _rmse, n_restarts=5, random_state=0)

    equal_weight_pred = np.mean(preds, axis=0)
    equal_weight_rmse = _rmse(y_true, equal_weight_pred)

    assert (
        result["loss"] < equal_weight_rmse
    ), f"constrained blend should beat naive equal-weight averaging: blend={result['loss']:.4f} equal_weight={equal_weight_rmse:.4f}"
    assert np.isclose(result["weights"].sum(), 1.0)
    assert np.all(result["weights"] >= -1e-9)
    assert result["weights"][:5].sum() > 0.7, f"weights should overwhelmingly favor the good models, got {result['weights']}"


def test_constrained_weight_blend_single_model_pool_returns_weight_one():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    preds = [y_true + 0.1]
    result = constrained_weight_blend(preds, y_true, _rmse, n_restarts=2)
    assert np.isclose(result["weights"][0], 1.0)


def test_constrained_weight_blend_empty_pool_raises():
    import pytest

    with pytest.raises(ValueError):
        constrained_weight_blend([], np.array([1.0]), _rmse)


def test_biz_val_constrained_weight_blend_sparsity_cap_selects_small_subset():
    """On a large pool where only a few variants are genuinely useful, ``max_nonzero_weights`` should
    prune down to a small production-friendly subset with quality close to the dense solution."""
    y_true, preds = _make_model_pool(n_samples=2000, n_good=4, n_bad=36, seed=7)

    dense = constrained_weight_blend(preds, y_true, _rmse, n_restarts=5, random_state=0)
    sparse = constrained_weight_blend(preds, y_true, _rmse, n_restarts=5, random_state=0, max_nonzero_weights=6)

    assert dense["n_nonzero"] > 6, f"dense solve should spread weight across more than 6 models, got {dense['n_nonzero']}"
    assert sparse["n_nonzero"] <= 6, f"sparse solve must respect the cap, got {sparse['n_nonzero']}"

    quality_loss_pct = (sparse["loss"] - dense["loss"]) / dense["loss"]
    assert quality_loss_pct < 0.08, (
        f"sparse blend (6 nonzero) should stay within 8% RMSE of the dense blend ({dense['n_nonzero']} nonzero): "
        f"sparse={sparse['loss']:.4f} dense={dense['loss']:.4f} loss_pct={quality_loss_pct:.4f}"
    )
    good_weight_share = sparse["weights"][:4].sum()
    assert good_weight_share > 0.85, (
        f"sparse solve should concentrate weight on the genuinely good models, got good_weight_share={good_weight_share:.4f} "
        f"selected_indices={sparse['selected_indices']}"
    )
    assert np.isclose(sparse["weights"].sum(), 1.0)
