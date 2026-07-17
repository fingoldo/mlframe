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
    assert (
        improvement > 0.025
    ), f"expected geometric blend to beat the best arithmetic blend by >2.5% log-loss, got geo={geo_result['loss']:.4f} arith={arith_result['loss']:.4f} (improvement={improvement:.4f})"


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


def _make_near_zero_glitch_dataset(n: int, seed: int):
    """Otherwise-good models, but one model occasionally emits a near-zero glitch probability on a row where
    the true label is 1 -- a sensor dropout / model-specific outlier, not evidence the row is actually
    negative. A pure geometric mean collapses the whole product on those rows; a hybrid alpha blend should
    recover a sane prediction closer to what the other (still-good) models say.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    true_logit = 4.0 * z
    y = (rng.uniform(size=n) < _sigmoid(true_logit)).astype(int)

    # Models 1 and 2 are weak/near-uninformative on their own (heavily noised versions of z), so the
    # loss-minimizing optimizer is forced to lean heavily on model 3 to get a good fit.
    p1 = _sigmoid(0.4 * z + rng.normal(scale=2.5, size=n))
    p2 = _sigmoid(0.4 * z + rng.normal(scale=2.5, size=n))

    # Model 3 is by far the strongest individual predictor (near-noiseless on the true logit), so the
    # optimizer must give it a large exponent to minimize loss, but it glitches to ~1e-6 on a random 5% of
    # positive rows (e.g. sensor dropout) -- downweighting it globally to dodge the glitch would throw away
    # real signal on the other 95% of rows, which is exactly the trap a pure geometric mean falls into.
    p3 = _sigmoid(4.0 * z + rng.normal(scale=0.2, size=n))
    glitch_mask = (rng.uniform(size=n) < 0.05) & (y == 1)
    p3 = np.where(glitch_mask, 1e-6, p3)

    return y, [p1, p2, p3], glitch_mask


def test_biz_val_geometric_weight_blend_hybrid_alpha_recovers_from_near_zero_glitch():
    y, preds, glitch_mask = _make_near_zero_glitch_dataset(n=4000, seed=1)

    pure_geo = geometric_weight_blend(preds, y, _log_loss, n_restarts=5, random_state=0)
    hybrid = geometric_weight_blend(preds, y, _log_loss, n_restarts=5, random_state=0, fit_alpha=True)

    # On the glitched rows specifically, pure geometric blending should be tanked toward 0 (wrong, since
    # y==1 there) while the hybrid blend keeps a materially higher, saner prediction.
    assert glitch_mask.sum() > 50, "need enough glitched rows for a stable comparison"
    pure_geo_glitch_pred = pure_geo["ensemble_pred"][glitch_mask]
    hybrid_glitch_pred = hybrid["ensemble_pred"][glitch_mask]
    assert np.median(pure_geo_glitch_pred) < 0.05, f"expected pure geometric blend to tank glitched rows, got median={np.median(pure_geo_glitch_pred):.4f}"
    assert np.median(hybrid_glitch_pred) > np.median(pure_geo_glitch_pred) * 3, (
        f"expected hybrid blend to materially recover glitched rows, pure_geo_median={np.median(pure_geo_glitch_pred):.4f} "
        f"hybrid_median={np.median(hybrid_glitch_pred):.4f}"
    )

    # Whole-dataset log-loss: the hybrid blend must not be worse than the pure geometric blend (it's fit to
    # minimize the same loss_fn, alpha=1 -- pure geometric -- is always in its feasible search space).
    assert hybrid["loss"] <= pure_geo["loss"] + 1e-9, f"hybrid fit_alpha loss ({hybrid['loss']:.4f}) should be <= pure geometric loss ({pure_geo['loss']:.4f})"

    improvement = (pure_geo["loss"] - hybrid["loss"]) / pure_geo["loss"]
    assert (
        improvement > 0.03
    ), f"expected hybrid blend to beat pure geometric blend by >3% log-loss on the glitch-prone dataset, got improvement={improvement:.4f} (pure_geo={pure_geo['loss']:.4f}, hybrid={hybrid['loss']:.4f})"
    assert hybrid["alpha"] < 1.0, f"expected fit_alpha to move away from pure-geometric (alpha=1), got {hybrid['alpha']}"


def test_geometric_weight_blend_default_unchanged_when_alpha_omitted():
    y, preds = _make_independent_evidence_dataset(n=500, seed=2)
    baseline = geometric_weight_blend(preds, y, _log_loss, n_restarts=3, random_state=0)
    still_default = geometric_weight_blend(preds, y, _log_loss, n_restarts=3, random_state=0)
    assert set(baseline.keys()) == {"exponents", "ensemble_pred", "loss"}
    np.testing.assert_array_equal(baseline["exponents"], still_default["exponents"])
    np.testing.assert_array_equal(baseline["ensemble_pred"], still_default["ensemble_pred"])
    assert baseline["loss"] == still_default["loss"]


def test_geometric_weight_blend_fixed_alpha_matches_convex_combination():
    y, preds = _make_independent_evidence_dataset(n=500, seed=3)
    base = geometric_weight_blend(preds, y, _log_loss, n_restarts=3, random_state=0)
    fixed = geometric_weight_blend(preds, y, _log_loss, n_restarts=3, random_state=0, alpha=0.7)
    np.testing.assert_allclose(fixed["exponents"], base["exponents"])
    expected = 0.7 * fixed["geometric_pred"] + 0.3 * fixed["arithmetic_pred"]
    np.testing.assert_allclose(fixed["ensemble_pred"], expected)
    assert fixed["alpha"] == 0.7
