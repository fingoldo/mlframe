"""Gap-3 wiring: rank_average as a rank-fusion ensemble method + Caruana metric-direct blend weights (PZAD ensemble)."""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------- rank_average
def test_rank_average_registered_and_matches_blend():
    from mlframe.models.ensembling.base import RANK_FUSION_METHODS, combine_probs
    from mlframe.models.ensembling.selection import rank_average_blend

    assert "rank_average" in RANK_FUSION_METHODS
    rng = np.random.default_rng(0)
    stacked = rng.random((4, 60, 3))  # M=4, N=60, K=3
    out = combine_probs(stacked, "rank_average")
    assert out.shape == (60, 3)
    assert (out >= 0).all() and (out <= 1).all()
    # combine_probs must not drift from the standalone blend (single source of truth).
    assert np.allclose(out, rank_average_blend(stacked, normalise=True))


def test_rank_average_through_public_ensemble_path():
    from mlframe.models.ensembling.predict import ensemble_probabilistic_predictions

    rng = np.random.default_rng(1)
    preds = [rng.random((50, 2)) for _ in range(3)]
    out, _, _ = ensemble_probabilistic_predictions(
        *preds,
        ensemble_method="rank_average",
        max_mae_relative=0,
        max_std_relative=0,
    )
    assert out is not None and out.shape == (50, 2)


def test_unknown_method_still_rejected():
    from mlframe.models.ensembling.predict import ensemble_probabilistic_predictions

    with pytest.raises(ValueError):
        ensemble_probabilistic_predictions(np.random.random((10, 2)), ensemble_method="not_a_method")


# ---------------------------------------------------------------- Caruana weights
def test_caruana_weights_favor_the_strong_member():
    """use_caruana_weights fits blend weights by metric-direct greedy selection on the OOF preds. Given a clearly
    stronger member (near-truth) and two near-random ones, the aligned weight vector must put the most weight on the
    strong member -- the metric-direct fit, an alternative to NNLS, reachable via the score_ensemble flag."""
    from mlframe.models.ensembling.score_flavours import run_stacking_aware_gate

    rng = np.random.default_rng(2)
    n = 500
    y = rng.integers(0, 2, size=n).astype(np.float64)
    strong = np.clip(y + rng.normal(0, 0.20, n), 0, 1)  # tracks the label
    weak1 = np.clip(0.5 + rng.normal(0, 0.45, n), 0, 1)  # near-random
    weak2 = rng.random(n)  # pure noise
    res: dict = {}
    w = run_stacking_aware_gate(
        enable_stacking_aware_gate=True,
        _gate_preds_for_check=[strong, weak1, weak2],
        target_arr=y,
        level_models_and_predictions=[None, None, None],
        _ensemble_member_tags=["strong", "weak1", "weak2"],
        stacking_gate_min_weight=0.0,
        use_nnls_weights=False,
        res=res,
        verbose=False,
        use_caruana_weights=True,
    )
    assert w is not None and w.shape == (3,)
    assert abs(float(w.sum()) - 1.0) < 1e-9
    assert res["_stacking_gate"]["weight_method"] == "caruana"
    assert w[0] == w.max() and w[0] > 0.5, f"caruana should concentrate weight on the strong member, got {w}"


# ---------------------------------------------------------------- biz_value
def test_biz_val_caruana_blend_beats_equal_weight_error():
    """Caruana's value is the OUTCOME, not just the weight vector: the metric-direct greedy blend must produce a LOWER
    prediction error than the naive equal-weight average when one member is strong and the others are near-random.
    Equal-weight dilutes the strong member with noise; Caruana concentrates on it and cuts the blended MSE."""
    from mlframe.models.ensembling.score_flavours import run_stacking_aware_gate

    rng = np.random.default_rng(7)
    n = 800
    y = rng.integers(0, 2, size=n).astype(np.float64)
    strong = np.clip(y + rng.normal(0, 0.20, n), 0, 1)
    weak1 = np.clip(0.5 + rng.normal(0, 0.45, n), 0, 1)
    weak2 = rng.random(n)
    members = np.vstack([strong, weak1, weak2])  # (3, n)
    res: dict = {}
    w = run_stacking_aware_gate(
        enable_stacking_aware_gate=True,
        _gate_preds_for_check=[strong, weak1, weak2],
        target_arr=y,
        level_models_and_predictions=[None, None, None],
        _ensemble_member_tags=["strong", "weak1", "weak2"],
        stacking_gate_min_weight=0.0,
        use_nnls_weights=False,
        res=res,
        verbose=False,
        use_caruana_weights=True,
    )
    caruana_blend = w @ members
    equal_blend = members.mean(axis=0)
    caruana_mse = float(np.mean((caruana_blend - y) ** 2))
    equal_mse = float(np.mean((equal_blend - y) ** 2))
    assert caruana_mse < equal_mse - 0.02, f"caruana blend MSE {caruana_mse:.3f} should beat equal-weight {equal_mse:.3f}"


def test_biz_val_rank_average_beats_arithmetic_on_miscalibrated_members():
    """Rank fusion's value: a member can be the BEST ranker yet contribute a tiny magnitude swing (compressed / poorly
    calibrated probabilities). The arithmetic mean weights by magnitude, so it is drowned out by wider but weaker
    members; rank_average weights every member by its ORDERING, so the strong ranker counts fully and lifts the AUC."""
    from sklearn.metrics import roc_auc_score

    from mlframe.models.ensembling.base import combine_probs

    rng = np.random.default_rng(11)
    n = 1500
    y = rng.integers(0, 2, size=n)

    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    # Strong ranker, but its probabilities are compressed to a narrow band around 0.5 (near-zero magnitude swing).
    strong = 0.5 + 0.01 * ((2.0 * y - 1.0) + rng.normal(0, 0.4, n))
    strong = np.clip(strong, 0.0, 1.0)
    # Two mediocre rankers on the full [0,1] range -- they dominate a magnitude-weighted average.
    wide1 = _sigmoid(0.6 * ((2.0 * y - 1.0) + rng.normal(0, 1.6, n)))
    wide2 = _sigmoid(0.6 * ((2.0 * y - 1.0) + rng.normal(0, 1.6, n)))

    def _as_probs(p):
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    stacked = np.stack([_as_probs(strong), _as_probs(wide1), _as_probs(wide2)], axis=0)  # (3, n, 2)
    auc_rank = roc_auc_score(y, combine_probs(stacked, "rank_average")[:, 1])
    auc_arith = roc_auc_score(y, combine_probs(stacked, "arithm")[:, 1])
    assert auc_rank >= auc_arith + 0.03, f"rank_average AUC {auc_rank:.3f} should beat arithmetic mean {auc_arith:.3f} on miscalibrated members"


def test_nnls_still_default_when_caruana_off():
    from mlframe.models.ensembling.score_flavours import run_stacking_aware_gate

    rng = np.random.default_rng(3)
    n = 300
    y = rng.integers(0, 2, size=n).astype(np.float64)
    preds = [np.clip(y + rng.normal(0, 0.3, n), 0, 1) for _ in range(3)]
    res: dict = {}
    run_stacking_aware_gate(
        enable_stacking_aware_gate=True,
        _gate_preds_for_check=preds,
        target_arr=y,
        level_models_and_predictions=[None] * 3,
        _ensemble_member_tags=["a", "b", "c"],
        stacking_gate_min_weight=0.0,
        use_nnls_weights=True,
        res=res,
        verbose=False,
    )
    assert res["_stacking_gate"]["weight_method"] == "nnls"
