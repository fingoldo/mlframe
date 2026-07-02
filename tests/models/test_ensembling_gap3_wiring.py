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
        *preds, ensemble_method="rank_average", max_mae_relative=0, max_std_relative=0,
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
    strong = np.clip(y + rng.normal(0, 0.20, n), 0, 1)   # tracks the label
    weak1 = np.clip(0.5 + rng.normal(0, 0.45, n), 0, 1)  # near-random
    weak2 = rng.random(n)                                 # pure noise
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


def test_nnls_still_default_when_caruana_off():
    from mlframe.models.ensembling.score_flavours import run_stacking_aware_gate

    rng = np.random.default_rng(3)
    n = 300
    y = rng.integers(0, 2, size=n).astype(np.float64)
    preds = [np.clip(y + rng.normal(0, 0.3, n), 0, 1) for _ in range(3)]
    res: dict = {}
    run_stacking_aware_gate(
        enable_stacking_aware_gate=True, _gate_preds_for_check=preds, target_arr=y,
        level_models_and_predictions=[None] * 3, _ensemble_member_tags=["a", "b", "c"],
        stacking_gate_min_weight=0.0, use_nnls_weights=True, res=res, verbose=False,
    )
    assert res["_stacking_gate"]["weight_method"] == "nnls"
