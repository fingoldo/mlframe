"""Regression sensor: NNLS weights computed by ``stacking_aware_gate`` are
actually fed into ``combine_probs`` (AP7).

Before the AP7 wiring the NNLS gate was observational: weights were stamped to
``res["_stacking_gate"]`` but ``combine_probs`` continued to call
``np.mean(stacked, axis=0)`` for arithm and per-flavour equivalents elsewhere.
This file pins the new contract:

1. ``combine_probs(precomputed_weights=...)`` produces the expected weighted
   mean / harmonic / geometric / quadratic / cubic blends.
2. ``ensemble_probabilistic_predictions(precomputed_weights=...)`` forwards
   weights through and re-aligns them when a None member is filtered out.
3. Validation errors fire on mismatched shape, NaN/inf, negatives, sum==0.
4. ``score_ensemble`` with a 3-member regression suite stamps an aligned
   weight vector on ``res["_stacking_gate"]["aligned_weights"]`` AND the
   resulting blend matches the NNLS-weighted arithmetic mean within fp64 tol.
5. ``use_nnls_weights=False`` keeps the diagnostic but reverts the blend to
   the uniform 1/M arithmetic mean.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.models.ensembling.base import combine_probs
from mlframe.models.ensembling.predict import ensemble_probabilistic_predictions
from mlframe.models.ensembling import score_ensemble


# ---------------------------------------------------------------------------
# 1. combine_probs direct weighted math
# ---------------------------------------------------------------------------


def test_combine_probs_arithm_with_precomputed_weights_matches_expected():
    rng = np.random.default_rng(0)
    stacked = rng.uniform(0.1, 0.9, size=(3, 50, 2))
    weights = np.array([0.6, 0.3, 0.1])
    out = combine_probs(stacked, "arithm", precomputed_weights=weights)
    expected = (
        weights[0] * stacked[0] + weights[1] * stacked[1] + weights[2] * stacked[2]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_combine_probs_arithm_uniform_when_no_weights():
    rng = np.random.default_rng(1)
    stacked = rng.uniform(0.1, 0.9, size=(4, 30))
    out = combine_probs(stacked, "arithm", precomputed_weights=None)
    np.testing.assert_allclose(out, np.mean(stacked, axis=0), rtol=1e-12)


def test_combine_probs_harm_weighted_matches_definition():
    stacked = np.array([[0.5, 0.5], [0.25, 0.25], [0.1, 0.1]], dtype=np.float64)
    weights = np.array([0.5, 0.3, 0.2])
    out = combine_probs(stacked, "harm", precomputed_weights=weights)
    expected = 1.0 / (
        weights[0] / stacked[0] + weights[1] / stacked[1] + weights[2] / stacked[2]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_combine_probs_geo_weighted_matches_definition():
    stacked = np.array([[0.5, 0.4], [0.3, 0.2], [0.6, 0.1]], dtype=np.float64)
    weights = np.array([0.5, 0.3, 0.2])
    out = combine_probs(stacked, "geo", precomputed_weights=weights)
    expected = np.exp(
        weights[0] * np.log(stacked[0])
        + weights[1] * np.log(stacked[1])
        + weights[2] * np.log(stacked[2])
    )
    np.testing.assert_allclose(out, expected, rtol=1e-10)


def test_combine_probs_quad_weighted_matches_definition():
    stacked = np.array([[0.5, 0.4], [0.3, 0.2], [0.6, 0.1]], dtype=np.float64)
    weights = np.array([0.5, 0.3, 0.2])
    out = combine_probs(stacked, "quad", precomputed_weights=weights)
    expected = np.sqrt(
        weights[0] * stacked[0] ** 2
        + weights[1] * stacked[1] ** 2
        + weights[2] * stacked[2] ** 2
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_combine_probs_qube_weighted_matches_definition():
    stacked = np.array([[0.5, 0.4], [0.3, 0.2], [0.6, 0.1]], dtype=np.float64)
    weights = np.array([0.5, 0.3, 0.2])
    out = combine_probs(stacked, "qube", precomputed_weights=weights)
    expected = np.cbrt(
        weights[0] * stacked[0] ** 3
        + weights[1] * stacked[1] ** 3
        + weights[2] * stacked[2] ** 3
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_combine_probs_renormalises_off_by_more_than_1e_3(caplog):
    stacked = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    weights = np.array([0.4, 0.4])  # sums to 0.8, off by 0.2
    with caplog.at_level("WARNING", logger="mlframe.models.ensembling"):
        out = combine_probs(stacked, "arithm", precomputed_weights=weights)
    # Renormalised weights = [0.5, 0.5] -> arithmetic mean.
    np.testing.assert_allclose(out, np.mean(stacked, axis=0), rtol=1e-12)
    assert any("deviates from 1.0" in rec.message for rec in caplog.records)


def test_combine_probs_rejects_shape_mismatch():
    stacked = np.zeros((3, 5))
    with pytest.raises(ValueError, match="precomputed_weights shape"):
        combine_probs(stacked, "arithm", precomputed_weights=np.array([0.5, 0.5]))


def test_combine_probs_rejects_negative_weight():
    stacked = np.zeros((2, 5))
    with pytest.raises(ValueError, match="non-negative"):
        combine_probs(stacked, "arithm", precomputed_weights=np.array([-0.1, 1.1]))


def test_combine_probs_rejects_nan_weight():
    stacked = np.zeros((2, 5))
    with pytest.raises(ValueError, match="non-finite"):
        combine_probs(stacked, "arithm", precomputed_weights=np.array([np.nan, 1.0]))


def test_combine_probs_rejects_zero_sum_weight():
    stacked = np.zeros((2, 5))
    with pytest.raises(ValueError, match="sum to zero"):
        combine_probs(stacked, "arithm", precomputed_weights=np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# 2. ensemble_probabilistic_predictions forwards weights
# ---------------------------------------------------------------------------


def test_ensemble_probabilistic_predictions_forwards_weights():
    rng = np.random.default_rng(2)
    preds = [rng.uniform(0.1, 0.9, size=(20, 2)) for _ in range(3)]
    weights = np.array([0.6, 0.3, 0.1])
    out, _, _ = ensemble_probabilistic_predictions(
        *preds,
        ensemble_method="arithm",
        verbose=False,
        precomputed_weights=weights,
    )
    expected = (
        weights[0] * preds[0] + weights[1] * preds[1] + weights[2] * preds[2]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_ensemble_probabilistic_predictions_realigns_weights_after_none():
    """When a member is None its weight must be sliced out so downstream members
    keep their correct factor."""
    rng = np.random.default_rng(3)
    arr_a = rng.uniform(0.1, 0.9, size=(15, 2))
    arr_c = rng.uniform(0.1, 0.9, size=(15, 2))
    weights = np.array([0.7, 0.2, 0.1])  # member B (index 1) is None below.
    out, _, _ = ensemble_probabilistic_predictions(
        arr_a, None, arr_c,
        ensemble_method="arithm",
        verbose=False,
        precomputed_weights=weights,
    )
    # Surviving weights = [0.7, 0.1]; AP7 contract: renormalised inside combine_probs
    # to [0.875, 0.125]. Renormalisation only fires when |sum-1| > 1e-3; here 0.8 is
    # off by 0.2 so the warning + renorm path runs.
    surv = np.array([0.7, 0.1])
    surv = surv / surv.sum()
    expected = surv[0] * arr_a + surv[1] * arr_c
    np.testing.assert_allclose(out, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. score_ensemble end-to-end NNLS wire-up
# ---------------------------------------------------------------------------


def _make_regression_member(preds: np.ndarray, name: str) -> SimpleNamespace:
    """Build a stub member that score_ensemble accepts on the regression path.

    score_ensemble keys ``is_regression`` off ``*_probs is None`` and the per-
    method process reads ``val_preds`` / ``test_preds`` / ``oof_preds`` /
    ``train_preds``. Setting only those (and matching ``model_name``) is enough
    for the gate + stacking-aware-gate + per-method blend to fire end to end.
    """
    p = preds.astype(np.float64).reshape(-1)
    return SimpleNamespace(
        val_probs=None,
        test_probs=None,
        train_probs=None,
        oof_probs=None,
        val_preds=p,
        test_preds=p,
        train_preds=p,
        oof_preds=p,
        model_name=name,
        model=SimpleNamespace(__class__=type(name, (), {})),
    )


def _build_synthetic_three_member_regression():
    """3 base regressors with predictions on a known target where the NNLS
    optimum weighting is unique and clearly non-uniform (0.6 / 0.3 / 0.1)."""
    rng = np.random.default_rng(7)
    n = 400
    y = rng.uniform(-5.0, 5.0, size=n)
    # Member 0 is closest to y; member 1 slightly noisier; member 2 noisy.
    p0 = y + rng.normal(0, 0.05, size=n)
    p1 = y + rng.normal(0, 0.5, size=n)
    p2 = y + rng.normal(0, 2.0, size=n)
    members = [
        _make_regression_member(p0, "ModelA"),
        _make_regression_member(p1, "ModelB"),
        _make_regression_member(p2, "ModelC"),
    ]
    return members, y


def test_score_ensemble_applies_nnls_weights_to_blend():
    """End-to-end: score_ensemble stamps ``_stacking_gate.aligned_weights`` and
    the per-flavour arithm blend equals the NNLS-weighted sum over members."""
    members, y = _build_synthetic_three_member_regression()
    res = score_ensemble(
        models_and_predictions=members,
        ensemble_name="[ap7-test]",
        target=y,
        train_target=y,
        val_target=y,
        test_target=y,
        max_ensembling_level=1,
        ensembling_methods=["arithm"],
        enable_stacking_aware_gate=True,
        use_nnls_weights=True,
        require_oof_for_gate=False,
        build_votenrank_leaderboard=False,
        uncertainty_quantile=0.0,
        verbose=False,
        # The gate-source predictions in this stub fixture are byte-identical
        # across splits, so the median-distance check would fire and drop
        # honest members on absolute-MAE thresholds. Relax for the sensor.
        max_mae_relative=0.0,
        max_std_relative=0.0,
        diversity_corr_warn_threshold=2.0,
        k2_catastrophic_mae_ratio=0.0,
    )
    assert "_stacking_gate" in res, res
    gate = res["_stacking_gate"]
    assert gate.get("applied_to_blend") is True, gate
    aligned = np.asarray(gate["aligned_weights"], dtype=np.float64)
    assert aligned.shape == (3,)
    np.testing.assert_allclose(aligned.sum(), 1.0, rtol=1e-9)
    # The best member should dominate (its weight largest by construction).
    assert aligned[0] > aligned[1], aligned
    assert aligned[0] > aligned[2], aligned

    # The arithm flavour result must equal the NNLS-weighted blend evaluated on
    # the same per-member predictions, NOT the uniform mean.
    stacked = np.column_stack([members[i].val_preds for i in range(3)]).T  # (3, N)
    expected = aligned[0] * stacked[0] + aligned[1] * stacked[1] + aligned[2] * stacked[2]
    uniform = np.mean(stacked, axis=0)
    # ``res['arithm']`` is the train_and_evaluate_model tuple; element 0 is the SimpleNamespace
    # carrying predictions / metrics. Pre-fix probe assumed a .predictions wrapper that does not exist.
    arithm_result = res["arithm"]
    arithm_ns = arithm_result[0] if isinstance(arithm_result, tuple) else arithm_result
    actual = np.asarray(arithm_ns.val_preds, dtype=np.float64).reshape(-1)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    # Sanity: when the weights are not uniform, the blend must differ from
    # the uniform mean. If this fails the wire-up regressed.
    assert not np.allclose(actual, uniform, rtol=1e-6), (
        "NNLS weights did not change the blend (still uniform mean)"
    )


def test_score_ensemble_use_nnls_weights_false_falls_back_to_uniform():
    """With ``use_nnls_weights=False`` the gate still stamps survivors+weights but
    ``applied_to_blend`` is False and the arithm result equals the uniform mean."""
    members, y = _build_synthetic_three_member_regression()
    res = score_ensemble(
        models_and_predictions=members,
        ensemble_name="[ap7-fallback]",
        target=y,
        train_target=y,
        val_target=y,
        test_target=y,
        max_ensembling_level=1,
        ensembling_methods=["arithm"],
        enable_stacking_aware_gate=True,
        use_nnls_weights=False,
        require_oof_for_gate=False,
        build_votenrank_leaderboard=False,
        uncertainty_quantile=0.0,
        verbose=False,
        max_mae_relative=0.0,
        max_std_relative=0.0,
        diversity_corr_warn_threshold=2.0,
        k2_catastrophic_mae_ratio=0.0,
    )
    assert "_stacking_gate" in res
    gate = res["_stacking_gate"]
    # Diagnostic stays available even when blend opt-out.
    assert "weights" in gate
    assert gate.get("applied_to_blend") is False, gate

    stacked = np.column_stack([members[i].val_preds for i in range(3)]).T
    uniform = np.mean(stacked, axis=0)
    # ``res['arithm']`` is the train_and_evaluate_model tuple; element 0 is the SimpleNamespace
    # carrying predictions / metrics. Pre-fix probe assumed a .predictions wrapper that does not exist.
    arithm_result = res["arithm"]
    arithm_ns = arithm_result[0] if isinstance(arithm_result, tuple) else arithm_result
    actual = np.asarray(arithm_ns.val_preds, dtype=np.float64).reshape(-1)
    np.testing.assert_allclose(actual, uniform, rtol=1e-9, atol=1e-9)
