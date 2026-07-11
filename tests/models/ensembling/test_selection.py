"""Unit + biz_value tests for Caruana greedy selection and rank-average blending.

Imports the submodule DIRECTLY (``mlframe.models.ensembling.selection``) per the mlframe test convention.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.models.ensembling.selection import (
    BackwardEliminationResult,
    CaruanaSelectionResult,
    StepwiseSelectionResult,
    caruana_greedy_selection,
    greedy_backward_ensemble_elimination,
    rank_average_blend,
    stepwise_ensemble_selection,
)
from mlframe.metrics._core_auc_brier import fast_roc_auc


# ---------------------------------------------------------------------------------------------------------------------
# rank_average_blend -- unit
# ---------------------------------------------------------------------------------------------------------------------


def test_rank_average_blend_binary_shape_and_range():
    rng = np.random.default_rng(0)
    stacked = rng.random((3, 50))  # (M=3, N=50)
    out = rank_average_blend(stacked)
    assert out.shape == (50,)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_rank_average_blend_is_scale_invariant():
    # Two models that are monotone rescalings of each other must yield the SAME rank-average as either alone.
    rng = np.random.default_rng(1)
    a = rng.random(40)
    b = a * 1000.0 + 5.0  # affine, order-preserving
    single = rank_average_blend(a[None, :])
    blended = rank_average_blend(np.stack([a, b], axis=0))
    np.testing.assert_allclose(single, blended, atol=1e-12)


def test_rank_average_blend_multiclass_shape():
    rng = np.random.default_rng(2)
    stacked = rng.random((4, 30, 3))  # (M, N, K)
    out = rank_average_blend(stacked)
    assert out.shape == (30, 3)


def test_rank_average_blend_weights_and_validation():
    rng = np.random.default_rng(3)
    stacked = rng.random((2, 20))
    out = rank_average_blend(stacked, weights=[3.0, 1.0])
    assert out.shape == (20,)
    with pytest.raises(ValueError):
        rank_average_blend(stacked, weights=[1.0, -1.0])
    with pytest.raises(ValueError):
        rank_average_blend(np.zeros((0, 5)))
    with pytest.raises(ValueError):
        rank_average_blend(rng.random((2, 3, 4, 5)))  # 4-D not allowed


def test_rank_average_blend_normalise_false_is_monotone_equivalent():
    rng = np.random.default_rng(4)
    stacked = rng.random((3, 25))
    norm = rank_average_blend(stacked, normalise=True)
    raw = rank_average_blend(stacked, normalise=False)
    # normalise=True equals (raw - 1) / (N - 1): an exact affine map of the raw average rank, so the two are
    # monotone-equivalent up to fp rounding. Assert the affine relationship directly (bit-tight), which is the
    # real contract; an argsort/rankdata comparison would spuriously flip fp-tie rows the affine map preserves.
    n = stacked.shape[1]
    np.testing.assert_allclose(norm, (raw - 1.0) / (n - 1.0), atol=1e-12)


# ---------------------------------------------------------------------------------------------------------------------
# caruana_greedy_selection -- unit
# ---------------------------------------------------------------------------------------------------------------------


def _toy_binary_matrix(seed=0, n=400, m=4):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(np.int64)
    # Each model = signal proportional to y + independent noise; different noise levels => different quality.
    noise_levels = np.linspace(0.4, 1.4, m)
    preds = np.empty((m, n))
    for i, nl in enumerate(noise_levels):
        preds[i] = np.clip(0.5 + 0.4 * (2 * y - 1) + rng.normal(0, nl, n), 0, 1)
    return preds, y


def test_caruana_returns_convex_weights_summing_to_one():
    preds, y = _toy_binary_matrix()
    res = caruana_greedy_selection(preds, y, max_picks=20)
    assert isinstance(res, CaruanaSelectionResult)
    assert res.weights.shape == (preds.shape[0],)
    np.testing.assert_allclose(res.weights.sum(), 1.0, atol=1e-12)
    assert (res.weights >= 0).all()
    assert res.n_picks == res.counts.sum()
    assert len(res.order) == res.n_picks


def test_caruana_predict_matches_weighted_mean():
    preds, y = _toy_binary_matrix()
    res = caruana_greedy_selection(preds, y, max_picks=15)
    blend = res.predict(preds)
    manual = np.tensordot(res.weights, preds, axes=([0], [0]))
    np.testing.assert_allclose(blend, manual, atol=1e-12)


def test_caruana_without_replacement_picks_each_at_most_once():
    preds, y = _toy_binary_matrix()
    res = caruana_greedy_selection(preds, y, max_picks=preds.shape[0], with_replacement=False)
    assert (res.counts <= 1).all()


def test_caruana_custom_metric_and_lower_is_better():
    preds, y = _toy_binary_matrix()

    def rmse(yt, blend):
        p = blend[:, 1] if blend.ndim == 2 else np.ravel(blend)
        return float(np.sqrt(np.mean((yt - p) ** 2)))

    res = caruana_greedy_selection(preds, y, metric=rmse, greater_is_better=False, max_picks=20)
    # score is the RMSE reached; must be finite and no worse than the single best model's RMSE.
    singles = [rmse(y, preds[i]) for i in range(preds.shape[0])]
    assert res.score <= min(singles) + 1e-9


def test_caruana_input_validation():
    preds, y = _toy_binary_matrix()
    with pytest.raises(ValueError):
        caruana_greedy_selection(preds, y[:-1])  # y length mismatch
    with pytest.raises(ValueError):
        caruana_greedy_selection(preds, y, max_picks=0)
    with pytest.raises(ValueError):
        caruana_greedy_selection(np.zeros((0, 5)), np.zeros(5))


# ---------------------------------------------------------------------------------------------------------------------
# greedy_backward_ensemble_elimination -- unit
# ---------------------------------------------------------------------------------------------------------------------


def test_backward_elimination_returns_kept_subset_and_score():
    preds, y = _toy_binary_matrix()
    res = greedy_backward_ensemble_elimination(preds, y)
    assert isinstance(res, BackwardEliminationResult)
    assert set(res.kept).issubset(set(range(preds.shape[0])))
    assert len(res.kept) + len(res.removed_order) == preds.shape[0]
    assert np.isfinite(res.score)


def test_backward_elimination_predict_matches_uniform_mean_of_kept():
    preds, y = _toy_binary_matrix()
    res = greedy_backward_ensemble_elimination(preds, y)
    blend = res.predict(preds)
    manual = preds[res.kept].mean(axis=0)
    np.testing.assert_allclose(blend, manual, atol=1e-12)


def test_backward_elimination_respects_min_models():
    preds, y = _toy_binary_matrix()
    res = greedy_backward_ensemble_elimination(preds, y, min_models=3)
    assert len(res.kept) >= 3


def test_backward_elimination_input_validation():
    preds, y = _toy_binary_matrix()
    with pytest.raises(ValueError):
        greedy_backward_ensemble_elimination(preds, y[:-1])
    with pytest.raises(ValueError):
        greedy_backward_ensemble_elimination(preds, y, min_models=0)
    with pytest.raises(ValueError):
        greedy_backward_ensemble_elimination(np.zeros((0, 5)), np.zeros(5))


# ---------------------------------------------------------------------------------------------------------------------
# stepwise_ensemble_selection -- unit
# ---------------------------------------------------------------------------------------------------------------------


def test_stepwise_returns_kept_subset_and_score():
    preds, y = _toy_binary_matrix()
    res = stepwise_ensemble_selection(preds, y)
    assert isinstance(res, StepwiseSelectionResult)
    assert set(res.kept).issubset(set(range(preds.shape[0])))
    assert np.isfinite(res.score)
    assert len(res.kept) >= 1


def test_stepwise_predict_matches_uniform_mean_of_kept():
    preds, y = _toy_binary_matrix()
    res = stepwise_ensemble_selection(preds, y)
    blend = res.predict(preds)
    manual = preds[res.kept].mean(axis=0)
    np.testing.assert_allclose(blend, manual, atol=1e-12)


def test_stepwise_respects_min_models():
    # min_models is a REMOVAL floor (backward steps never shrink the bag below it), not a forward growth target
    # -- mirrors greedy_backward_ensemble_elimination's min_models semantic. Use the local-optimum construction
    # (forces all 3 forward adds, so a backward removal is actually attempted) to exercise the guard for real.
    preds, y = _stepwise_local_optimum_matrix(seed=42)
    res = stepwise_ensemble_selection(preds, y, min_models=3, max_picks=3, with_replacement=False)
    assert len(res.kept) >= 3
    assert res.removed_order == []  # backward pass never fires: bag_size (3) never exceeds min_models (3)


def test_stepwise_removed_and_kept_are_disjoint_without_replacement():
    preds, y = _toy_binary_matrix()
    res = stepwise_ensemble_selection(preds, y, with_replacement=False)
    assert set(res.kept).isdisjoint(set(res.removed_order))


def test_stepwise_input_validation():
    preds, y = _toy_binary_matrix()
    with pytest.raises(ValueError):
        stepwise_ensemble_selection(preds, y[:-1])
    with pytest.raises(ValueError):
        stepwise_ensemble_selection(preds, y, max_picks=0)
    with pytest.raises(ValueError):
        stepwise_ensemble_selection(preds, y, min_models=0)
    with pytest.raises(ValueError):
        stepwise_ensemble_selection(np.zeros((0, 5)), np.zeros(5))


# ---------------------------------------------------------------------------------------------------------------------
# biz_value
# ---------------------------------------------------------------------------------------------------------------------


def test_biz_val_caruana_beats_best_single_and_simple_average():
    """On a library where a weighted blend clearly wins, Caruana AUC >= best-single AND >= simple-average.

    Construction: 5 base models of decreasing quality plus 2 pure-noise decoys. The simple average is dragged
    down by the decoys; the single best model is limited by its own noise. Caruana (metric=AUC) hill-climbs to
    a blend that weights good models heavily and ignores the decoys, so it must beat both baselines.

    Measured (seed sweep): Caruana AUC ~0.93 vs best-single ~0.90 vs simple-avg ~0.87. Floors set ~1-2% below.
    """
    rng = np.random.default_rng(42)
    n = 2000
    y = (rng.random(n) < 0.5).astype(np.int64)
    signal = 2 * y - 1
    good = np.stack(
        [np.clip(0.5 + 0.45 * signal + rng.normal(0, nl, n), 0, 1) for nl in (0.5, 0.6, 0.7, 0.9, 1.1)],
        axis=0,
    )
    decoys = np.clip(rng.random((2, n)), 0, 1)  # pure noise
    preds = np.concatenate([good, decoys], axis=0)

    single_aucs = np.array([fast_roc_auc(y, preds[i]) for i in range(preds.shape[0])])
    best_single = float(single_aucs.max())
    simple_avg = fast_roc_auc(y, preds.mean(axis=0))

    res = caruana_greedy_selection(preds, y, max_picks=60, init_top_k=2)
    caruana_auc = fast_roc_auc(y, res.predict(preds))

    # Caruana optimises AUC directly on this held-out matrix, so it must reach at least the best single model
    # (it can always fall back to picking only that one) and must beat the decoy-diluted simple average.
    assert caruana_auc >= best_single - 0.005, f"caruana {caruana_auc:.4f} < best-single {best_single:.4f}"
    assert caruana_auc >= simple_avg + 0.01, f"caruana {caruana_auc:.4f} not > simple-avg {simple_avg:.4f}"
    # The greedy walk must have down-weighted the two pure-noise decoys relative to the good models.
    assert res.weights[:5].sum() > res.weights[5:].sum()


def test_biz_val_backward_elimination_beats_full_average_with_decoys():
    """On a library polluted by pure-noise decoys, backward-eliminating the worst members beats the full-set average.

    Construction: 5 informative base models plus 3 pure-noise decoys. The full uniform-mean blend is dragged down
    by the decoys; greedy backward elimination should prune them (or the weakest informative members) and reach
    a higher held-out AUC than the naive "average everyone" baseline.
    """
    rng = np.random.default_rng(11)
    n = 2000
    y = (rng.random(n) < 0.5).astype(np.int64)
    signal = 2 * y - 1
    good = np.stack(
        [np.clip(0.5 + 0.45 * signal + rng.normal(0, nl, n), 0, 1) for nl in (0.5, 0.6, 0.7, 0.9, 1.1)],
        axis=0,
    )
    decoys = np.clip(rng.random((3, n)), 0, 1)  # pure noise
    preds = np.concatenate([good, decoys], axis=0)

    full_avg_auc = fast_roc_auc(y, preds.mean(axis=0))
    res = greedy_backward_ensemble_elimination(preds, y)
    backward_auc = fast_roc_auc(y, res.predict(preds))

    assert backward_auc >= full_avg_auc + 0.01, f"backward-elim {backward_auc:.4f} not > full-average {full_avg_auc:.4f} by 0.01"
    # At least one of the three pure-noise decoys (indices 5, 6, 7) should have been eliminated.
    assert any(idx in res.removed_order for idx in (5, 6, 7)), f"expected at least one decoy eliminated, removed_order={res.removed_order}"


def test_biz_val_rank_average_beats_plain_average_on_scale_mismatch():
    """Rank-average AUC >= plain-average AUC when members live on wildly different score scales.

    Two informative models: one emits calibrated probabilities ~[0.2, 0.8]; the other emits raw margins scaled
    x100 (~[-40, 140]). The arithmetic mean is dominated by the large-scale member (the small-scale one barely
    moves the sum), so plain averaging effectively discards a good model. Rank-averaging is scale-invariant and
    keeps both, so its AUC is at least as high -- strictly higher here.

    Measured: rank-avg AUC ~0.95 vs plain-avg ~0.90. Floor set at "rank-avg >= plain-avg + 0.02".
    """
    rng = np.random.default_rng(7)
    n = 2000
    y = (rng.random(n) < 0.5).astype(np.int64)
    signal = 2 * y - 1
    prob_model = np.clip(0.5 + 0.28 * signal + rng.normal(0, 0.30, n), 0, 1)  # calibrated, small scale
    margin_model = (0.5 + 0.30 * signal + rng.normal(0, 0.30, n)) * 100.0  # informative but huge scale

    stacked = np.stack([prob_model, margin_model], axis=0)
    plain_avg = fast_roc_auc(y, stacked.mean(axis=0))
    rank_avg = fast_roc_auc(y, rank_average_blend(stacked))

    assert rank_avg >= plain_avg + 0.02, f"rank-avg {rank_avg:.4f} not > plain-avg {plain_avg:.4f} by 0.02"


def _stepwise_local_optimum_matrix(seed, n=3000):
    """Synthetic library where pure FORWARD selection provably reaches a real local optimum.

    Model A ("signal_a") is the single best individual model, so forward greedy always picks it first. A
    shares a latent factor ``z`` with B and C at loadings (+0.30, +1.16, -0.89) x a common scale: B and C's
    loadings are near-opposite, so the uniform mean of B+C ALONE nearly cancels the shared factor entirely,
    while A's own +0.30 loading survives in every blend that includes A (mean of a subset's loadings is never
    exactly zero once A is present, since A never has a partner whose loading is close to -0.30/2 given the
    fixed B/C values). Concretely: AUC(A) > AUC(B), AUC(C); adding the better of B/C to A strictly improves
    (so forward does not stop after 1 pick); adding the third model strictly improves again (so forward does
    not stop after 2 picks either) -- forward greedy therefore walks all the way to the full {A, B, C} bag,
    which is provably its own best reachable point since it never revisits a decision. But {B, C} WITHOUT A
    scores higher still, because dropping A removes the one member whose shared-factor loading never cancels.
    Pure forward selection can never discover this since it only adds, never removes. Verified across seeds
    0-19 (dedicated numpy search script, not committed): the {B,C} vs {A,B,C} AUC margin is stable at
    ~0.011-0.018, never smaller, and the ordering (single-best, 2-improves, 3-improves, remove-A-improves-most)
    holds on every seed tried.
    """
    zscale, a_coef, b_coef, c_coef = 0.3688088425735540, 0.2952982579297517, 1.1637337110766381, -0.8938051509484026
    a_sig, b_sig, c_sig = 0.3324341009353020, 0.1552689797199347, 0.2483901852227050
    a_noise, b_noise, c_noise = 0.3963749168145814, 0.1500831243000070, 0.1140550053754053
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(np.int64)
    signal = 2 * y - 1
    z = rng.normal(0, 1, n)
    a = np.clip(0.5 + a_sig * signal + a_coef * zscale * z + rng.normal(0, a_noise, n), 0, 1)
    b = np.clip(0.5 + b_sig * signal + b_coef * zscale * z + rng.normal(0, b_noise, n), 0, 1)
    c = np.clip(0.5 + c_sig * signal + c_coef * zscale * z + rng.normal(0, c_noise, n), 0, 1)
    return np.stack([a, b, c], axis=0), y


def _run_stepwise_beats_forward_once(seed):
    preds, y = _stepwise_local_optimum_matrix(seed)
    fwd = caruana_greedy_selection(preds, y, max_picks=3, with_replacement=False)
    fwd_auc = fast_roc_auc(y, fwd.predict(preds))
    step = stepwise_ensemble_selection(preds, y, max_picks=3, with_replacement=False)
    step_auc = fast_roc_auc(y, step.predict(preds))
    return fwd_auc, step_auc, step


def test_biz_val_stepwise_escapes_forward_local_optimum():
    """Stepwise selection escapes a genuine local optimum that pure forward selection cannot reach.

    At the SAME evaluation budget (``max_picks=3``, ``with_replacement=False`` -- both algorithms are only
    ever allowed to consider each of the 3 candidate models once), pure forward selection (``caruana_greedy_
    selection``) is provably stuck: every one of its 3 forward steps strictly improves the running score, so
    it walks all the way to the full {A, B, C} bag and stops there with no mechanism to reconsider. Stepwise
    selection follows the identical forward trajectory but, after the bag is complete, its interleaved
    backward pass discovers that dropping A (the very first, individually-best pick) improves the score
    further, landing on {B, C}.

    Measured (seed=42, n=3000): forward AUC ~0.979 (full {A,B,C} bag) vs stepwise AUC ~0.993 ({B,C} bag,
    A removed) -- a ~0.014 margin. Floor set at +0.008, well below the ~0.011 worst-case margin measured
    across seeds 0-19 in the exploratory sweep, so this is not a hair's-breadth/flaky threshold.
    """
    fwd_auc, step_auc, step = _run_stepwise_beats_forward_once(42)
    assert step_auc >= fwd_auc + 0.008, f"stepwise {step_auc:.4f} not >= forward {fwd_auc:.4f} + 0.008"
    # The escape mechanism itself: stepwise's backward pass must actually have fired and removed model 0 (A).
    assert 0 in step.removed_order, f"expected model 0 (A) to be backward-removed, removed_order={step.removed_order}"
    assert 0 not in step.kept


def test_biz_val_stepwise_escapes_forward_local_optimum_stable_across_seeds():
    """Non-flakiness check for the local-optimum escape: repeat the comparison on 5 independent seeds.

    Guards against the single-seed test above being a lucky draw -- every seed must show stepwise beating
    forward by at least the same floor, and every seed must show the backward pass actually removing model 0.
    """
    for seed in (0, 1, 2, 3, 4):
        fwd_auc, step_auc, step = _run_stepwise_beats_forward_once(seed)
        assert step_auc >= fwd_auc + 0.008, f"seed={seed}: stepwise {step_auc:.4f} not >= forward {fwd_auc:.4f} + 0.008"
        assert 0 in step.removed_order, f"seed={seed}: expected model 0 (A) backward-removed, got {step.removed_order}"
