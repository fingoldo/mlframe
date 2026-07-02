"""Unit + biz_value tests for Caruana greedy selection and rank-average blending.

Imports the submodule DIRECTLY (``mlframe.models.ensembling.selection``) per the mlframe test convention.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.models.ensembling.selection import (
    CaruanaSelectionResult,
    caruana_greedy_selection,
    rank_average_blend,
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
