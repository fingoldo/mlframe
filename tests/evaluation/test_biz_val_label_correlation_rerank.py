"""biz_value test for ``evaluation.label_correlation_rerank``.

Source: 2nd_santander-product-recommendation.md -- MAP@7 simulation study shows that when two labels are
near-perfectly correlated (always co-occur), each label's own raw predicted probability is a noisier estimate
of "should this be in the top-K" than the pair's joint evidence; the source "closed gaps" between correlated
labels' predicted ranks. On a synthetic where two labels represent the SAME underlying event but the model's
per-label score estimates carry independent noise, averaging the detected pair's scores should pull their
ranks together and improve MAP@7 versus scoring them fully independently.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import label_ranking_average_precision_score

from mlframe.evaluation.label_correlation_rerank import (
    detect_correlated_label_groups,
    detect_correlated_label_pairs,
    label_correlation_rerank,
    optimize_group_blend_weight,
)


def _map_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    n = y_true.shape[0]
    average_precisions = []
    for i in range(n):
        top_k_idx = np.argsort(-scores[i])[:k]
        hits = y_true[i, top_k_idx]
        if hits.sum() == 0:
            average_precisions.append(0.0)
            continue
        cumulative_hits = np.cumsum(hits)
        precision_at_each_rank = cumulative_hits / (np.arange(k) + 1)
        average_precisions.append(float((precision_at_each_rank * hits).sum() / hits.sum()))
    return float(np.mean(average_precisions))


def _make_correlated_label_data(n: int, n_labels: int, seed: int):
    rng = np.random.default_rng(seed)
    latent = rng.random(n) < 0.3  # a single underlying event drives labels 0 and 1 identically.

    y_true = np.zeros((n, n_labels), dtype=int)
    y_true[:, 0] = latent
    y_true[:, 1] = latent
    for label in range(2, n_labels):
        y_true[:, label] = rng.random(n) < 0.15

    pred = np.zeros((n, n_labels))
    pred[:, 0] = np.clip(latent * 0.7 + rng.normal(scale=0.35, size=n), 0, 1)
    pred[:, 1] = np.clip(latent * 0.7 + rng.normal(scale=0.35, size=n), 0, 1)  # independent noise draw.
    for label in range(2, n_labels):
        pred[:, label] = np.clip(y_true[:, label] * 0.5 + rng.normal(scale=0.25, size=n), 0, 1)

    return y_true, pred


def test_biz_val_label_correlation_rerank_improves_map_at_k():
    y_true, pred = _make_correlated_label_data(n=2000, n_labels=20, seed=3)

    map_raw = _map_at_k(y_true, pred, k=7)

    pairs = detect_correlated_label_pairs(y_true, min_cooccurrence_rate=0.9, min_support=5)
    assert pairs == [(0, 1)], f"expected only the truly correlated pair to be detected, got {pairs}"

    pred_reranked = label_correlation_rerank(pred, pairs)
    map_reranked = _map_at_k(y_true, pred_reranked, k=7)

    assert map_reranked > map_raw * 1.01, f"expected correlation-aware reranking to improve MAP@7 by >=1%, got reranked={map_reranked:.4f} raw={map_raw:.4f}"


def test_detect_correlated_label_pairs_ignores_low_support_and_uncorrelated():
    rng = np.random.default_rng(0)
    n = 500
    y = np.zeros((n, 4), dtype=int)
    y[:, 0] = rng.random(n) < 0.3
    y[:, 1] = y[:, 0]  # perfectly correlated, ample support.
    y[:, 2] = rng.random(n) < 0.3  # independent -- not correlated with anything.
    y[:2, 3] = 1  # correlated with column 0 by chance but below min_support.

    pairs = detect_correlated_label_pairs(y, min_cooccurrence_rate=0.9, min_support=10)
    assert pairs == [(0, 1)]


def test_label_correlation_rerank_averages_only_flagged_pairs():
    scores = np.array([[0.2, 0.8, 0.5], [0.6, 0.4, 0.9]])
    reranked = label_correlation_rerank(scores, correlated_pairs=[(0, 1)])
    np.testing.assert_allclose(reranked[:, 0], reranked[:, 1])
    np.testing.assert_allclose(reranked[:, 2], scores[:, 2])  # untouched.


def _make_3way_group_data(n: int, n_labels: int, seed: int, group_size: int = 4):
    # labels 0..group_size-1 all mirror the SAME underlying event -- a genuine multi-way group that pairwise
    # detection alone still flags edge-by-edge, but sequential pairwise rerank corrupts (see module docstring).
    rng = np.random.default_rng(seed)
    latent = rng.random(n) < 0.3

    y_true = np.zeros((n, n_labels), dtype=int)
    for label in range(group_size):
        y_true[:, label] = latent
    for label in range(group_size, n_labels):
        y_true[:, label] = rng.random(n) < 0.15

    pred = np.zeros((n, n_labels))
    for label in range(group_size):
        pred[:, label] = np.clip(latent * 0.7 + rng.normal(scale=0.4, size=n), 0, 1)  # independent noise draw.
    for label in range(group_size, n_labels):
        pred[:, label] = np.clip(y_true[:, label] * 0.5 + rng.normal(scale=0.25, size=n), 0, 1)

    return y_true, pred


def test_biz_val_label_correlation_rerank_group_beats_sequential_pairs_on_3way_group():
    y_true, pred = _make_3way_group_data(n=3000, n_labels=20, seed=2, group_size=4)

    pairs = detect_correlated_label_pairs(y_true, min_cooccurrence_rate=0.9, min_support=5)
    assert len(pairs) == 6, f"expected all 6 pairwise edges of the 4-way group to be individually flagged, got {pairs}"

    groups = detect_correlated_label_groups(y_true, min_cooccurrence_rate=0.9, min_support=5)
    assert groups == [(0, 1, 2, 3)], f"expected pairwise edges merged into one 4-way group, got {groups}"

    map_pairs = _map_at_k(y_true, label_correlation_rerank(pred, pairs), k=7)
    map_groups = _map_at_k(y_true, label_correlation_rerank(pred, [], correlated_groups=groups), k=7)

    assert map_groups > map_pairs * 1.002, (
        f"expected true group-mean averaging to beat sequential 'last pair wins' pairwise averaging by >=0.2% MAP@7, "
        f"got groups={map_groups:.4f} pairs={map_pairs:.4f}"
    )

    # default behavior (correlated_groups omitted) stays bit-identical to the pre-extension pairwise path.
    np.testing.assert_array_equal(label_correlation_rerank(pred, pairs), label_correlation_rerank(pred, pairs, correlated_groups=None))


def _make_group_with_uninformative_member(n: int, n_labels: int, seed: int):
    # labels 0,1,2 co-occur near-perfectly (occasional independent flips keep the pairwise threshold met but
    # short of exact identity), label 0/1 have informative scores, label 2's score is pure noise unrelated to
    # its own true label -- full group averaging (w=1.0) over-dilutes 0/1's good signal with 2's garbage.
    rng = np.random.default_rng(seed)
    latent = rng.random(n) < 0.3
    flip1 = rng.random(n) < 0.06
    flip2 = rng.random(n) < 0.06

    y_true = np.zeros((n, n_labels), dtype=int)
    y_true[:, 0] = latent
    y_true[:, 1] = np.where(flip1, 1 - latent, latent)
    y_true[:, 2] = np.where(flip2, 1 - latent, latent)
    for label in range(3, n_labels):
        y_true[:, label] = rng.random(n) < 0.15

    pred = np.zeros((n, n_labels))
    pred[:, 0] = np.clip(latent * 0.9 + rng.normal(scale=0.1, size=n), 0, 1)
    pred[:, 1] = np.clip(latent * 0.9 + rng.normal(scale=0.1, size=n), 0, 1)
    pred[:, 2] = rng.random(n)  # uninformative -- independent of the true label.
    for label in range(3, n_labels):
        pred[:, label] = np.clip(y_true[:, label] * 0.5 + rng.normal(scale=0.25, size=n), 0, 1)

    return y_true, pred


def test_biz_val_label_correlation_rerank_cv_weight_beats_fixed_average_on_lrap():
    y_true, pred = _make_group_with_uninformative_member(n=4000, n_labels=15, seed=3)

    groups = detect_correlated_label_groups(y_true, min_cooccurrence_rate=0.85, min_support=5)
    assert groups == [(0, 1, 2)]

    lrap_fixed_average = label_ranking_average_precision_score(y_true, label_correlation_rerank(pred, [], correlated_groups=groups))

    cv_weights = optimize_group_blend_weight(y_true, pred, groups, random_state=3)
    lrap_cv_optimized = label_ranking_average_precision_score(
        y_true, label_correlation_rerank(pred, [], correlated_groups=groups, group_weights=cv_weights)
    )

    assert cv_weights[(0, 1, 2)] < 1.0, f"expected the CV search to prefer a partial blend over the fixed full average, got {cv_weights}"
    assert lrap_cv_optimized > lrap_fixed_average * 1.005, (
        f"expected the CV-optimized blend weight to beat the fixed simple average by >=0.5% LRAP, "
        f"got cv_optimized={lrap_cv_optimized:.4f} fixed_average={lrap_fixed_average:.4f}"
    )
