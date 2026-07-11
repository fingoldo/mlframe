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

from mlframe.evaluation.label_correlation_rerank import detect_correlated_label_pairs, label_correlation_rerank


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
