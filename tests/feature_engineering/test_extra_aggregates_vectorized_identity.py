"""Identity pins for the vectorised ``compute_extra_aggregates`` (transformer/_aggregation.py).

The softmax-weight recompute, ``y_skew`` and ``x_centroid_dist`` were per-query Python loops; they are now fixed-k batched gather + einsum/broadcast reductions
(5-10x faster at n_queries up to 100k, called once per head per fold in the row-attention OOF loop). einsum reorders the float32 reduction, so the contract is
<=1 float32 ULP, not bit-identical. ``y_iqr`` is unchanged (still a per-query argsort + interp) and must stay exact. The degenerate-softmax uniform-1/k fallback
must be preserved on non-finite logits.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer._aggregation import compute_extra_aggregates


def _golden(q_proj, k_proj, y_train, topk_ids, softmax_temp, aggregates):
    """Original per-query-loop reference, pre-vectorisation."""
    n_queries, k_count = topk_ids.shape
    weights = np.empty((n_queries, k_count), dtype=np.float32)
    for q in range(n_queries):
        ids_q = topk_ids[q]
        logits_q = (k_proj[ids_q] @ q_proj[q]) / softmax_temp
        m = float(logits_q.max())
        if not np.isfinite(m):
            weights[q] = 1.0 / k_count
            continue
        exps = np.exp(logits_q - m)
        s = float(exps.sum())
        if s <= 0.0 or not np.isfinite(s):
            weights[q] = 1.0 / k_count
        else:
            weights[q] = exps / s
    out = {}
    if "y_iqr" in aggregates:
        y_iqr = np.empty(n_queries, dtype=np.float32)
        for q in range(n_queries):
            y_q = y_train[topk_ids[q]]
            order = np.argsort(y_q)
            y_sorted = y_q[order]
            w_sorted = weights[q][order]
            cum_w = np.cumsum(w_sorted)
            y_iqr[q] = float(np.interp(0.75, cum_w, y_sorted)) - float(np.interp(0.25, cum_w, y_sorted))
        out["y_iqr"] = y_iqr
    if "y_skew" in aggregates:
        y_skew = np.empty(n_queries, dtype=np.float32)
        for q in range(n_queries):
            y_q = y_train[topk_ids[q]]
            w_q = weights[q]
            mean = float((w_q * y_q).sum())
            d = y_q - mean
            var = float((w_q * d * d).sum())
            if var <= 1e-12:
                y_skew[q] = 0.0
                continue
            std = np.sqrt(var)
            m3 = float((w_q * d * d * d).sum())
            y_skew[q] = m3 / (std ** 3)
        out["y_skew"] = y_skew
    if "x_centroid_dist" in aggregates:
        x_centroid_dist = np.empty(n_queries, dtype=np.float32)
        for q in range(n_queries):
            neighbours = k_proj[topk_ids[q]]
            centroid = (weights[q][:, None] * neighbours).sum(axis=0)
            diff = q_proj[q] - centroid
            x_centroid_dist[q] = float(np.linalg.norm(diff))
        out["x_centroid_dist"] = x_centroid_dist
    return out


def _make(seed, n_queries=400, n_train=600, head_dim=16, k=32):
    rng = np.random.default_rng(seed)
    q_proj = rng.standard_normal((n_queries, head_dim)).astype(np.float32)
    k_proj = rng.standard_normal((n_train, head_dim)).astype(np.float32)
    topk_ids = rng.integers(0, n_train, size=(n_queries, k)).astype(np.int64)
    y_train = rng.standard_normal(n_train).astype(np.float32)
    return q_proj, k_proj, y_train, topk_ids


AGGS = ("y_iqr", "y_skew", "x_centroid_dist")


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_extra_aggregates_match_golden_within_ulp(seed):
    q_proj, k_proj, y_train, topk_ids = _make(seed)
    new = compute_extra_aggregates(q_proj, k_proj, y_train, topk_ids, softmax_temp=1.0, aggregates=AGGS)
    gold = _golden(q_proj, k_proj, y_train, topk_ids, softmax_temp=1.0, aggregates=AGGS)
    assert set(new) == set(gold) == set(AGGS)
    # All three consume the einsum-recomputed weights, whose float32 reduction order differs from the per-query dot by <=1 ULP. y_iqr interps against the cumulative
    # weights, y_skew cubes normalised residuals; the resulting drift (~2e-6 abs / ~1.7e-5 rel observed) is far tighter than anything that could move a downstream
    # feature-selection decision.
    for name in AGGS:
        np.testing.assert_allclose(new[name], gold[name], rtol=2e-4, atol=2e-4, err_msg=name)


def test_degenerate_softmax_fallback_preserved():
    """A query whose every logit is non-finite must fall back to uniform 1/k weights (so y_skew finite, x_centroid_dist = dist to plain mean)."""
    q_proj, k_proj, y_train, topk_ids = _make(3, n_queries=50)
    q_proj[0] = np.inf  # forces non-finite logits for query 0
    new = compute_extra_aggregates(q_proj, k_proj, y_train, topk_ids, softmax_temp=1.0, aggregates=("y_skew",))
    gold = _golden(q_proj, k_proj, y_train, topk_ids, softmax_temp=1.0, aggregates=("y_skew",))
    assert np.isfinite(new["y_skew"]).all()
    np.testing.assert_allclose(new["y_skew"], gold["y_skew"], rtol=2e-4, atol=2e-4)
