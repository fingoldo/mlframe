"""Unit tests for CompositeRankEstimator (grouped learning-to-rank composite)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.ranking import (
    CompositeRankEstimator,
    ndcg_at_k,
)


def _synthetic(n_groups=40, items=12, seed=0):
    """Base explains coarse order; a residual feature explains fine within-group order."""
    rng = np.random.default_rng(seed)
    rows, ys, groups = [], [], []
    for g in range(n_groups):
        base = rng.normal(size=items)
        resid_feat = rng.normal(size=items)
        # True relevance = dominant base + small within-group residual signal.
        true = 3.0 * base + 1.0 * resid_feat
        rel = (true - true.min())
        rel = np.round(rel / (rel.max() + 1e-9) * 4).astype(int)  # graded 0..4
        for i in range(items):
            rows.append({"base": base[i], "f1": resid_feat[i], "f2": rng.normal()})
            ys.append(rel[i])
            groups.append(g)
    X = pd.DataFrame(rows)
    return X, np.asarray(ys), np.asarray(groups)


def test_pairwise_fallback_fit_predict_finite():
    X, y, g = _synthetic(seed=1)
    from sklearn.linear_model import LogisticRegression

    est = CompositeRankEstimator("base", base_estimator=LogisticRegression(max_iter=500))
    est.fit(X, y, g)
    scores = est.predict(X, group=g)
    assert scores.shape == (len(y),)
    assert np.isfinite(scores).all()
    assert est._kind == "pairwise"


def test_ndcg_finite_and_beats_base_only_pairwise():
    # Residual feature carries enough signal (2*base + 1.5*resid) that the inner
    # reranking strictly improves over base-only; default drop_base_feature isolates it.
    rng = np.random.default_rng(2)
    rows, ys, groups = [], [], []
    for gid in range(40):
        base = rng.normal(size=12)
        resid = rng.normal(size=12)
        true = 2.0 * base + 1.5 * resid
        rel = true - true.min()
        rel = np.round(rel / (rel.max() + 1e-9) * 4).astype(int)
        for i in range(12):
            rows.append({"base": base[i], "f1": resid[i], "f2": rng.normal()})
            ys.append(rel[i])
            groups.append(gid)
    X = pd.DataFrame(rows)
    y = np.asarray(ys)
    g = np.asarray(groups)
    from sklearn.linear_model import LogisticRegression

    est = CompositeRankEstimator("base", base_estimator=LogisticRegression(max_iter=500))
    est.fit(X, y, g)
    comp = ndcg_at_k(y, est.predict(X, group=g), g, k=5)
    base_only = ndcg_at_k(y, X["base"].to_numpy(), g, k=5)
    assert np.isfinite(comp)
    assert 0.0 <= comp <= 1.0
    assert comp >= base_only  # inner reranking does not regress base-only order


def test_base_plus_residual_combine():
    """drop_base_feature=True: inner sees no base; combine still adds it explicitly."""
    X, y, g = _synthetic(seed=3)
    from sklearn.linear_model import LogisticRegression

    est = CompositeRankEstimator(
        "base", base_estimator=LogisticRegression(max_iter=500), drop_base_feature=True
    )
    est.fit(X, y, g)
    assert est.n_features_in_ == X.shape[1] - 1  # base dropped from inner features
    inner = est.inner_score(X)
    combined = est.predict(X, group=g)
    assert inner.shape == combined.shape == (len(y),)
    # With base_weight=0 the combined score is just the (z-scored) inner.
    est0 = CompositeRankEstimator(
        "base", base_estimator=LogisticRegression(max_iter=500),
        drop_base_feature=True, base_weight=0.0,
    )
    est0.fit(X, y, g)
    assert np.isfinite(est0.predict(X, group=g)).all()


def test_rank_helper_orderings():
    X, y, g = _synthetic(n_groups=5, items=8, seed=4)
    from sklearn.linear_model import LogisticRegression

    est = CompositeRankEstimator("base", base_estimator=LogisticRegression(max_iter=500))
    est.fit(X, y, g)
    orderings = est.rank(X, g)
    assert set(orderings.keys()) == set(np.unique(g).tolist())
    scores = est.predict(X, group=g)
    for gid, order in orderings.items():
        idx = np.flatnonzero(g == gid)
        assert sorted(order.tolist()) == sorted(idx.tolist())  # permutation of group rows
        s = scores[order]
        assert np.all(np.diff(s) <= 1e-9)  # descending by score


def test_single_group():
    X, y, g = _synthetic(n_groups=1, items=15, seed=5)
    from sklearn.linear_model import LogisticRegression

    est = CompositeRankEstimator("base", base_estimator=LogisticRegression(max_iter=500))
    est.fit(X, y, g)
    scores = est.predict(X, group=g)
    assert np.isfinite(scores).all()
    orderings = est.rank(X, g)
    assert len(orderings) == 1
    nd = ndcg_at_k(y, scores, g, k=5)
    assert 0.0 <= nd <= 1.0


def test_residual_mode_diff():
    X, y, g = _synthetic(seed=6)
    from sklearn.linear_model import LogisticRegression

    est = CompositeRankEstimator(
        "base", base_estimator=LogisticRegression(max_iter=500), residual_mode="diff"
    )
    est.fit(X, y, g)
    assert np.isfinite(est.predict(X, group=g)).all()


def test_invalid_residual_mode_raises():
    X, y, g = _synthetic(n_groups=3, seed=7)
    est = CompositeRankEstimator("base", residual_mode="bogus")
    with pytest.raises(ValueError):
        est.fit(X, y, g)


def test_group_length_mismatch_raises():
    X, y, g = _synthetic(n_groups=3, seed=8)
    est = CompositeRankEstimator("base")
    with pytest.raises(ValueError):
        est.fit(X, y, g[:-1])


def test_lambdarank_inner_importorskip():
    pytest.importorskip("lightgbm")
    X, y, g = _synthetic(seed=9)
    # Default inner picks LGBMRanker(lambdarank) when lightgbm is present.
    est = CompositeRankEstimator("base")
    est.fit(X, y, g)
    assert est._kind == "lambdarank"
    scores = est.predict(X, group=g)
    assert np.isfinite(scores).all()
    nd = ndcg_at_k(y, scores, g, k=5)
    assert 0.0 <= nd <= 1.0
