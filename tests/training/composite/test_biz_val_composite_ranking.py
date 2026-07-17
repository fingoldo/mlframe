"""biz_value: the composite ranker beats base-only on NDCG@k when a residual
feature explains the within-group fine ordering the base score gets wrong."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.ranking import CompositeRankEstimator, ndcg_at_k


def _ranking_data(n_groups=120, items=20, seed=0):
    """Dominant base => coarse order; residual feature => fine within-group reorder."""
    rng = np.random.default_rng(seed)
    rows, ys, groups = [], [], []
    for g in range(n_groups):
        base = rng.normal(size=items)
        resid = rng.normal(size=items)
        # The within-group ideal order needs BOTH base (coarse) and resid (fine).
        true = 2.0 * base + 1.5 * resid
        rel = true - true.min()
        rel = np.round(rel / (rel.max() + 1e-9) * 4).astype(int)
        for i in range(items):
            rows.append({"base": base[i], "resid": resid[i], "noise": rng.normal()})
            ys.append(rel[i])
            groups.append(g)
    return pd.DataFrame(rows), np.asarray(ys), np.asarray(groups)


def _split(X, y, g, frac=0.6):
    gids = np.unique(g)
    cut = int(len(gids) * frac)
    train_g, test_g = set(gids[:cut].tolist()), set(gids[cut:].tolist())
    tr = np.array([gi in train_g for gi in g])
    te = np.array([gi in test_g for gi in g])
    return (X[tr].reset_index(drop=True), y[tr], g[tr], X[te].reset_index(drop=True), y[te], g[te])


def test_biz_val_composite_ranking_beats_base_only_lambdarank():
    pytest.importorskip("lightgbm")
    X, y, g = _ranking_data(seed=11)
    Xtr, ytr, gtr, Xte, yte, gte = _split(X, y, g)
    est = CompositeRankEstimator("base")  # default LGBMRanker lambdarank
    est.fit(Xtr, ytr, gtr)
    comp = ndcg_at_k(yte, est.predict(Xte, group=gte), gte, k=10)
    base_only = ndcg_at_k(yte, Xte["base"].to_numpy(), gte, k=10)
    # Measured comp ~0.97 vs base-only ~0.91; floor the win at +0.02 (held-out groups).
    assert comp >= base_only + 0.02, f"composite {comp:.4f} did not beat base {base_only:.4f}"
    assert comp <= 1.0


def test_biz_val_composite_ranking_beats_base_only_pairwise():
    from sklearn.linear_model import LogisticRegression

    X, y, g = _ranking_data(seed=12)
    Xtr, ytr, gtr, Xte, yte, gte = _split(X, y, g)
    est = CompositeRankEstimator("base", base_estimator=LogisticRegression(max_iter=1000))
    est.fit(Xtr, ytr, gtr)
    comp = ndcg_at_k(yte, est.predict(Xte, group=gte), gte, k=10)
    base_only = ndcg_at_k(yte, Xte["base"].to_numpy(), gte, k=10)
    assert comp >= base_only + 0.01, f"pairwise composite {comp:.4f} <= base {base_only:.4f}"
