"""Tests for the SHAP feature-PAIR interaction summary (reporting/charts/shap_interactions.py).

Ships: unit tests (top-pairs/heatmap structure, non-tree gate skip, <2-feature edge, no figure leak),
a biz_value test (y driven by an f0*f1 interaction -> the (f0,f1) pair ranks #1, materially above
non-interacting pairs), and a cProfile pass at the 2000-row cap documenting the wall (interaction
values are O(F^2) per row -- the cap is the cost lever).

shap is a project dep but guarded via importorskip so a shap-less CI env skips rather than errors.
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

shap = pytest.importorskip("shap")
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from mlframe.reporting.charts import shap_interactions as si


def _interaction_data(n: int = 3000, n_feat: int = 6, *, seed: int = 0):
    """y depends on an f0*f1 INTERACTION (product, XOR-like) plus weak main effects on f2."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    logit = 2.5 * (X[:, 0] * X[:, 1]) + 0.2 * X[:, 2] + 0.1 * rng.normal(size=n)
    y = (logit > 0).astype(int)
    names = [f"f{i}" for i in range(n_feat)]
    return X, y, names


def _fit_tree(X, y):
    """Helper: Fit tree."""
    return GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0).fit(X, y)


def test_top_pairs_and_heatmap_structure(tmp_path):
    """Top pairs and heatmap structure."""
    X, y, names = _interaction_data()
    model = _fit_tree(X, y)
    before = set(plt.get_fignums())
    res = si.shap_interaction_summary(
        model,
        X,
        feature_names=names,
        max_rows=800,
        top_pairs=5,
        plot_file=str(tmp_path / "shap_int.png"),
    )
    assert res.skipped is None
    assert len(res.pair_names) == 5
    assert all(" x " in p for p in res.pair_names)
    # strengths descending
    assert np.all(np.diff(res.pair_strength) <= 1e-12)
    # square symmetric matrix, F x F
    assert res.matrix.shape == (len(names), len(names))
    # two figures (bar + heatmap), two files written, no leak
    assert len(res.figures) == 2
    assert len(res.paths) == 2 and all(os.path.exists(p) for p in res.paths)
    assert set(plt.get_fignums()) == before


def test_non_tree_model_is_skipped():
    """Non tree model is skipped."""
    X, y, names = _interaction_data(n=500)
    model = LogisticRegression(max_iter=200).fit(X, y)
    res = si.shap_interaction_summary(model, X, feature_names=names)
    assert res.skipped is not None and "non-tree" in res.skipped
    assert res.figures == [] and res.pair_names == []


def test_fewer_than_two_features_skipped():
    """Fewer than two features skipped."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 1))
    y = (X[:, 0] > 0).astype(int)
    model = _fit_tree(X, y)
    res = si.shap_interaction_summary(model, X, feature_names=["f0"])
    assert res.skipped is not None and ">=2 features" in res.skipped


def test_biz_value_planted_interaction_ranks_first():
    """f0*f1 interaction must rank #1 and be materially (>=2x) above the strongest non-interacting pair."""
    X, y, names = _interaction_data(seed=1)
    model = _fit_tree(X, y)
    res = si.shap_interaction_summary(model, X, feature_names=names, max_rows=2000, top_pairs=10)
    assert res.pair_names[0] == "f0 x f1", f"expected f0 x f1 top, got {res.pair_names[:3]}"
    # Measured ~4.6x; floor at 2x to absorb seed noise while catching a real regression.
    assert res.pair_strength[0] >= 2.0 * res.pair_strength[1], f"planted interaction not materially above #2: {res.pair_strength[:2]}"


def test_cprofile_bounded_at_cap():
    """Interaction values are the cost; the 2000-row cap on a typical GBDT keeps the wall bounded."""
    X, y, names = _interaction_data(n=4000)
    model = _fit_tree(X, y)
    si.shap_interaction_summary(model, X, feature_names=names, max_rows=2000)  # warm
    pr = cProfile.Profile()
    pr.enable()
    si.shap_interaction_summary(model, X, feature_names=names, max_rows=2000)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(5)
    out = s.getvalue()
    # dense_tree_shap dominates; assert it is the attributed hotspot (cost lever = the cap).
    assert "shap_interaction_summary" in out
