"""Regressions for two ShapProxiedFS parsimony leaks.

1. ``revalidate_top_n`` picked its winner inside a relative band (2% of the best honest loss) far narrower than the
   single-holdout noise of a loss difference, so on a 3-signal + 9-noise binary bed it kept 2 noise columns per seed.
   The paired 1-SE pick (``_shap_proxy_paired_parsimony``) now re-scores the winner against smaller near-best candidates
   with out-of-fold losses over all rows.
2. ``refine_mode="core"``'s honest gate only compared the core proposal to the PRE-refine loss, so core could return a
   subset worse than greedy's (breast_cancer + 120 permuted decoys, seed 1: 14 columns vs greedy's 23, lower AUC),
   contradicting its documented "never worse than greedy" contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_paired_parsimony import paired_one_se_pick


def _bed(seed=0, n=1200):
    """Search / holdout halves of a binary target driven by columns 0 and 1."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = ((X[:, 0] + X[:, 1] + 0.5 * rng.normal(size=n)) > 0).astype(int)
    half = n // 2
    return X[:half], y[:half], X[half:], y[half:]


def test_paired_pick_prefers_smaller_equivalent_subset():
    """The paired one-SE rule picks the smaller subset whose loss is statistically indistinguishable from the winner's."""
    Xs, ys, Xh, yh = _bed()
    ranked = [
        dict(features=(0, 1, 2, 3), n_members=4, stable_score=0.100),
        dict(features=(0, 1), n_members=2, stable_score=0.101),
        dict(features=(0,), n_members=1, stable_score=0.115),
    ]
    feats, info = paired_one_se_pick(
        ranked, ranked[0], LogisticRegression(), Xs, ys, Xh, yh, classification=True, metric="brier", unit_to_members=None, n_se=1.0
    )
    assert feats == (0, 1), info
    assert info["tested"][0]["accepted"] is False  # dropping x1 costs many SEs


def test_paired_pick_disabled_or_undecomposable_keeps_winner():
    """n_se=0 or a metric without a per-row decomposition keeps the original winner."""
    Xs, ys, Xh, yh = _bed()
    ranked = [dict(features=(0, 1, 2, 3), n_members=4, stable_score=0.1), dict(features=(0, 1), n_members=2, stable_score=0.1)]
    common = dict(classification=True, unit_to_members=None)
    assert paired_one_se_pick(ranked, ranked[0], LogisticRegression(), Xs, ys, Xh, yh, metric="brier", n_se=0.0, **common)[0] == (0, 1, 2, 3)
    assert paired_one_se_pick(ranked, ranked[0], LogisticRegression(), Xs, ys, Xh, yh, metric="auc", n_se=1.0, **common)[0] == (0, 1, 2, 3)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 2])
def test_spfs_binary_bed_drops_noise(seed):
    """On a binary bed with three informative and nine noise columns, the selector drops the noise."""
    pytest.importorskip("xgboost")
    from mlframe.feature_selection.registry import get

    rng = np.random.default_rng(seed)
    n = 1500
    inf = rng.normal(size=(n, 3))
    X = pd.DataFrame(np.column_stack([inf, rng.normal(size=(n, 9))]), columns=[f"inf{i}" for i in range(3)] + [f"noise{i}" for i in range(9)])
    y = pd.Series(((inf @ np.array([0.9, 0.8, -0.7]) + 0.3 * rng.normal(size=n)) > 0).astype(int))
    sel = get("ShapProxiedFS").instantiate(
        classification=True, metric="brier", optimizer="bruteforce", max_features=6, top_n=12, n_splits=3, n_revalidation_models=2,
        random_state=0, verbose=False, n_jobs=1,
    )
    sel.fit(X, y)
    got = {str(c) for c in sel.selected_features_}
    assert {"inf0", "inf1", "inf2"} <= got
    assert len([c for c in got if c.startswith("noise")]) <= 1, got


@pytest.mark.slow
def test_core_refine_never_worse_than_greedy_on_real_bed():
    """The core refine mode never ends with a worse honest loss than the greedy mode on a real dataset."""
    pytest.importorskip("shap")
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier

    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    d = load_breast_cancer()
    rng = np.random.default_rng(1)
    n, p = d.data.shape
    cols = {f"bc{j}": d.data[:, j] for j in range(p)}
    for i in range(120):
        cols[f"perm_{i}"] = d.data[rng.permutation(n), i % p]
    for i in range(6):
        cols[f"dup_{i}"] = d.data[:, i % p] + rng.normal(0.0, 0.05 * float(d.data[:, i % p].std()), size=n)
    df = pd.DataFrame(cols)

    def _fit(mode):
        """Fit with ``refine_mode=mode`` and return (number selected, honest full-set loss)."""
        sel = ShapProxiedFS(
            model=RandomForestClassifier(n_estimators=10, random_state=0), classification=True, n_splits=3, n_models=1, max_features=None,
            top_n=10, holdout_size=0.25, revalidate=False, trust_guard=False, prefilter_top=None, cluster_features=False, random_state=0,
            n_jobs=1, refine_mode=mode,
        )
        sel.fit(df, d.target)
        return len(sel.selected_features_), sel.shap_proxy_report_["within_cluster_refine"]["honest_loss_full"]

    n_auto, loss_auto = _fit("auto")
    n_greedy, loss_greedy = _fit("greedy")
    # A core proposal SMALLER than greedy's may only be taken when it costs no honest loss (pre-fix: 14 vs 23 columns, worse).
    assert n_auto >= n_greedy or loss_auto <= loss_greedy + 1e-12, (n_auto, loss_auto, n_greedy, loss_greedy)
