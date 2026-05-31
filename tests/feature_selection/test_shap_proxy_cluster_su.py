"""Unit tests for SU-pairwise clustering in ShapProxiedFS (iter67).

Locks:
* SU-pairwise clustering reproduces the Pearson partition on purely linear
  redundancy (sanity that the SU primitive on quantile bins is sensitive
  enough to pick up strong linear correlations at threshold=0.5).
* SU-pairwise CATCHES non-linear redundancy (sinusoidal X[:, k] = f(X[:, 0]))
  that Pearson clustering misses.
* Pipeline integration: ``MRMR -> ShapProxiedFS(precomputed=...)`` selects the
  SU backend by default and surfaces it under
  ``report['clustering']['backend'] == 'su'``; without ``precomputed`` the
  Pearson backend is selected (no regression on legacy path).
* Constant and unrelated columns become singleton clusters.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection._shap_proxy_cluster import cluster_correlated_features
from mlframe.feature_selection._shap_proxy_cluster_su import cluster_correlated_features_su


def _partition(labels):
    d = defaultdict(set)
    for i, x in enumerate(labels):
        d[int(x)].add(i)
    return {frozenset(v) for v in d.values()}


def _quantile_bin(col: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Quantile bin a column into ``[0, n_bins)`` int labels (mirrors MRMR's
    categorize_dataset for a continuous column)."""
    col = np.asarray(col, dtype=np.float64)
    if np.unique(col).size <= 1:
        return np.zeros_like(col, dtype=np.int32)
    qs = np.quantile(col, np.linspace(0, 1, n_bins + 1))
    # collapse duplicate edges; np.digitize handles arbitrary monotone edges.
    qs = np.unique(qs)
    if qs.size <= 1:
        return np.zeros_like(col, dtype=np.int32)
    edges = qs[1:-1] if qs.size > 2 else qs[1:]
    return np.clip(np.digitize(col, edges, right=False), 0, max(0, qs.size - 2)).astype(np.int32)


def _build_bins(X: np.ndarray, names: list[str], n_bins: int = 10) -> dict[str, np.ndarray]:
    return {n: _quantile_bin(X[:, i], n_bins=n_bins) for i, n in enumerate(names)}


def test_su_clustering_linear_redundancy_matches_pearson_partition():
    """Linear-redundancy regime: Pearson catches it at |corr|>=0.7, SU at the
    binned-MI scale catches it at SU>=0.3 (calibrated for the n_bins=10
    quantile binner; SU on 10x10 contingency is bounded by log(10)/log(10)=1
    but realises much smaller for noisy linear relationships)."""
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.standard_normal((n, 3))
    refl = np.hstack([z[:, [k]] + 0.15 * rng.standard_normal((n, 3)) for k in range(3)])
    noise = rng.standard_normal((n, 4))
    X = np.hstack([refl, noise])
    names = [f"f{i}" for i in range(X.shape[1])]
    bins = _build_bins(X, names, n_bins=10)

    pearson_labels = cluster_correlated_features(X, threshold=0.7, use_gpu=False)
    su_labels = cluster_correlated_features_su(bins, threshold=0.3, feature_names=names)
    p_pearson = _partition(pearson_labels)
    p_su = _partition(su_labels)
    # Both partitions must group the 3 reflection blocks identically.
    for k in range(3):
        block = frozenset(range(k * 3, (k + 1) * 3))
        assert block in p_pearson, f"Pearson missed reflection block {k}: {sorted(p_pearson)}"
        assert block in p_su, f"SU missed reflection block {k}: {sorted(p_su)}"


def test_su_clustering_catches_nonlinear_redundancy_pearson_misses():
    """Inserted non-linear redundancy: X[:, 4] = sin(2*pi*X[:, 0]).
    Pearson misses this (|corr| ~ 0 by symmetry) and X4 ends up a singleton;
    SU clustering links X0 and X4 because the binned mutual information is
    high (sin is deterministic in X0 once binned)."""
    rng = np.random.default_rng(7)
    n = 4000
    # X[:, 0]: U(-1, 1) so the period of sin(2*pi*x) lands inside the support.
    x0 = rng.uniform(-1.0, 1.0, size=n)
    x4 = np.sin(2.0 * np.pi * x0) + 0.05 * rng.standard_normal(n)
    # Other features: independent gaussians (no link to x0/x4).
    others = rng.standard_normal((n, 6))
    X = np.column_stack([x0, others[:, 0], others[:, 1], others[:, 2], x4, others[:, 3], others[:, 4], others[:, 5]])
    names = [f"f{i}" for i in range(X.shape[1])]
    bins = _build_bins(X, names, n_bins=12)

    pearson_labels = cluster_correlated_features(X, threshold=0.7, use_gpu=False)
    # Pearson should NOT link f0 and f4 (sin is anti-symmetric around 0).
    assert pearson_labels[0] != pearson_labels[4], (
        f"Pearson unexpectedly linked f0/f4 (expected miss on sin): labels={pearson_labels.tolist()}"
    )

    su_labels = cluster_correlated_features_su(
        bins, threshold=0.25, feature_names=names,
    )
    # SU SHOULD link f0 and f4.
    assert su_labels[0] == su_labels[4], (
        f"SU clustering missed the non-linear f0~f4 link: labels={su_labels.tolist()}"
    )


def test_su_clustering_constant_column_singleton():
    """Constant columns must become their own singleton (SU=0 with everyone)."""
    rng = np.random.default_rng(11)
    n = 1500
    x0 = rng.standard_normal(n)
    x1 = x0 + 0.05 * rng.standard_normal(n)
    x2 = np.full(n, 3.14)  # constant
    X = np.column_stack([x0, x1, x2])
    names = ["a", "b", "c"]
    bins = _build_bins(X, names, n_bins=10)
    labels = cluster_correlated_features_su(bins, threshold=0.3, feature_names=names)
    assert labels[0] == labels[1], "highly correlated x0~x1 not grouped"
    assert labels[2] != labels[0], "constant column unexpectedly clustered with non-constant"


def test_su_clustering_empty_bins_returns_empty():
    labels = cluster_correlated_features_su({}, threshold=0.3, feature_names=[])
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (0,)


def test_su_clustering_respects_feature_names_ordering():
    """When feature_names re-orders the bins dict, the returned labels axis
    matches that ordering rather than the dict insertion order."""
    rng = np.random.default_rng(3)
    n = 1200
    z = rng.standard_normal(n)
    bins = {
        "alpha": _quantile_bin(z + 0.1 * rng.standard_normal(n)),
        "noise": _quantile_bin(rng.standard_normal(n)),
        "beta": _quantile_bin(z + 0.1 * rng.standard_normal(n)),
    }
    # Permuted order: alpha/beta should still cluster as the first/third
    # positions because both encode z.
    labels = cluster_correlated_features_su(
        bins, threshold=0.3, feature_names=["alpha", "noise", "beta"]
    )
    assert labels[0] == labels[2], f"alpha/beta not grouped under permuted order: {labels}"
    assert labels[1] != labels[0], f"noise was grouped with z-encoders: {labels}"


def test_pipeline_mrmr_then_shap_proxied_fs_selects_su_backend():
    """ShapProxiedFS(precomputed=mrmr.export_artifacts()) -> clustering uses
    SU backend by default, surfaces backend='su' in report."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    X, y, _roles = make_regime_dataset(
        n_samples=1500,
        n_informative=5,
        n_redundant=5,
        redundancy_rho=0.8,
        n_noise=70,
        snr=8.0,
        task="binary",
        seed=0,
    )
    mrmr = MRMR(
        retain_artifacts=True,
        dcd_enable=False,
        build_friend_graph=False,
        cluster_aggregate_enable=False,
        verbose=0,
    ).fit(X, y)
    artifacts = mrmr.export_artifacts()
    assert isinstance(artifacts.get("bins"), dict) and len(artifacts["bins"]) > 0, (
        "MRMR did not export bins -- precondition for SU clustering"
    )

    common = dict(
        random_state=0,
        verbose=False,
        prefilter_top=30,
        max_features=5,
        n_models=1,
        n_splits=2,
        out_of_fold=False,
        revalidate=False,
        trust_guard=False,
        run_importance_ablation=False,
        cluster_features=True,        # force the clustering branch
        cluster_auto_threshold=10,    # tiny so 'auto' fires too
        brute_force_max_features=12,
        shap_prefilter_enabled=False,
    )

    # Backend = 'su' when precomputed provided.
    sps_su = ShapProxiedFS(precomputed=artifacts, **common).fit(X, y)
    rep_su = sps_su.shap_proxy_report_
    assert "clustering" in rep_su, "clustering block missing under SU path"
    assert rep_su["clustering"].get("backend") == "su", (
        f"expected backend='su', got {rep_su['clustering']!r}"
    )

    # Backend = 'pearson' without precomputed (no regression on legacy path).
    sps_pearson = ShapProxiedFS(**common).fit(X, y)
    rep_p = sps_pearson.shap_proxy_report_
    assert "clustering" in rep_p
    assert rep_p["clustering"].get("backend") == "pearson", (
        f"expected backend='pearson' without precomputed, got {rep_p['clustering']!r}"
    )


def test_pipeline_su_backend_opt_out():
    """``cluster_use_precomputed_bins=False`` falls back to Pearson even when
    precomputed bins are available -- gives users a deterministic opt-out."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    X, y, _ = make_regime_dataset(
        n_samples=1500,
        n_informative=5,
        n_redundant=3,
        redundancy_rho=0.8,
        n_noise=30,
        snr=8.0,
        task="binary",
        seed=1,
    )
    artifacts = MRMR(
        retain_artifacts=True,
        dcd_enable=False,
        build_friend_graph=False,
        cluster_aggregate_enable=False,
        verbose=0,
    ).fit(X, y).export_artifacts()

    sps = ShapProxiedFS(
        precomputed=artifacts,
        cluster_use_precomputed_bins=False,  # opt out of SU
        random_state=0,
        verbose=False,
        prefilter_top=20,
        max_features=5,
        n_models=1,
        n_splits=2,
        out_of_fold=False,
        revalidate=False,
        trust_guard=False,
        run_importance_ablation=False,
        cluster_features=True,
        cluster_auto_threshold=5,
        brute_force_max_features=12,
        shap_prefilter_enabled=False,
    ).fit(X, y)
    rep = sps.shap_proxy_report_
    assert rep.get("clustering", {}).get("backend") == "pearson", (
        f"opt-out did not return to pearson: {rep.get('clustering')!r}"
    )
