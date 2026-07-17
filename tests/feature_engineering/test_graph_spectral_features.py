"""Unit + biz_value tests for per-graph spectral descriptors (PZAD Spectral Graph Theory)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.graph_spectral_features import (
    graph_spectral_feature_names,
    graph_spectral_features,
)


def _cycle_edges(n):
    """Helper: Cycle edges."""
    return np.array([[i, (i + 1) % n] for i in range(n)], dtype=np.int64)


def _complete_edges(n):
    """Helper: Complete edges."""
    return np.array([[i, j] for i in range(n) for j in range(i + 1, n)], dtype=np.int64)


# ---------------------------------------------------------------- unit
def test_triangle_counts_and_edges():
    # a single triangle: 3 nodes, 3 edges, 1 triangle
    """Triangle counts and edges."""
    f = graph_spectral_features(3, np.array([[0, 1], [1, 2], [0, 2]]), k=3)
    assert f["n_edges"] == 3.0
    assert f["n_triangles"] == 1.0
    assert f["n_components"] == 1.0


def test_components_of_disconnected_graph():
    # two disjoint edges + one isolated node -> 3 components
    """Components of disconnected graph."""
    f = graph_spectral_features(5, np.array([[0, 1], [2, 3]]), k=3)
    assert f["n_components"] == 3.0
    assert f["algebraic_connectivity"] == 0.0  # disconnected -> Fiedler value 0


def test_algebraic_connectivity_grows_with_connectivity():
    """Algebraic connectivity grows with connectivity."""
    path = graph_spectral_features(6, np.array([[i, i + 1] for i in range(5)]), k=3)
    comp = graph_spectral_features(6, _complete_edges(6), k=3)
    assert comp["algebraic_connectivity"] > path["algebraic_connectivity"]  # complete graph is far better connected


def test_spectral_radius_bounded_by_max_degree():
    # star graph: centre degree n-1 -> spectral radius sqrt(n-1) <= max degree
    """Spectral radius bounded by max degree."""
    n = 6
    edges = np.array([[0, i] for i in range(1, n)])
    f = graph_spectral_features(n, edges, k=3)
    assert f["spectral_radius"] <= (n - 1) + 1e-9
    assert abs(f["spectral_radius"] - np.sqrt(n - 1)) < 1e-6  # known star spectral radius


def test_bipartite_signal_largest_norm_lap_eig():
    # even cycle C4 is bipartite -> largest normalized-Laplacian eigenvalue == 2
    """Bipartite signal largest norm lap eig."""
    bip = graph_spectral_features(4, _cycle_edges(4), k=3)
    assert abs(bip["largest_norm_lap_eig"] - 2.0) < 1e-6
    # odd cycle C5 is NOT bipartite -> strictly below 2
    nonbip = graph_spectral_features(5, _cycle_edges(5), k=3)
    assert nonbip["largest_norm_lap_eig"] < 2.0 - 1e-3


def test_permutation_invariance_isospectral():
    """Permutation invariance isospectral."""
    rng = np.random.default_rng(0)
    n = 8
    edges = _complete_edges(n)[rng.permutation(_complete_edges(n).shape[0])[:12]]
    perm = rng.permutation(n)
    edges2 = perm[edges]
    f1 = graph_spectral_features(n, edges, k=4)
    f2 = graph_spectral_features(n, edges2, k=4)
    for key in f1:
        assert abs(f1[key] - f2[key]) < 1e-6, f"{key} not permutation-invariant"


def test_names_and_length():
    """Names and length."""
    names = graph_spectral_feature_names(5)
    f = graph_spectral_features(4, _cycle_edges(4), k=5)
    assert set(names) == set(f.keys())
    assert len(f) == len(names)


def test_guards():
    """Guards."""
    with pytest.raises(ValueError):
        graph_spectral_features(0, np.zeros((0, 2)))
    with pytest.raises(ValueError):
        graph_spectral_features(5, np.array([[0, 9]]))  # endpoint out of range
    with pytest.raises(ValueError):
        graph_spectral_features(5000, np.zeros((0, 2)), max_nodes=2000)  # size guard


# ---------------------------------------------------------------- biz_value
def _matched_graph_of_class(cls, n, m, rng):
    """Same node count n and edge budget m for both classes so (n_nodes, n_edges) are UN-informative; only structure
    differs. Class 0: m edges placed uniformly at random. Class 1: the same m edges placed ~90% within two communities
    (a bottleneck between them). Identical size, different spectral structure."""
    half = n // 2
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if cls == 0:
        idx = rng.choice(len(all_pairs), size=min(m, len(all_pairs)), replace=False)
        edges = [all_pairs[t] for t in idx]
    else:
        within = [(i, j) for (i, j) in all_pairs if (i < half) == (j < half)]
        between = [(i, j) for (i, j) in all_pairs if (i < half) != (j < half)]
        n_within = min(round(0.9 * m), len(within))
        n_between = min(m - n_within, len(between))
        wi = rng.choice(len(within), size=n_within, replace=False)
        bi = rng.choice(len(between), size=n_between, replace=False)
        edges = [within[t] for t in wi] + [between[t] for t in bi]
    if not edges:
        edges = [(0, 1)]
    return np.array(edges, dtype=np.int64)


def test_biz_val_spectral_descriptor_classifies_graph_structure():
    """The spectral fingerprint separates graph FAMILIES that raw size features cannot: class 0 = uniform random graphs,
    class 1 = two-community graphs, generated with the SAME node count and the SAME edge budget so (n_nodes, n_edges,
    mean_degree) are uninformative. The spectral descriptor (algebraic connectivity / Fiedler value, normalized-Laplacian
    fingerprint) captures the community bottleneck and lifts CV accuracy far above the size-only baseline."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(7)
    rows, y = [], []
    for _ in range(120):
        cls = int(rng.integers(0, 2))
        n = int(rng.integers(20, 30))
        m = 2 * n  # same edge budget for both classes -> size features carry no class signal
        edges = _matched_graph_of_class(cls, n, m, rng)
        rows.append(graph_spectral_features(n, edges, k=5))
        y.append(cls)
    names = graph_spectral_feature_names(5)
    X = np.array([[r[nm] for nm in names] for r in rows])
    y = np.array(y)

    clf = RandomForestClassifier(n_estimators=120, random_state=0)
    acc_full = cross_val_score(clf, X, y, cv=5).mean()

    # size-only baseline: n_nodes, n_edges, mean_degree
    size_idx = [names.index(c) for c in ("n_nodes", "n_edges", "mean_degree")]
    acc_size = cross_val_score(clf, X[:, size_idx], y, cv=5).mean()

    assert acc_full >= 0.85, f"spectral descriptor should classify graph structure with CV acc >=0.85, got {acc_full:.3f}"
    assert acc_full >= acc_size + 0.10, f"spectral features {acc_full:.3f} should beat size-only {acc_size:.3f} by >=0.10"
