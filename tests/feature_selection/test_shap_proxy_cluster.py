"""Unit tests for the scalable correlated-feature clustering (the wide-data path).

Locks: union-find partition correctness, CPU/GPU partition parity, denoised-unit construction
(noise variance ~ sigma^2/k so the aggregate correlates with the latent better than any member),
and that constant/independent columns become singletons.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection._shap_proxy_cluster import (
    build_unit_matrix, cluster_correlated_features, cluster_summary)


def _partition(labels):
    from collections import defaultdict

    d = defaultdict(set)
    for i, x in enumerate(labels):
        d[int(x)].add(i)
    return {frozenset(v) for v in d.values()}


def _make_clustered(n=2000, n_factors=3, refl=4, n_noise=8, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, n_factors))
    refl_cols = np.hstack([z[:, [k]] + 0.2 * rng.normal(size=(n, refl)) for k in range(n_factors)])
    noise = rng.normal(size=(n, n_noise))
    X = np.hstack([refl_cols, noise])
    return X, z, n_factors, refl, n_noise


def test_clustering_recovers_known_partition_cpu():
    X, z, nf, refl, n_noise = _make_clustered()
    labels = cluster_correlated_features(X, threshold=0.7, use_gpu=False)
    # nf clusters of `refl` reflections + n_noise singletons
    assert labels.max() + 1 == nf + n_noise
    part = _partition(labels)
    # each factor's reflections must be one cluster
    for k in range(nf):
        block = frozenset(range(k * refl, (k + 1) * refl))
        assert block in part, f"reflections of factor {k} not grouped: {sorted(part)}"


def test_clustering_cpu_gpu_partition_parity():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no CUDA device")
    X, *_ = _make_clustered(seed=1)
    lab_cpu = cluster_correlated_features(X, threshold=0.7, use_gpu=False)
    lab_gpu = cluster_correlated_features(X, threshold=0.7, use_gpu=True)
    assert _partition(lab_cpu) == _partition(lab_gpu)


def test_blocked_path_matches_dense():
    X, *_ = _make_clustered(seed=2)
    dense = cluster_correlated_features(X, threshold=0.7, use_gpu=False, max_dense_features=10_000)
    blocked = cluster_correlated_features(X, threshold=0.7, use_gpu=False, max_dense_features=4, block=4)
    assert _partition(dense) == _partition(blocked)


def test_denoised_unit_beats_members():
    X, z, nf, refl, n_noise = _make_clustered(n=4000, seed=3)
    labels = cluster_correlated_features(X, threshold=0.7, use_gpu=False)
    units, u2m, kind = build_unit_matrix(X, labels, weighting="pca_pc1")
    summ = cluster_summary(u2m)
    assert summ["n_multi_clusters"] == nf
    assert summ["n_singletons"] == n_noise
    # the aggregate of factor 0's reflections correlates with the latent better than the best member
    cl0 = next(i for i, m in enumerate(u2m) if len(m) > 1 and 0 in m)
    agg = units[:, cl0]
    members = [X[:, c] for c in u2m[cl0]]
    best_member_corr = max(abs(np.corrcoef(m, z[:, 0])[0, 1]) for m in members)
    agg_corr = abs(np.corrcoef(agg, z[:, 0])[0, 1])
    assert agg_corr > best_member_corr, f"denoised aggregate {agg_corr:.3f} !> best member {best_member_corr:.3f}"


def test_constant_and_independent_columns_are_singletons():
    rng = np.random.default_rng(0)
    n = 1000
    X = np.column_stack([rng.normal(size=n), np.ones(n), rng.normal(size=n)])  # col1 constant
    labels = cluster_correlated_features(X, threshold=0.7, use_gpu=False)
    assert labels.max() + 1 == 3  # all singletons


def test_gpu_min_features_routes_small_f_to_cpu(monkeypatch):
    """At small ``f`` the dispatcher must skip the GPU dense path (cold cupy/CUDA load costs >10s,
    CPU GEMM costs <1s) regardless of CUDA availability. Verified by mocking the GPU edge function
    so the test does not require CUDA: when ``use_gpu='auto'`` and ``f < gpu_min_features``, the
    GPU function must NOT be invoked. The legacy explicit-on path (``use_gpu=True``) honours the
    caller and bypasses the size gate -- production callers that already paid the cupy cold-start
    can force GPU back on through that route."""
    from mlframe.feature_selection import _shap_proxy_cluster as mod

    calls = {"gpu_dense": 0}

    def _mock_gpu(*a, **kw):
        calls["gpu_dense"] += 1
        # Return None to force the dispatcher's fallback path so we don't actually need cupy.
        return None

    monkeypatch.setattr(mod, "_edges_dense_gpu", _mock_gpu)
    # Pretend cupy is importable + a GPU is visible so the `gpu = True` branch is taken before
    # the small-f gate. The gate must then flip gpu back to False at f=20 < gpu_min_features=2000.
    class _FakeRuntime:
        @staticmethod
        def getDeviceCount():
            return 1
    class _FakeCuda:
        runtime = _FakeRuntime()
    class _FakeCp:
        cuda = _FakeCuda()
    import sys
    monkeypatch.setitem(sys.modules, "cupy", _FakeCp())

    X, *_ = _make_clustered(n=400, n_factors=2, refl=3, n_noise=4, seed=4)
    assert X.shape[1] == 10  # well below the 2000 default gpu_min_features
    labels = cluster_correlated_features(X, threshold=0.7, use_gpu="auto", gpu_min_features=2000)
    assert calls["gpu_dense"] == 0, "small-f path must skip GPU dense (cold cupy is dwarfed by CPU GEMM)"
    # And the partition must still be CPU-correct (we land in the CPU dense branch).
    cpu_labels = cluster_correlated_features(X, threshold=0.7, use_gpu=False)
    assert _partition(labels) == _partition(cpu_labels)


def test_gpu_min_features_explicit_true_overrides_gate(monkeypatch):
    """When the caller passes ``use_gpu=True`` (not 'auto') the size gate must NOT fire -- the user
    has explicitly asked for GPU and may have a warm cupy process where the size threshold is wrong."""
    from mlframe.feature_selection import _shap_proxy_cluster as mod

    calls = {"gpu_dense": 0}

    def _mock_gpu(Z, threshold, edge_cap):
        calls["gpu_dense"] += 1
        # Return a valid edge tuple to satisfy the downstream union-find call.
        return np.empty(0, np.int64), np.empty(0, np.int64)

    monkeypatch.setattr(mod, "_edges_dense_gpu", _mock_gpu)
    class _FakeRuntime:
        @staticmethod
        def getDeviceCount():
            return 1
    class _FakeCuda:
        runtime = _FakeRuntime()
    class _FakeCp:
        cuda = _FakeCuda()
    import sys
    monkeypatch.setitem(sys.modules, "cupy", _FakeCp())

    X, *_ = _make_clustered(n=400, n_factors=2, refl=3, n_noise=4, seed=5)
    cluster_correlated_features(X, threshold=0.7, use_gpu=True, gpu_min_features=2000)
    assert calls["gpu_dense"] == 1, "explicit use_gpu=True must still route to GPU even at small f"
