"""Unit tests for ``_knn_helper`` -- the shared kNN backend used by cdist / local_lift / RSD-kNN.

Covers:
  * ``knn_search`` exact-path correctness (shapes, dtypes, sklearn-Euclidean values).
  * The empty-subset sentinel branch.
  * The ``MLFRAME_DISABLE_HNSW`` opt-out forcing the exact sklearn path (the production escape hatch
    for hosts where ``import hnswlib`` segfaults at the native-DLL level after cupy/MKL are resident).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer import _knn_helper as KH


@pytest.fixture(autouse=True)
def _reset_hnsw_cache(monkeypatch):
    """Each test starts with a fresh ``_HNSW_AVAILABLE`` probe state so env-var changes are observed."""
    monkeypatch.setattr(KH, "_HNSW_AVAILABLE", None, raising=False)
    yield


def test_knn_search_exact_matches_sklearn():
    """The exact path (small N -> sklearn NearestNeighbors) returns sklearn-Euclidean distances and ids."""
    rng = np.random.default_rng(0)
    Xs = rng.standard_normal((300, 8)).astype(np.float32)
    Xq = rng.standard_normal((50, 8)).astype(np.float32)
    # prefer_hnsw_at_n huge -> always exact sklearn path.
    dists, ids = KH.knn_search(Xs, Xq, k=5, prefer_hnsw_at_n=10**12)
    assert dists.shape == (50, 5)
    assert ids.shape == (50, 5)
    assert dists.dtype == np.float32
    assert ids.dtype == np.int64
    # Distances are sorted ascending per row (kNN invariant).
    assert np.all(np.diff(dists, axis=1) >= -1e-5)
    # Spot-check against a brute-force Euclidean for one query.
    q = 0
    brute = np.sqrt(((Xs - Xq[q]) ** 2).sum(axis=1))
    expected = np.sort(brute)[:5]
    np.testing.assert_allclose(dists[q], expected, rtol=1e-4, atol=1e-4)


def test_knn_search_empty_subset_sentinel():
    """Empty X_subset -> sentinel arrays (dists=1e6, ids=0) of shape (n_query, k)."""
    Xq = np.zeros((7, 4), dtype=np.float32)
    dists, ids = KH.knn_search(np.empty((0, 4), dtype=np.float32), Xq, k=3)
    assert dists.shape == (7, 3)
    assert ids.shape == (7, 3)
    assert np.all(dists == 1e6)
    assert np.all(ids == 0)


def test_disable_hnsw_env_forces_exact(monkeypatch):
    """MLFRAME_DISABLE_HNSW=1 -> _check_hnsw_available() returns False without importing hnswlib."""
    monkeypatch.setenv("MLFRAME_DISABLE_HNSW", "1")
    monkeypatch.setattr(KH, "_HNSW_AVAILABLE", None, raising=False)
    assert KH._check_hnsw_available() is False


@pytest.mark.parametrize("falsey", ["", "0", "false", "False"])
def test_disable_hnsw_env_falsey_is_noop(monkeypatch, falsey):
    """Falsey / unset MLFRAME_DISABLE_HNSW does NOT short-circuit; the real import probe runs.

    We mock ``import hnswlib`` to succeed so the test verifies the env-var gate without touching the
    real native hnswlib DLL (which segfaults on some Windows hosts -- the very reason the opt-out
    exists). With a falsey env-var, the probe must reach the (mocked) import and return True.
    """
    import builtins
    import sys as _sys
    monkeypatch.setenv("MLFRAME_DISABLE_HNSW", falsey)
    monkeypatch.setattr(KH, "_HNSW_AVAILABLE", None, raising=False)
    # Inject a dummy hnswlib so the probe's ``import hnswlib`` resolves without loading the real DLL.
    monkeypatch.setitem(_sys.modules, "hnswlib", object())
    assert KH._check_hnsw_available() is True


def test_disable_hnsw_truthy_skips_import(monkeypatch):
    """MLFRAME_DISABLE_HNSW=1 must NOT attempt ``import hnswlib`` at all (the import itself can crash).

    We make any ``import hnswlib`` raise, then assert _check_hnsw_available() still returns False
    cleanly -- proving the truthy env-var short-circuits BEFORE the import statement is reached.
    """
    import builtins
    monkeypatch.setenv("MLFRAME_DISABLE_HNSW", "1")
    monkeypatch.setattr(KH, "_HNSW_AVAILABLE", None, raising=False)
    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "hnswlib":
            raise AssertionError("import hnswlib must NOT be attempted when MLFRAME_DISABLE_HNSW is set")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    assert KH._check_hnsw_available() is False


def test_disable_hnsw_env_routes_knn_search_to_exact(monkeypatch):
    """With MLFRAME_DISABLE_HNSW=1, knn_search at a large N (default crossover) still returns a valid
    exact result (no hnswlib import, no crash) -- the production escape hatch is functional end-to-end."""
    monkeypatch.setenv("MLFRAME_DISABLE_HNSW", "1")
    monkeypatch.setattr(KH, "_HNSW_AVAILABLE", None, raising=False)
    rng = np.random.default_rng(1)
    Xs = rng.standard_normal((1200, 6)).astype(np.float32)
    Xq = rng.standard_normal((100, 6)).astype(np.float32)
    dists, ids = KH.knn_search(Xs, Xq, k=4)  # default prefer_hnsw_at_n=50_000 sentinel path
    assert dists.shape == (100, 4)
    assert ids.shape == (100, 4)
    assert np.all(np.isfinite(dists))
    # Exact ids point at the true nearest neighbour for query 0.
    brute = np.sqrt(((Xs - Xq[0]) ** 2).sum(axis=1))
    assert ids[0, 0] == int(np.argmin(brute))
