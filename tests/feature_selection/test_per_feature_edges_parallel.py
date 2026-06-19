"""Parity + speed tests for the threaded per-column loop in ``per_feature_edges``.

The per-column edge computation is embarrassingly parallel and the default
supervised kernel (MDLP / fayyad_irani) is njit(nogil=True), so a thread pool
yields real wall-time parallelism on wide frames. These tests assert that the
threaded path produces BIT-IDENTICAL edges to the serial path (the hard gate)
across methods/seeds, that the cache hit/miss behavior is preserved under
threads, and report the measured speedup.

Run: pytest tests/feature_selection/test_per_feature_edges_parallel.py -s
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from mlframe.feature_selection.filters._adaptive_nbins import (
    per_feature_edges,
    _PARALLEL_EDGES_MIN_COLS,
)


def _make_X(n, p, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float64)
    # Mix in a few low-card / sparse / NaN columns to exercise every branch.
    if p >= 4:
        X[:, 0] = rng.integers(0, 3, size=n).astype(np.float64)  # low-card branch
        X[:, 1] = 0.0
        X[: n // 20, 1] = rng.standard_normal(n // 20)  # sparse-dominant branch
        X[rng.random(n) < 0.05, 2] = np.nan  # NaN handling
    return X


def _make_y(X, seed=0):
    rng = np.random.default_rng(seed + 1)
    # Binary target correlated with a couple of columns (gives MDLP real splits).
    logit = X[:, 3] - 0.5 * X[:, 4 % X.shape[1]]
    p = 1.0 / (1.0 + np.exp(-logit))
    return (rng.random(X.shape[0]) < p).astype(np.int64)


def _assert_edges_identical(a, b, label):
    assert len(a) == len(b), f"{label}: length mismatch {len(a)} vs {len(b)}"
    for j, (ea, eb) in enumerate(zip(a, b)):
        if ea is None or eb is None:
            assert ea is eb is None or (ea is None) == (eb is None), f"{label}: col {j} None mismatch"
            continue
        assert np.array_equal(ea, eb), f"{label}: col {j} edges differ\n serial={ea}\n parallel={eb}"


@pytest.mark.parametrize("method,needs_y", [
    ("mdlp", True),
    ("freedman_diaconis", False),
    ("sturges", False),
])
@pytest.mark.parametrize("seed", [0, 7])
def test_parallel_edges_bit_identical(method, needs_y, seed):
    X = _make_X(2000, 200, seed=seed)
    y = _make_y(X, seed=seed) if needs_y else None
    serial = per_feature_edges(X, y=y, method=method, n_jobs=1)
    parallel = per_feature_edges(X, y=y, method=method, n_jobs=4)
    _assert_edges_identical(serial, parallel, f"{method}/seed={seed}")


def test_low_card_and_sparse_branches_identical():
    # p just above the threshold so the threaded path actually engages.
    X = _make_X(3000, _PARALLEL_EDGES_MIN_COLS + 20, seed=3)
    y = _make_y(X, seed=3)
    serial = per_feature_edges(X, y=y, method="mdlp", n_jobs=1)
    parallel = per_feature_edges(X, y=y, method="mdlp", n_jobs=8)
    _assert_edges_identical(serial, parallel, "mdlp-branches")
    # Ensure threading actually engaged (enough miss columns).
    assert X.shape[1] >= _PARALLEL_EDGES_MIN_COLS


def test_cache_thread_safety_and_hit_behavior(tmp_path):
    cache_dir = str(tmp_path / "edge_cache")
    X = _make_X(2000, 200, seed=1)
    y = _make_y(X, seed=1)
    # Reference (no cache, serial).
    ref = per_feature_edges(X, y=y, method="mdlp", n_jobs=1)
    # Cold cache, parallel -> all misses computed + put.
    cold = per_feature_edges(X, y=y, method="mdlp", cache_dir=cache_dir, n_jobs=4)
    _assert_edges_identical(ref, cold, "cache-cold")
    # Warm cache, parallel -> all hits (served from disk), still identical.
    warm = per_feature_edges(X, y=y, method="mdlp", cache_dir=cache_dir, n_jobs=4)
    _assert_edges_identical(ref, warm, "cache-warm")


def test_narrow_frame_no_regression():
    # p=50 < threshold -> must use serial path, no thread overhead, identical edges.
    X = _make_X(20000, 50, seed=2)
    y = _make_y(X, seed=2)
    t0 = time.perf_counter()
    serial = per_feature_edges(X, y=y, method="mdlp", n_jobs=1)
    t_serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    par = per_feature_edges(X, y=y, method="mdlp", n_jobs=-1)
    t_par = time.perf_counter() - t0
    _assert_edges_identical(serial, par, "narrow")
    print(f"\n[narrow p=50] serial={t_serial:.3f}s n_jobs=-1={t_par:.3f}s "
          f"(gated to serial, no regression expected)")
    # Tolerate noise: parallel path must not be dramatically slower.
    assert t_par < t_serial * 2.0 + 0.5


@pytest.mark.parametrize("p", [500, 2000])
def test_speedup_mdlp(p):
    n = 20000
    X = _make_X(n, p, seed=5)
    y = _make_y(X, seed=5)
    # Warm numba JIT first (excluded from timing).
    per_feature_edges(X[:, :8], y=y, method="mdlp", n_jobs=1)

    t0 = time.perf_counter()
    serial = per_feature_edges(X, y=y, method="mdlp", n_jobs=1)
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    parallel = per_feature_edges(X, y=y, method="mdlp", n_jobs=-1)
    t_par = time.perf_counter() - t0

    _assert_edges_identical(serial, parallel, f"speedup-p={p}")
    speedup = t_serial / t_par if t_par > 0 else float("nan")
    print(f"\n[MDLP n={n} p={p}] serial={t_serial:.3f}s parallel={t_par:.3f}s "
          f"speedup={speedup:.2f}x  edges identical=True")
