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
from tests.conftest import skip_under_numba_disabled_jit


def _make_X(n, p, seed=0):
    """Make X."""
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
    """Make y."""
    rng = np.random.default_rng(seed + 1)
    # Binary target correlated with a couple of columns (gives MDLP real splits).
    logit = X[:, 3] - 0.5 * X[:, 4 % X.shape[1]]
    p = 1.0 / (1.0 + np.exp(-logit))
    return (rng.random(X.shape[0]) < p).astype(np.int64)


def _assert_edges_identical(a, b, label):
    """Assert edges identical."""
    assert len(a) == len(b), f"{label}: length mismatch {len(a)} vs {len(b)}"
    for j, (ea, eb) in enumerate(zip(a, b)):
        if ea is None or eb is None:
            assert ea is eb is None or (ea is None) == (eb is None), f"{label}: col {j} None mismatch"
            continue
        assert np.array_equal(ea, eb), f"{label}: col {j} edges differ\n serial={ea}\n parallel={eb}"


@pytest.mark.parametrize(
    "method,needs_y",
    [
        ("mdlp", True),
        ("freedman_diaconis", False),
        ("sturges", False),
    ],
)
@pytest.mark.parametrize("seed", [0, 7])
def test_parallel_edges_bit_identical(method, needs_y, seed):
    """Parallel edges bit identical."""
    X = _make_X(2000, 200, seed=seed)
    y = _make_y(X, seed=seed) if needs_y else None
    serial = per_feature_edges(X, y=y, method=method, n_jobs=1)
    parallel = per_feature_edges(X, y=y, method=method, n_jobs=4)
    _assert_edges_identical(serial, parallel, f"{method}/seed={seed}")


def test_low_card_and_sparse_branches_identical():
    # p just above the threshold so the threaded path actually engages.
    """Low card and sparse branches identical."""
    X = _make_X(3000, _PARALLEL_EDGES_MIN_COLS + 20, seed=3)
    y = _make_y(X, seed=3)
    serial = per_feature_edges(X, y=y, method="mdlp", n_jobs=1)
    parallel = per_feature_edges(X, y=y, method="mdlp", n_jobs=8)
    _assert_edges_identical(serial, parallel, "mdlp-branches")
    # Ensure threading actually engaged (enough miss columns).
    assert X.shape[1] >= _PARALLEL_EDGES_MIN_COLS


def test_cache_thread_safety_and_hit_behavior(tmp_path):
    """Cache thread safety and hit behavior."""
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


@skip_under_numba_disabled_jit
def test_narrow_frame_no_regression():
    # p=50 < threshold -> must use serial path, no thread overhead, identical edges.
    """Narrow frame no regression.

    Skipped under NUMBA_DISABLE_JIT=1: same class as test_speedup_mdlp -- this is fundamentally a
    wall-clock timing assertion (t_par < t_serial * 2.0 + 0.5), meaningless once the per_feature_edges
    dispatcher forces serial-only execution under disabled JIT (see per_feature_edges's own
    numba.config.DISABLE_JIT gate) regardless of n_jobs, AND n=20000/p=50 interpreted MDLP is slow
    enough to risk the workflow's per-test timeout on its own. Bit-identity correctness at this
    n_jobs>1-forced-to-serial config is already covered at smaller/faster scale by
    test_parallel_edges_bit_identical and test_low_card_and_sparse_branches_identical.
    """
    X = _make_X(20000, 50, seed=2)
    y = _make_y(X, seed=2)
    t0 = time.perf_counter()
    serial = per_feature_edges(X, y=y, method="mdlp", n_jobs=1)
    t_serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    par = per_feature_edges(X, y=y, method="mdlp", n_jobs=-1)
    t_par = time.perf_counter() - t0
    _assert_edges_identical(serial, par, "narrow")
    print(f"\n[narrow p=50] serial={t_serial:.3f}s n_jobs=-1={t_par:.3f}s (gated to serial, no regression expected)")
    # Tolerate noise: parallel path must not be dramatically slower.
    assert t_par < t_serial * 2.0 + 0.5


@skip_under_numba_disabled_jit
@pytest.mark.parametrize("p", [500, 2000])
def test_speedup_mdlp(p):
    """Speedup mdlp.

    n=8000 (down from 20000): this test has no numeric speedup floor -- it only pins bit-identity
    (the hard gate, already covered at n=2000/p=200 by test_parallel_edges_bit_identical) and PRINTS
    the measured speedup for humans running it with -s. n only needs to be large enough that per-column
    njit MDLP cost dominates thread-pool dispatch overhead so the printed ratio is still meaningful;
    verified n=8000/p=2000 still crosses that comfortably (speedup ratio unchanged in kind, only wall
    time drops) since this test runs BOTH the serial and parallel pass at full size (2x the cost of a
    single pass) and p=2000 already carries the "wide frame" scale claim on its own.

    Skipped under NUMBA_DISABLE_JIT=1: the whole premise (njit MDLP cost dominating thread-pool
    dispatch overhead) doesn't hold once njit itself is off, AND a ThreadPoolExecutor buys zero
    real parallelism over pure-Python GIL-bound code -- both the serial and n_jobs=-1 passes run at
    full uncompiled MDLP cost across up to 2000 columns at n=8000, timing out the workflow's
    3600s-per-test cap and crashing the xdist worker (confirmed live: numba-coverage-nightly run
    32616328513, shard 2/8). Bit-identity between serial/parallel is already covered at a much
    smaller n=2000/p=200 by test_parallel_edges_bit_identical, so nothing correctness-relevant is
    lost by skipping this pure speed-measurement test here.
    """
    n = 8000
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
    print(f"\n[MDLP n={n} p={p}] serial={t_serial:.3f}s parallel={t_par:.3f}s speedup={speedup:.2f}x  edges identical=True")
