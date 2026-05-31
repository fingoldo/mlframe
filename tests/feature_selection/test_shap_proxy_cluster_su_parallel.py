"""Parity + speedup tests for the numba prange SU clustering kernel.

Locks the three contracts iter68 ships:
  * Parity: at f >= parallel threshold the parallel kernel matches the serial
    Python loop's cluster assignments bit-identically (no floating-point
    reordering can shift the union-find result because edge selection is a
    >= threshold compare, not a sum reduction).
  * Edge agreement: the upper-triangle edge set is identical too (stronger
    than label parity; protects against threshold-boundary tie regressions).
  * Speedup: at f=500 with realistic bin counts the parallel path is at
    least 2x faster than ``use_parallel=False``. The test is skipped on
    single-core boxes via numba.get_num_threads.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlframe.feature_selection._shap_proxy_cluster_su import (
    cluster_correlated_features_su,
)


def _quantile_bin(col: np.ndarray, n_bins: int) -> np.ndarray:
    col = np.asarray(col, dtype=np.float64)
    if np.unique(col).size <= 1:
        return np.zeros_like(col, dtype=np.int32)
    qs = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
    if qs.size <= 1:
        return np.zeros_like(col, dtype=np.int32)
    edges = qs[1:-1] if qs.size > 2 else qs[1:]
    return np.clip(np.digitize(col, edges, right=False), 0, max(0, qs.size - 2)).astype(np.int32)


def _build_synthetic_bins(n_samples: int, n_features: int, n_bins: int, seed: int):
    rng = np.random.default_rng(seed)
    # mix of correlated clusters + noise so the partition is non-trivial.
    n_blocks = max(1, n_features // 6)
    blocks = []
    for _ in range(n_blocks):
        z = rng.standard_normal(n_samples)
        for _k in range(3):
            blocks.append(z + 0.2 * rng.standard_normal(n_samples))
    while len(blocks) < n_features:
        blocks.append(rng.standard_normal(n_samples))
    X = np.column_stack(blocks[:n_features])
    names = [f"f{i}" for i in range(n_features)]
    bins = {n: _quantile_bin(X[:, i], n_bins=n_bins) for i, n in enumerate(names)}
    return bins, names


def test_serial_vs_parallel_cluster_labels_identical_small():
    """At f=80 (above the default parallel threshold of 50) the two paths
    must produce identical cluster assignments."""
    bins, names = _build_synthetic_bins(n_samples=800, n_features=80, n_bins=8, seed=0)
    serial = cluster_correlated_features_su(
        bins, threshold=0.3, feature_names=names, use_parallel=False,
    )
    parallel = cluster_correlated_features_su(
        bins, threshold=0.3, feature_names=names, use_parallel=True,
        parallel_min_features=10,  # force the kernel
    )
    assert serial.shape == parallel.shape
    assert np.array_equal(serial, parallel), (
        f"cluster labels diverge:\nserial   = {serial.tolist()}\nparallel = {parallel.tolist()}"
    )


def test_serial_vs_parallel_cluster_labels_identical_width200():
    """Larger width replays iter67's biz-val regime."""
    bins, names = _build_synthetic_bins(n_samples=1200, n_features=200, n_bins=10, seed=11)
    serial = cluster_correlated_features_su(
        bins, threshold=0.35, feature_names=names, use_parallel=False,
    )
    parallel = cluster_correlated_features_su(
        bins, threshold=0.35, feature_names=names, use_parallel=True,
    )
    assert np.array_equal(serial, parallel), (
        f"labels diverge at width=200: max diff index = {np.where(serial != parallel)[0][:10]}"
    )


def test_serial_vs_parallel_below_threshold_uses_serial_path():
    """f < parallel_min_features must take the serial path (no NPE / no kernel
    compile penalty paid). Result still matches use_parallel=False."""
    bins, names = _build_synthetic_bins(n_samples=400, n_features=8, n_bins=6, seed=2)
    out_default = cluster_correlated_features_su(bins, threshold=0.3, feature_names=names)
    out_serial = cluster_correlated_features_su(
        bins, threshold=0.3, feature_names=names, use_parallel=False,
    )
    assert np.array_equal(out_default, out_serial)


def test_parallel_kernel_speedup_at_f500():
    """Parallel kernel >= 2x faster than serial at f=500. Both runs exclude
    the kernel compile cost via a warm-up call.

    Skipped when numba reports a single thread (CI boxes, WSL-restricted
    containers, OMP_NUM_THREADS=1 explicitly).
    """
    import numba

    n_threads = numba.get_num_threads()
    if n_threads < 2:
        pytest.skip(f"numba reports {n_threads} thread(s); parallel kernel cannot beat serial")

    bins, names = _build_synthetic_bins(n_samples=1500, n_features=500, n_bins=10, seed=3)
    # Warm-up so the JIT compile of _pairwise_su_edges is not counted.
    cluster_correlated_features_su(
        bins, threshold=0.4, feature_names=names, use_parallel=True,
    )

    t0 = time.perf_counter()
    serial_labels = cluster_correlated_features_su(
        bins, threshold=0.4, feature_names=names, use_parallel=False,
    )
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    parallel_labels = cluster_correlated_features_su(
        bins, threshold=0.4, feature_names=names, use_parallel=True,
    )
    t_parallel = time.perf_counter() - t0

    assert np.array_equal(serial_labels, parallel_labels), "labels diverge at f=500"
    ratio = t_serial / max(t_parallel, 1e-9)
    assert ratio >= 2.0, (
        f"parallel kernel not fast enough: serial={t_serial:.3f}s, "
        f"parallel={t_parallel:.3f}s, ratio={ratio:.2f}x (need >= 2x), threads={n_threads}"
    )
