"""Regression guard for the GPU SU memory gate / OOM fix (audit 2026-06-03:
shap-proxy-clustering-3).

The GPU pairwise-SU kernel allocates an einsum joint tensor (chunk, f, mb, mb)
plus float64 siblings. The old gate sized ONLY the float32 one-hot and the chunk
was hardcoded 4096 (>= f), so it tried to allocate the FULL f*f*mb^2*8 tensor
(~19 GB at f=2000/mb=10) on a 4 GB card -> OOM. The fix: the gate accounts for a
chunk=1 joint row, the kernel auto-shrinks the chunk to fit free memory, and the
einsum runs in float32 (exact integer counts) to drop the full float64 one-hot
copy.

Requires CUDA/cupy; skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su import (
    _should_route_su_gpu,
    cluster_correlated_features_su,
    cluster_su_gpu_available,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not cluster_su_gpu_available(), reason="CUDA/cupy GPU not available"),
]


def test_gate_rejects_config_whose_joint_row_cannot_fit():
    # Absurd width: even a SINGLE i-row's joint working set (10*f*mb^2*8) dwarfs
    # GPU memory. The gate MUST route to CPU (False), not green-light an OOM.
    assert (
        _should_route_su_gpu(
            n_features=200_000,
            n_samples=2_000,
            max_n_bins=64,
            gpu_min_features=0,
        )
        is False
    )


def test_gate_accepts_feasible_config():
    assert (
        _should_route_su_gpu(
            n_features=600,
            n_samples=1_500,
            max_n_bins=10,
            gpu_min_features=0,
        )
        is True
    )


def _clustered_bins(n=1500, f=600, seed=0):
    rng = np.random.default_rng(seed)
    z0 = rng.standard_normal(n)
    z1 = rng.standard_normal(n)
    cols = {}
    nb = {}
    for i in range(f):
        if i < 3:
            x = z0 + 0.15 * rng.standard_normal(n)
        elif i < 6:
            x = z1 + 0.15 * rng.standard_normal(n)
        else:
            x = rng.standard_normal(n)
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
        b = np.searchsorted(edges[1:-1], x, side="right").astype(np.int32)
        name = f"f{i}"
        cols[name] = b
        nb[name] = int(b.max()) + 1
    return cols, nb, [f"f{i}" for i in range(f)]


def test_gpu_cpu_parity_and_no_oom_on_wide_config():
    # f=600 would make the OLD hardcoded chunk=4096->600 allocate a
    # 600*600*10*10*8 *~6 ~ 3 GB tensor at once; the auto-chunk now keeps it
    # bounded. Assert the GPU path runs (no OOM) AND its partition matches CPU.
    from sklearn.metrics import adjusted_rand_score

    bins, nb, names = _clustered_bins()
    common = dict(threshold=0.3, feature_names=names, nbins_per_feature=nb, use_bitmap=False, gpu_pair_chunk_size=4096)
    labels_gpu = cluster_correlated_features_su(bins, use_gpu=True, **common)
    labels_cpu = cluster_correlated_features_su(bins, use_gpu=False, **common)
    # Single-linkage SU partitions must be identical (label-permutation invariant).
    assert adjusted_rand_score(labels_cpu, labels_gpu) == 1.0, (
        f"GPU partition != CPU partition; gpu_clusters={len(set(labels_gpu.tolist()))} cpu_clusters={len(set(labels_cpu.tolist()))}"
    )
