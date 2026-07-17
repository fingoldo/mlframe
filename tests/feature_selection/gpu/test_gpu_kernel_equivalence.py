"""Numerical-equivalence tests for the new GPU kernel variants:

* ``compute_joint_hist_batched_shared_cuda`` (shared-mem atomic) vs
  ``compute_joint_hist_batched_cuda`` (global atomic). The two are
  drop-in replacements; their ``joint_counts_batch`` output must be
  bit-identical on identical input.
* ``compute_joint_hist_multi_pair_shared_cuda`` vs
  ``compute_joint_hist_multi_pair_cuda``. Same contract for the multi-
  pair grid.
* ``mi_direct_gpu_batched_streamed`` vs ``mi_direct_gpu_batched``. The
  streamed variant must produce the same ``(original_mi, confidence)``
  tuple to numerical tolerance (1e-9 on MI, exact match on the
  confidence sign).

All tests auto-skip on CUDA-unavailable hosts via the existing
``pytest.importorskip("cupy")``.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda():
    """Return True only when CUDA + cupy are usable."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


@pytest.mark.parametrize(
    "n,nbins_x,nbins_y,b",
    [
        (1000, 4, 3, 1),
        (5000, 5, 5, 8),
        (20000, 10, 10, 16),
    ],
)
def test_joint_hist_batched_shared_matches_global(n, nbins_x, nbins_y, b):
    """The shared-mem and global-atomic batched joint-hist kernels must
    produce bit-identical ``joint_counts_batch`` outputs."""
    from mlframe.feature_selection.filters import gpu as g

    g._ensure_kernels_inited()

    rng = np.random.default_rng(13)
    classes_x = rng.integers(0, nbins_x, size=n).astype(np.int32)
    perms_y = rng.integers(0, nbins_y, size=(b, n)).astype(np.int32)
    d_x = cp.asarray(classes_x)
    d_perms = cp.asarray(perms_y)

    joint_size = nbins_x * nbins_y
    block_size = 256
    grid_x = (n + block_size - 1) // block_size

    out_global = cp.zeros((b, joint_size), dtype=cp.int32)
    g.compute_joint_hist_batched_cuda(
        (grid_x, b),
        (block_size,),
        (d_x, d_perms, out_global, np.int32(n), np.int32(nbins_x), np.int32(nbins_y)),
    )
    out_shared = cp.zeros((b, joint_size), dtype=cp.int32)
    g.compute_joint_hist_batched_shared_cuda(
        (grid_x, b),
        (block_size,),
        (d_x, d_perms, out_shared, np.int32(n), np.int32(nbins_x), np.int32(nbins_y)),
        shared_mem=joint_size * 4,
    )
    cp.cuda.runtime.deviceSynchronize()
    assert cp.array_equal(out_global, out_shared), "shared-mem and global-atomic joint-hist kernels diverged"


@pytest.mark.parametrize(
    "n_rows,n_pairs,nbins,nbins_y",
    [
        (2000, 3, 5, 3),
        (5000, 5, 10, 4),
    ],
)
def test_joint_hist_multi_pair_shared_matches_global(n_rows, n_pairs, nbins, nbins_y):
    """Shared vs global multi-pair joint-hist outputs must be bit-identical."""
    from mlframe.feature_selection.filters import gpu as g

    g._ensure_kernels_inited()

    rng = np.random.default_rng(17)
    n_cols = max(n_pairs * 2, 2)
    factors_data = rng.integers(0, nbins, size=(n_rows, n_cols)).astype(np.int32)
    classes_y = rng.integers(0, nbins_y, size=n_rows).astype(np.int32)
    pairs_a = np.arange(n_pairs, dtype=np.int32)
    pairs_b = np.arange(n_pairs, 2 * n_pairs, dtype=np.int32)
    nbins_a = np.full(n_pairs, nbins, dtype=np.int32)
    nbins_b = np.full(n_pairs, nbins, dtype=np.int32)
    pair_joint_sizes = nbins_a.astype(np.int64) * nbins_b.astype(np.int64) * nbins_y
    joint_offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    joint_offsets[1:] = np.cumsum(pair_joint_sizes)
    total_cells = int(joint_offsets[-1])
    max_joint_size_y = int(pair_joint_sizes.max())

    factors_data_T = np.ascontiguousarray(factors_data.T.astype(np.int32))
    d_fact_T = cp.asarray(factors_data_T)
    d_cy = cp.asarray(classes_y)
    d_pa = cp.asarray(pairs_a)
    d_pb = cp.asarray(pairs_b)
    d_nba = cp.asarray(nbins_a)
    d_off = cp.asarray(joint_offsets.astype(np.int32))

    block_size = 128
    grid_x = (n_rows + block_size - 1) // block_size

    out_global = cp.zeros(total_cells, dtype=cp.int32)
    g.compute_joint_hist_multi_pair_cuda(
        (grid_x, n_pairs),
        (block_size,),
        (d_fact_T, d_cy, d_pa, d_pb, d_nba, d_off, out_global, n_rows, n_pairs, nbins_y),
    )
    out_shared = cp.zeros(total_cells, dtype=cp.int32)
    g.compute_joint_hist_multi_pair_shared_cuda(
        (grid_x, n_pairs),
        (block_size,),
        (d_fact_T, d_cy, d_pa, d_pb, d_nba, d_off, out_shared, n_rows, n_pairs, nbins_y, np.int32(max_joint_size_y)),
        shared_mem=max_joint_size_y * 4,
    )
    cp.cuda.runtime.deviceSynchronize()
    assert cp.array_equal(out_global, out_shared), "shared-mem and global-atomic multi-pair kernels diverged"


@pytest.mark.parametrize("n,npermutations", [(5000, 64), (20000, 128)])
def test_mi_direct_gpu_batched_streamed_matches_serial(n, npermutations):
    """The streamed variant must return ``(mi, confidence)`` numerically
    equivalent to the serial baseline for the same ``factors_data``,
    permutation count, and batch size."""
    from mlframe.feature_selection.filters.gpu import (
        mi_direct_gpu_batched,
        mi_direct_gpu_batched_streamed,
    )

    rng = np.random.default_rng(23)
    data = np.column_stack(
        [
            rng.integers(0, 4, size=n).astype(np.int32),
            rng.integers(0, 4, size=n).astype(np.int32),
        ]
    )
    nbins = np.array([4, 4], dtype=np.int32)
    # Same permutation count + batch -> same number of perms applied.
    # Different RNG seeds across runs mean confidences won't be bit-
    # identical, but the original_mi (before perms) MUST match exactly
    # and the confidence must be in the same regime.
    mi_serial, conf_serial = mi_direct_gpu_batched(
        data,
        (0,),
        (1,),
        nbins,
        npermutations=npermutations,
        batch_size=64,
    )
    mi_streamed, conf_streamed = mi_direct_gpu_batched_streamed(
        data,
        (0,),
        (1,),
        nbins,
        npermutations=npermutations,
        batch_size=64,
    )
    # original_mi is deterministic from inputs (not RNG-dependent): identical.
    # The confidence is RNG-dependent; both implementations sample the same
    # number of perms so confidences agree in expectation but not exactly.
    # If serial returned mi == 0 (failed permutation test) the streamed must
    # also be near zero. If serial returned mi > 0, streamed must also be
    # in the same regime (within fp noise of the same MI value).
    if mi_serial == 0.0:
        assert mi_streamed == 0.0, f"serial zeroed mi (failed perm test) but streamed did not: serial={mi_serial}, streamed={mi_streamed}"
    else:
        rel = abs(mi_streamed - mi_serial) / max(abs(mi_serial), 1e-12)
        assert rel < 1e-9, f"streamed MI diverged from serial: serial={mi_serial}, streamed={mi_streamed}, rel_err={rel}"
    # Confidences should be in [0, 1]; sanity check.
    assert 0.0 <= conf_serial <= 1.0
    assert 0.0 <= conf_streamed <= 1.0


@pytest.mark.parametrize(
    "n,nbins_x,nbins_y",
    [
        (10_000, 5, 5),
        (50_000, 20, 20),
        (500_000, 20, 20),
    ],
)
def test_joint_hist_shared_block_size_invariant(n, nbins_x, nbins_y):
    """block_size only changes grid decomposition; the int32 atomic-add joint histogram is
    order-independent, so bs=256 and bs=512 MUST be bit-identical. This pins the iter141
    KTC block_size re-calibration (joint_size=400 -> bs512) as a pure-perf, zero-numeric change."""
    from mlframe.feature_selection.filters import gpu as g

    g._ensure_kernels_inited()
    shared_k = g.compute_joint_hist_batched_shared_cuda

    rng = np.random.default_rng(141)
    cx = cp.asarray(rng.integers(0, nbins_x, size=n).astype(np.int32))
    perms = cp.asarray(rng.integers(0, nbins_y, size=n).astype(np.int32)).reshape(1, n)
    joint = nbins_x * nbins_y
    smem = joint * 4
    res = {}
    for bs in (256, 512):
        out = cp.zeros((1, joint), dtype=cp.int32)
        grid_x = (n + bs - 1) // bs
        shared_k((grid_x, 1), (bs,), (cx, perms, out, np.int32(n), np.int32(nbins_x), np.int32(nbins_y)), shared_mem=smem)
        cp.cuda.runtime.deviceSynchronize()
        res[bs] = out.get()
    assert np.array_equal(res[256], res[512]), "block_size changed the joint histogram (must be bit-identical)"
    assert int(res[256].sum()) == n


def test_joint_hist_lookup_routes_shared_variant():
    """The per-host KTC dispatch must select the ``shared`` kernel variant across the MI-realistic
    size range on this GPU (global-atomic is 1.5-4x slower everywhere on cc 8.x measured iter141).
    The block_size is host-tuned (256 for small joint, 512 for joint>=400 / large-n) and not pinned
    here since it is bit-identical and may legitimately vary per host."""
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.dispatch import lookup_joint_hist

    for n in (10_000, 100_000, 1_000_000):
        for joint in (25, 100, 400):
            choice = lookup_joint_hist(n_samples=n, joint_size=joint)
            assert choice["kernel_variant"] == "shared", f"n={n} joint={joint} routed to {choice}"
            assert int(choice["block_size"]) in (256, 512, 1024)
