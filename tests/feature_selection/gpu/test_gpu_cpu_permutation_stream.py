"""The permutation null is one stream shared by the CPU kernels and the GPU batched tests, so the accept/reject decision does not depend on the backend."""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._gpu_host_permutations import host_perm_batch_cap, host_permuted_y_batch
from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes

_MASK = (1 << 64) - 1


def _reference_permutation(y, seed, k):
    """Pure-Python Fisher-Yates driven by the per-permutation LCG ``parallel_mi_prange`` seeds with ``base_seed * 2654435761 + k + 1``."""
    state = (seed * 2654435761 + k + 1) & _MASK
    arr = [int(v) for v in y]
    for j in range(len(arr) - 1, 0, -1):
        state = (state * 6364136223846793005 + 1442695040888963407) & _MASK
        i = (state >> 33) % (j + 1)
        arr[j], arr[i] = arr[i], arr[j]
    return arr


def _cuda_ok() -> bool:
    """True when cupy sees a device with at least ~300 MB free."""
    try:
        import cupy as cp

        free, _total = cp.cuda.runtime.memGetInfo()
        return bool(free > 300 * 1024**2)
    except Exception:
        return False


@pytest.mark.parametrize("seed", [0, 7, 123456789])
def test_host_permutations_match_the_lcg_reference(seed):
    """Row ``b`` of the batch is the reference LCG shuffle of permutation ``first + b``."""
    y = np.random.default_rng(1).integers(0, 4, 57).astype(np.int32)
    got = host_permuted_y_batch(y, seed, 3, 5)
    assert got.dtype == np.int32 and got.shape == (5, 57)
    for b in range(5):
        assert got[b].tolist() == _reference_permutation(y, seed, 3 + b)


def test_batches_are_slices_of_one_stream():
    """Splitting the permutation range into batches yields the same rows as one batch (the GPU batch size does not change the null)."""
    y = np.random.default_rng(2).integers(0, 3, 40).astype(np.int32)
    whole = host_permuted_y_batch(y, 11, 0, 9)
    parts = np.vstack([host_permuted_y_batch(y, 11, 0, 4), host_permuted_y_batch(y, 11, 4, 5)])
    np.testing.assert_array_equal(whole, parts)


def test_unseeded_stream_is_the_mi_direct_default_seed():
    """``base_seed=None`` draws the same stream as seed 0, the ``mi_direct`` default."""
    y = np.arange(30, dtype=np.int32) % 3
    np.testing.assert_array_equal(host_permuted_y_batch(y, None, 0, 3), host_permuted_y_batch(y, 0, 0, 3))


def test_host_stream_reproduces_the_cpu_kernel_failure_count():
    """Counting exceedances over the host-generated permutations equals the CPU ``parallel_mi_prange`` count for the same seed."""
    from mlframe.feature_selection.filters._mi_prange_kernel import _parallel_mi_prange_serial
    from mlframe.feature_selection.filters.info_theory import compute_relevance_score

    rng = np.random.default_rng(4)
    n, nbins = 400, 4
    cx = rng.integers(0, nbins, n).astype(np.int32)
    cy = ((cx + rng.integers(0, 3, n)) % 3).astype(np.int32)
    fx = np.bincount(cx, minlength=nbins).astype(np.float64) / n
    fy = np.bincount(cy, minlength=3).astype(np.float64) / n
    original = float(compute_mi_from_classes(cx, fx, cy, fy))
    for seed in (0, 5, 99):
        cpu_failed, _ = _parallel_mi_prange_serial(cx, fx, cy, fy, 48, original, np.uint64(seed), np.int32, False)
        rows = host_permuted_y_batch(cy, seed, 0, 48)
        host_failed = sum(compute_relevance_score(False, cx, fx, rows[k], fy, dtype=np.int32) >= original for k in range(48))
        assert host_failed == cpu_failed


def test_host_perm_batch_cap_bounds_staging_memory():
    """The host staging cap is the budget over the row bytes, never below one permutation."""
    assert host_perm_batch_cap(1_000, budget_bytes=4_000 * 10) == 10
    assert host_perm_batch_cap(10**9, budget_bytes=1) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA device with free VRAM")
def test_gpu_batched_decision_matches_cpu_decision_near_the_threshold():
    """On weakly dependent data (p-value near the 0.05 gate) the GPU batched test and the CPU test agree on every seed, with or without the VRAM cushion."""
    from mlframe.feature_selection.filters import _gpu_batched
    from mlframe.feature_selection.filters.permutation import mi_direct

    mismatches = []
    n = 300
    for seed in range(24):
        rng = np.random.default_rng(1000 + seed)
        x = rng.integers(0, 4, n)
        y = np.where(rng.random(n) < 0.22, x % 2, rng.integers(0, 2, n))
        data = np.column_stack([x, y]).astype(np.int32)
        nbins = np.array([4, 2], dtype=np.int32)
        gpu_mi, _ = _gpu_batched.mi_direct_gpu_batched(
            data, (0,), (1,), nbins, npermutations=64, batch_size=16, min_nonzero_confidence=0.95, base_seed=seed,
        )
        cpu_mi, _ = mi_direct(
            data, x=(0,), y=(1,), factors_nbins=nbins, npermutations=64, min_nonzero_confidence=0.95, prefer_gpu=False, base_seed=seed,
        )
        if (gpu_mi > 0) != (cpu_mi > 0):
            mismatches.append(seed)
    assert mismatches == []


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA device with free VRAM")
def test_gpu_result_does_not_depend_on_free_vram(monkeypatch):
    """Forcing the VRAM cushion check to refuse (CPU fallback) gives the same accept/reject decision as the GPU path for the same seed."""
    from mlframe.feature_selection.filters import _fe_gpu_vram, _gpu_batched

    n = 300
    decisions = {}
    for refuse in (False, True):
        monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda *_a, _r=refuse, **_k: not _r)
        row = []
        for seed in range(16):
            rng = np.random.default_rng(2000 + seed)
            x = rng.integers(0, 4, n)
            y = np.where(rng.random(n) < 0.22, x % 2, rng.integers(0, 2, n))
            data = np.column_stack([x, y]).astype(np.int32)
            mi, _ = _gpu_batched.mi_direct_gpu_batched(
                data, (0,), (1,), np.array([4, 2], dtype=np.int32), npermutations=64, min_nonzero_confidence=0.95, base_seed=seed,
            )
            row.append(bool(mi > 0))
        decisions[refuse] = row
    assert decisions[False] == decisions[True]
