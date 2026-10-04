"""Host-side code-range screens in front of the GPU histogram kernels, and the fixed-order reduction of the fused pair-MI kernel."""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._batch_pair_mi_cuda_shared_fused import validate_pair_codes
from mlframe.feature_selection.filters.batch_mi_noise_gate_gpu import _screen_host_codes


def _cuda_ok() -> bool:
    """True when cupy sees a device with at least ~300 MB free."""
    try:
        import cupy as cp

        free, _total = cp.cuda.runtime.memGetInfo()
        return bool(free > 300 * 1024**2)
    except Exception:
        return False


def _pair_frame():
    """Three columns with cardinalities 3, 4, 2 (column 2 is not referenced by the pair under test)."""
    rng = np.random.default_rng(0)
    data = np.column_stack([rng.integers(0, 3, 50), rng.integers(0, 4, 50), rng.integers(0, 2, 50)]).astype(np.int32)
    return data, np.array([3, 4, 2], dtype=np.int32), rng.integers(0, 5, 50).astype(np.int32)


def test_validate_pair_codes_accepts_in_range_codes():
    """In-range codes and classes pass the screen."""
    data, nbins, y = _pair_frame()
    assert validate_pair_codes(data, np.array([0]), np.array([1]), nbins, y, 5) is None


@pytest.mark.parametrize("col,bad", [(0, -1), (1, 4), (0, 3)])
def test_validate_pair_codes_rejects_codes_outside_their_columns_range(col, bad):
    """A NaN sentinel (-1) or a code at/above its own column's cardinality would index outside the shared-memory histogram."""
    data, nbins, y = _pair_frame()
    data[7, col] = bad
    with pytest.raises(ValueError):
        validate_pair_codes(data, np.array([0]), np.array([1]), nbins, y, 5)


def test_validate_pair_codes_ignores_unreferenced_columns():
    """Only the columns the pairs reference are screened."""
    data, nbins, y = _pair_frame()
    data[3, 2] = 99
    assert validate_pair_codes(data, np.array([0]), np.array([1]), nbins, y, 5) is None


@pytest.mark.parametrize("bad", [-1, 5])
def test_validate_pair_codes_rejects_classes_outside_range(bad):
    """Target classes outside [0, n_classes_y) are refused."""
    data, nbins, y = _pair_frame()
    y[2] = bad
    with pytest.raises(ValueError):
        validate_pair_codes(data, np.array([0]), np.array([1]), nbins, y, 5)


def test_screen_host_codes_uses_each_columns_own_cardinality():
    """A code valid for the widest column but not for its own (mixed cardinality) is rejected; a frame within every column's range passes."""
    disc = np.array([[0, 7], [4, 19], [1, 3]], dtype=np.int16)
    y = np.array([0, 1, 1])
    freqs_y = np.array([0.3, 0.7])
    _screen_host_codes(disc, np.array([5, 20]), y, freqs_y, "t")
    disc_bad = disc.copy()
    disc_bad[1, 0] = 7
    with pytest.raises(ValueError, match="column 0"):
        _screen_host_codes(disc_bad, np.array([5, 20]), y, freqs_y, "t")


def test_screen_host_codes_rejects_negative_codes_and_bad_targets():
    """A -1 sentinel and an out-of-range target class are both refused."""
    disc = np.array([[0, 1], [1, 0]], dtype=np.int8)
    freqs_y = np.array([0.5, 0.5])
    bad_disc = disc.copy()
    bad_disc[0, 1] = -1
    with pytest.raises(ValueError):
        _screen_host_codes(bad_disc, np.array([2, 2]), np.array([0, 1]), freqs_y, "t")
    with pytest.raises(ValueError):
        _screen_host_codes(disc, np.array([2, 2]), np.array([0, 2]), freqs_y, "t")


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA device with free VRAM")
@pytest.mark.parametrize("entry", ["cupy_v1", "cupy", "cuda", "cuda_resident"])
def test_every_noise_gate_entry_point_screens_mixed_cardinality_codes(entry):
    """Column 0 holds code 7 although its own nbins is 5 (another column has nbins 20): every GPU entry point refuses it instead of bleeding into column 1."""
    from mlframe.feature_selection.filters import batch_mi_noise_gate_gpu as g

    fn = {
        "cupy_v1": g.batch_mi_with_noise_gate_cupy_v1,
        "cupy": g.batch_mi_with_noise_gate_cupy,
        "cuda": g.batch_mi_with_noise_gate_cuda,
        "cuda_resident": g.batch_mi_with_noise_gate_cuda_resident,
    }[entry]
    rng = np.random.default_rng(1)
    n = 64
    disc = np.column_stack([rng.integers(0, 5, n), rng.integers(0, 20, n)]).astype(np.int16)
    disc[3, 0] = 7
    y = rng.integers(0, 2, n).astype(np.int32)
    freqs_y = np.bincount(y, minlength=2).astype(np.float64) / n
    with pytest.raises(ValueError, match="out of range"):
        fn(disc, np.array([5, 20], dtype=np.int64), y, y.copy(), freqs_y, 4, np.uint64(1), 0.95, False, np.int32)


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA device with free VRAM")
def test_fused_pair_mi_kernel_is_bit_reproducible_and_matches_cpu():
    """Repeated launches of the fused kernel return bit-identical MI (no order-dependent atomic double accumulation) and agree with the CPU kernel."""
    from mlframe.feature_selection.filters._batch_pair_mi_cuda_shared_fused import batch_pair_mi_cuda_shared_fused
    from mlframe.feature_selection.filters.info_theory._batch_kernels import batch_pair_mi_prange

    rng = np.random.default_rng(2)
    n, n_features, n_classes = 20_000, 12, 20
    nbins = rng.integers(8, 22, n_features).astype(np.int32)
    data = np.column_stack([rng.integers(0, int(b), n) for b in nbins]).astype(np.int32)
    y = rng.integers(0, n_classes, n).astype(np.int32)
    freqs_y = np.bincount(y, minlength=n_classes).astype(np.float64) / n
    pa = rng.integers(0, n_features, 60).astype(np.int64)
    pb = ((pa + rng.integers(1, n_features, 60)) % n_features).astype(np.int64)
    first = batch_pair_mi_cuda_shared_fused(data, pa, pb, nbins, y, freqs_y)
    for _ in range(6):
        np.testing.assert_array_equal(batch_pair_mi_cuda_shared_fused(data, pa, pb, nbins, y, freqs_y), first)
    np.testing.assert_allclose(first, batch_pair_mi_prange(data, pa, pb, nbins, y, freqs_y), atol=1e-12)


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA device with free VRAM")
def test_fused_pair_mi_kernel_refuses_codes_that_would_corrupt_shared_memory():
    """A -1 sentinel in a referenced column raises before any launch instead of writing outside the shared-memory histogram."""
    from mlframe.feature_selection.filters._batch_pair_mi_cuda_shared_fused import batch_pair_mi_cuda_shared_fused

    data, nbins, y = _pair_frame()
    data[5, 1] = -1
    freqs_y = np.bincount(y, minlength=5).astype(np.float64) / len(y)
    with pytest.raises(ValueError):
        batch_pair_mi_cuda_shared_fused(data, np.array([0], dtype=np.int64), np.array([1], dtype=np.int64), nbins, y, freqs_y)
    with pytest.raises(ValueError, match="multiple of 32"):
        batch_pair_mi_cuda_shared_fused(data[:, :2], np.array([0], dtype=np.int64), np.array([1], dtype=np.int64), nbins, y, freqs_y, threads_per_block=100)
