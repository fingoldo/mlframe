"""The device walk of the bootstrap rank-MI equals the host walk (to double rounding) for distinct columns and leaves tied and non-finite columns to the host."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _bootstrap_rank_mi_njit as host
from tests._cuda_require import require_cuda_device


def _data(n, k, n_classes=2, seed=0):
    """Distinct columns, one with ties, one with a NaN, and the resamples."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(n, k)).astype(np.float32)
    arr[:, 1] = np.round(arr[:, 1] * 4)
    arr[3, 2] = np.nan
    y = rng.integers(0, n_classes, n).astype(np.int64)
    m = max(2, round(0.8 * n))
    reps = [np.sort(rng.integers(0, n, m)).astype(np.int64) for _ in range(6)]
    return arr, y, reps


@pytest.mark.parametrize("n,k,n_classes", [(129, 5, 2), (1009, 9, 3), (6000, 12, 2), (6000, 4, 5)])
def test_device_equals_host(monkeypatch, n, k, n_classes):
    """The scored-column mask is identical in both modes and the MI matrix equal to double rounding."""
    pytest.importorskip("cupy")
    arr, y, reps = _data(n, k, n_classes)
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_RANK_DEVICE", "0")
    host._PREP_CACHE.clear()
    want_mi, want_ok = host.bootstrap_rank_mi(arr, y, reps, 10)
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_RANK_DEVICE", "1")
    host._PREP_CACHE.clear()
    require_cuda_device()
    got_mi, got_ok = host.bootstrap_rank_mi(arr, y, reps, 10)
    np.testing.assert_array_equal(got_ok, want_ok)
    # the same walk and the same sums; CUDA's log and numba's differ in the last bit
    np.testing.assert_allclose(got_mi, want_mi, rtol=1e-12, atol=1e-14)


def test_too_many_classes_stay_on_the_host(monkeypatch):
    """More than 16 classes is outside the kernel: the device helper declines and the host path answers."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._bootstrap_rank_mi_gpu import distinct_walk_gpu

    arr, y, reps = _data(2000, 4, 20)
    prep = host._resample_tables(reps, y, arr.shape[0], 10, True)
    assert distinct_walk_gpu(arr, prep, 10) is None


def test_small_column_blocks_and_a_device_fault_keep_the_result(monkeypatch):
    """A block of 3 columns gives the same MI; a fault inside the device walk falls back to the host walk instead of raising."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters import _bootstrap_rank_mi_gpu as gpu

    arr, y, reps = _data(3000, 10)
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_RANK_DEVICE", "0")
    host._PREP_CACHE.clear()
    want_mi, want_ok = host.bootstrap_rank_mi(arr, y, reps, 10)
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_RANK_DEVICE", "1")
    monkeypatch.setattr(gpu, "_bootstrap_walk_column_block", lambda cp, n: 3)
    host._PREP_CACHE.clear()
    require_cuda_device()
    got_mi, got_ok = host.bootstrap_rank_mi(arr, y, reps, 10)
    np.testing.assert_array_equal(got_ok, want_ok)
    np.testing.assert_allclose(got_mi, want_mi, rtol=1e-12, atol=1e-14)

    def _boom(cp, n):
        """A block-size probe that fails like an exhausted pool."""
        raise MemoryError("pool exhausted")

    monkeypatch.setattr(gpu, "_bootstrap_walk_column_block", _boom)
    host._PREP_CACHE.clear()
    mi2, ok2 = host.bootstrap_rank_mi(arr, y, reps, 10)
    np.testing.assert_array_equal(ok2, want_ok)
    np.testing.assert_allclose(mi2, want_mi, rtol=1e-12, atol=1e-14)


def test_a_genuine_oom_notes_the_shared_cooldown(monkeypatch):
    """GPU11-13 follow-up: this call site adaptively shrinks its own block (pool_column_block) rather than
    using the fe_gpu_has_vram_cushion pre-check, but can still OOM past that; a genuine cupy OutOfMemoryError
    (not any other fault, e.g. the plain MemoryError the test above simulates) must arm the shared cooldown."""
    pytest.importorskip("cupy")
    import cupy as cp
    from mlframe.feature_selection.filters import _bootstrap_rank_mi_gpu as gpu
    import mlframe.feature_selection.filters._fe_gpu_vram as vram_mod

    vram_mod._reset_gpu_oom_cooldown()
    arr, y, reps = _data(3000, 10)
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_RANK_DEVICE", "1")

    def _boom(*_args, **_kwargs):
        """Simulate a genuine device OutOfMemoryError, not a generic fault."""
        raise cp.cuda.memory.OutOfMemoryError(1, 0, 0)

    monkeypatch.setattr(gpu, "_bootstrap_walk_column_block", _boom)
    host._PREP_CACHE.clear()
    require_cuda_device()
    mi3, ok3 = host.bootstrap_rank_mi(arr, y, reps, 10)
    assert mi3 is not None and ok3 is not None  # still falls back to the host walk, never raises
    assert vram_mod._in_oom_cooldown() is True
    vram_mod._reset_gpu_oom_cooldown()
