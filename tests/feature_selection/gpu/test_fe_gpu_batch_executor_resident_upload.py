"""RESIDENT UPLOAD (2026-07-13): ``gpu_fe_batch_mi`` (``_fe_gpu_batch/_executor.py``) must upload the
fit-constant ``y_codes`` ONCE across repeated calls with the SAME target (mirrors many candidate-matrix
batches scored against one fixed y within a single fit, via ``_fe_batch_dispatch.fe_batch_mi`` /
``_orth_mi_backends._mi_classif_batch_numba``), instead of a fresh ``cp.asarray`` every call. Also pins that
``y_min``/``n_classes`` (derived host-side post-fix, no device sync) stay correct.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._fe_cpu_batch import cpu_fe_batch_mi
from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands


def _gpu_available() -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() >= 1
    except Exception:  # pragma: no cover - no driver / no GPU
        return False


_GPU_AVAILABLE = _gpu_available()
if not _GPU_AVAILABLE:  # pragma: no cover - guarded at collection time
    pytest.skip("No CUDA device available", allow_module_level=True)

from mlframe.feature_selection.filters._fe_gpu_batch._executor import gpu_fe_batch_mi


@pytest.fixture(autouse=True)
def _clear_resident_cache():
    clear_fe_resident_operands()
    cp.get_default_memory_pool().free_all_blocks()
    yield
    clear_fe_resident_operands()
    cp.get_default_memory_pool().free_all_blocks()


def _make_two_candidate_batches(n=4000, k=20, nbins=10, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.uniform(1, 5, n)
    y = np.searchsorted(np.quantile(a, np.linspace(0, 1, nbins + 1))[1:-1], a).astype(np.int64)
    X1 = np.column_stack([np.nan_to_num((a**2 / rng.uniform(1, 5, n)).astype(np.float64)) for _ in range(k)])
    X2 = np.column_stack([rng.uniform(0, 1, n).astype(np.float64) for _ in range(k)])
    return X1, X2, y, nbins


def _count_asarray_calls_matching(monkeypatch, target_values):
    calls = {"n": 0}
    orig = cp.asarray
    target = np.ascontiguousarray(target_values)

    def spy(a, *args, **kw):
        if isinstance(a, np.ndarray) and a.shape == target.shape and a.dtype == target.dtype and np.array_equal(a, target):
            calls["n"] += 1
        return orig(a, *args, **kw)

    monkeypatch.setattr(cp, "asarray", spy)
    return calls


@pytest.mark.gpu
def test_y_uploaded_once_across_two_candidate_batches(monkeypatch):
    X1, X2, y, nbins = _make_two_candidate_batches(seed=2)
    y_i64 = np.ascontiguousarray(y, dtype=np.int64)
    y_calls = _count_asarray_calls_matching(monkeypatch, y_i64)

    mi1 = gpu_fe_batch_mi(X1, y, nbins)
    mi2 = gpu_fe_batch_mi(X2, y, nbins)

    assert y_calls["n"] == 1, f"y-content cp.asarray called {y_calls['n']}x across 2 candidate-matrix calls (expected 1, resident)"
    assert mi1.shape == (X1.shape[1],)
    assert mi2.shape == (X2.shape[1],)
    assert np.all(np.isfinite(mi1)) and np.all(np.isfinite(mi2))


@pytest.mark.gpu
def test_resident_y_matches_cpu_reference_and_is_bit_identical_to_disabled_cache(monkeypatch):
    X1, X2, y, nbins = _make_two_candidate_batches(seed=6, n=3000, k=12)

    clear_fe_resident_operands()
    gpu1 = gpu_fe_batch_mi(X1, y, nbins)
    gpu2 = gpu_fe_batch_mi(X2, y, nbins)  # second call must hit the resident y cache

    monkeypatch.setenv("MLFRAME_FE_RESIDENT_OPERANDS", "0")
    clear_fe_resident_operands()
    gpu1_fresh = gpu_fe_batch_mi(X1, y, nbins)
    gpu2_fresh = gpu_fe_batch_mi(X2, y, nbins)

    np.testing.assert_array_equal(gpu1, gpu1_fresh)
    np.testing.assert_array_equal(gpu2, gpu2_fresh)

    cpu1 = cpu_fe_batch_mi(X1, y, nbins)
    cpu2 = cpu_fe_batch_mi(X2, y, nbins)
    assert np.allclose(gpu1, cpu1, atol=1e-9, rtol=0)
    assert np.allclose(gpu2, cpu2, atol=1e-9, rtol=0)


@pytest.mark.gpu
def test_host_derived_y_min_n_classes_match_device_derivation():
    """The fix moves y_min/n_classes off a device .min()/.max() sync onto a host numpy reduction -- pin
    that the values (and therefore the MI they gate) are unaffected by a y_codes array whose min is > 0."""
    n, nbins = 2000, 6
    rng = np.random.default_rng(11)
    a = rng.uniform(1, 5, n)
    y = np.searchsorted(np.quantile(a, np.linspace(0, 1, nbins + 1))[1:-1], a).astype(np.int64)
    y_shifted = y + 3  # y_min != 0 -- exercises the y_min offset path host- and device-side alike
    X = np.column_stack([np.nan_to_num((a**2 / rng.uniform(1, 5, n)).astype(np.float64)) for _ in range(8)])

    out_shifted = gpu_fe_batch_mi(X, y_shifted, nbins)
    out_unshifted = gpu_fe_batch_mi(X, y, nbins)
    np.testing.assert_allclose(out_shifted, out_unshifted, atol=1e-9, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
