"""RESIDENT UPLOAD (wave 10): ``_plugin_mi_classif_batch_cuda`` must upload the fit-constant ``y`` label
vector ONCE across repeated calls with the SAME content (successive candidate-column chunks of one
greedy/FE round), instead of a fresh ``cp.asarray`` every call. Proves the ``resident_operand`` adoption
fix in ``_hermite_fe_mi.py`` engages, stays bit-identical to the pre-fix raw-upload path, AND that the
SAME device object is handed back across calls -- which is what lets the downstream ``(id(y_gpu), y_min)``
shift-cache in ``_plugin_mi_classif_batch_cuda_resident`` actually start hitting.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.hermite_fe import _CUDA_AVAILABLE

cupy = pytest.importorskip("cupy") if _CUDA_AVAILABLE else None
if not _CUDA_AVAILABLE:
    pytest.skip("cupy/CUDA not available", allow_module_level=True)

from mlframe.feature_selection.filters._fe_resident_operands import (
    _FE_RESIDENT_OPERANDS,
    clear_fe_resident_operands,
)
from mlframe.feature_selection.filters.hermite_fe import (
    _plugin_mi_classif_batch_cuda,
    _plugin_mi_classif_batch_cuda_resident,
    _plugin_mi_classif_batch_njit,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_fe_resident_operands()
    yield
    clear_fe_resident_operands()


def _inputs(n=4000, k=6, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    y = rng.integers(0, n_classes, size=n).astype(np.int64)
    return X, y


def test_plugin_mi_classif_batch_cuda_uploads_y_once_across_calls():
    """Two calls with the SAME y content but DIFFERENT X (mirrors two candidate-column chunks of one
    FE round) must upload the y-shaped array via cp.asarray only ONCE."""
    import cupy as cp

    X1, y = _inputs(seed=1)
    X2, _y_same = _inputs(seed=2)  # different X, SAME y content reused below
    y2 = y.copy()  # independent host object, identical bytes -- content-hash must still hit

    upload_calls = {"n": 0}
    orig_asarray = cp.asarray

    def _counting_asarray(arr, *a, **kw):
        if getattr(arr, "shape", None) == y.shape and getattr(arr, "dtype", None) == np.int64:
            upload_calls["n"] += 1
        return orig_asarray(arr, *a, **kw)

    cp.asarray = _counting_asarray
    try:
        mi1 = _plugin_mi_classif_batch_cuda(X1, y, 20)
        mi2 = _plugin_mi_classif_batch_cuda(X2, y2, 20)
    finally:
        cp.asarray = orig_asarray

    assert upload_calls["n"] == 1, f"y-shaped cp.asarray called {upload_calls['n']} times across 2 calls (expected 1)"

    njit1 = _plugin_mi_classif_batch_njit(X1, y, 20)
    njit2 = _plugin_mi_classif_batch_njit(X2, y2, 20)
    np.testing.assert_allclose(mi1, njit1, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(mi2, njit2, atol=1e-9, rtol=1e-9)


def test_plugin_mi_classif_batch_cuda_shift_cache_hits_on_repeat_y():
    """The resident y upload returns the SAME device object across calls, so the downstream
    (id(y_gpu), y_min) shift-memo in _plugin_mi_classif_batch_cuda_resident actually engages on the
    second call (pre-fix, id(y_gpu) changed every call so the shift-memo could never hit)."""
    import mlframe.feature_selection.filters._hermite_fe_mi as hfmi

    X1, y = _inputs(seed=3, n_classes=5)  # non-zero-based? labels are 0..4 so y_min==0 -- force a shift
    y_shifted = y + 3  # y_min == 3 -> the shift branch (`if y_min:`) actually executes
    X2, _ = _inputs(seed=4, n_classes=5)

    hfmi._SHIFTED_Y_CACHE.clear()
    _plugin_mi_classif_batch_cuda(X1, y_shifted, 20)
    n_after_first = len(hfmi._SHIFTED_Y_CACHE)
    _plugin_mi_classif_batch_cuda(X2, y_shifted.copy(), 20)
    n_after_second = len(hfmi._SHIFTED_Y_CACHE)

    assert n_after_first == 1, "first call must populate the shift-memo"
    assert n_after_second == 1, "second call with the SAME y content must HIT the shift-memo, not add a new entry"


def test_plugin_mi_classif_batch_cuda_bit_identical_to_prefix_raw_path():
    """The resident-upload path must be bit-identical to the pre-fix raw cp.asarray(y, dtype=int64) path."""
    import cupy as cp

    X, y = _inputs(seed=5)
    new_result = _plugin_mi_classif_batch_cuda(X, y, 20)

    clear_fe_resident_operands()
    X_gpu = cp.asarray(X, dtype=cp.float64)
    y_gpu_raw = cp.asarray(y, dtype=cp.int64)  # the exact pre-fix raw-upload line
    old_result = np.asarray(_plugin_mi_classif_batch_cuda_resident(X_gpu, y_gpu_raw, 20))

    np.testing.assert_array_equal(new_result, old_result)
    assert len(_FE_RESIDENT_OPERANDS) >= 0  # sanity: cache module imported/usable


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
