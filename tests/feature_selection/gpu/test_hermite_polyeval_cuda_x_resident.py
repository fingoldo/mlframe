"""RESIDENT UPLOAD (wave 10): ``_polyeval_cuda`` must upload the fit-constant ``x`` column ONCE across
repeated calls with the SAME content (mirrors successive CMA-ES/Optuna trials re-evaluating the SAME
column with DIFFERENT coefficient vectors ``c``), instead of a fresh ``cp.asarray`` every call. ``c``
genuinely varies per trial and stays a raw upload. Proves the ``resident_operand`` adoption fix in
``hermite_fe/__init__.py`` engages and stays bit-identical to the pre-fix raw-upload path.

Only reachable via ``MLFRAME_POLYEVAL_BACKEND=cuda`` (never the un-forced default -- see
``polyeval_dispatch``'s own docstring), so this test calls ``_polyeval_cuda`` directly rather than
through the dispatcher.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.hermite_fe import _CUDA_AVAILABLE

cupy = pytest.importorskip("cupy") if _CUDA_AVAILABLE else None
if not _CUDA_AVAILABLE:
    pytest.skip("cupy/CUDA not available", allow_module_level=True)

from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands
from mlframe.feature_selection.filters.hermite_fe import _hermeval_njit, _polyeval_cuda


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_fe_resident_operands()
    yield
    clear_fe_resident_operands()


def test_polyeval_cuda_uploads_x_once_across_trials_with_same_column():
    """Two calls on the SAME x column content but DIFFERENT coefficient vectors (mirrors two CMA-ES/Optuna
    trials scoring the same column) must upload the x-shaped array via cp.asarray only ONCE; c must still
    upload every call (it genuinely varies per trial)."""
    import cupy as cp

    rng = np.random.default_rng(7)
    x = rng.normal(size=5000)
    x_same_content = x.copy()  # independent host object, identical bytes
    c1 = rng.normal(size=6)
    c2 = rng.normal(size=6)

    upload_calls = {"x": 0, "c": 0}
    orig_asarray = cp.asarray

    def _counting_asarray(arr, *a, **kw):
        shp = getattr(arr, "shape", None)
        if shp == x.shape:
            upload_calls["x"] += 1
        elif shp == c1.shape:
            upload_calls["c"] += 1
        return orig_asarray(arr, *a, **kw)

    cp.asarray = _counting_asarray
    try:
        out1 = _polyeval_cuda("hermite", x, c1)
        out2 = _polyeval_cuda("hermite", x_same_content, c2)
    finally:
        cp.asarray = orig_asarray

    assert upload_calls["x"] == 1, f"x-shaped cp.asarray called {upload_calls['x']} times across 2 trials (expected 1)"
    assert upload_calls["c"] == 2, f"c-shaped cp.asarray called {upload_calls['c']} times across 2 trials (expected 2, c varies per trial)"

    ref1 = _hermeval_njit(x, c1)
    ref2 = _hermeval_njit(x_same_content, c2)
    np.testing.assert_allclose(out1, ref1, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(out2, ref2, atol=1e-9, rtol=1e-9)


def test_polyeval_cuda_bit_identical_to_prefix_raw_path():
    """The resident-upload path must be bit-identical to the pre-fix raw cp.asarray(x, dtype=float64) path."""
    rng = np.random.default_rng(9)
    x = rng.normal(size=3000)
    c = rng.normal(size=4)

    new_result = _polyeval_cuda("legendre", x, c)

    # Reconstruct the exact pre-fix body (raw cp.asarray for both x and c, no resident cache).
    import contextlib

    import cupy as cp

    # _ensure_cuda_kernels populates the _hermite_fe_mi module's _CUDA_KERNELS dict (the one _polyeval_cuda
    # actually reads via `_hfmi._CUDA_KERNELS`) -- the hermite_fe package-level dict of the same name is a
    # stale, always-empty local (see _polyeval_cuda's own comment on this).
    import mlframe.feature_selection.filters._hermite_fe_mi as hfmi
    from mlframe.feature_selection.filters.hermite_fe import _ensure_cuda_kernels

    clear_fe_resident_operands()
    with contextlib.nullcontext():
        _ensure_cuda_kernels()
        x_gpu = cp.asarray(x, dtype=cp.float64)
        c_gpu = cp.asarray(c, dtype=cp.float64)
        n = x.shape[0]
        out_gpu = cp.empty(n, dtype=cp.float64)
        block = 256
        grid = (n + block - 1) // block
        hfmi._CUDA_KERNELS["legendre"]((grid,), (block,), (x_gpu, c_gpu, c_gpu.shape[0], n, out_gpu))
        old_result = np.asarray(cp.asnumpy(out_gpu))

    np.testing.assert_array_equal(new_result, old_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
