"""RESIDENT UPLOAD (wave 10): ``batch_pair_usability_corr_cuda`` must upload the fit-constant ``y``/
``operand_matrix`` ONCE across repeated calls with the SAME content -- e.g.
``batch_pair_tail_concentration_rankaware``'s own two back-to-back dispatcher calls (pair-form gate, then
single-form gate) on the SAME ``y``/``operand_matrix`` -- instead of a fresh ``numba.cuda.to_device`` every
call. Proves the ``resident_operand`` adoption fix in ``batch_pair_usability_corr_gpu.py`` engages and
stays bit-identical to the pre-fix raw-upload path. ``pair_a``/``pair_b``/``form_ids`` genuinely vary per
call and stay raw uploads.

Only reachable via ``force_backend="cuda"`` (the un-forced dispatcher default is CPU on this dev host --
see the module docstring), so these tests call ``batch_pair_usability_corr_cuda`` directly / force cuda.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.batch_pair_usability_corr_gpu import (
    ALL_FORM_IDS,
    ALL_PAIR_FORM_IDS,
    ALL_SINGLE_FORM_IDS,
    _CUDA_AVAIL,
    batch_pair_usability_corr_cuda,
    batch_pair_usability_corr_njit_parallel,
)
from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands

_gpu_only = pytest.mark.skipif(not _CUDA_AVAIL, reason="numba.cuda not available on this host")


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear cache."""
    clear_fe_resident_operands()
    yield
    clear_fe_resident_operands()


def _inputs(n_pairs=40, n=500, n_operands=None, seed=0):
    """Helper that inputs."""
    rng = np.random.default_rng(seed)
    if n_operands is None:
        n_operands = n_pairs + 2
    y = rng.normal(size=n)
    operand_matrix = rng.normal(size=(n_operands, n))
    pair_a = rng.integers(0, n_operands, size=n_pairs).astype(np.int64)
    pair_b = rng.integers(0, n_operands, size=n_pairs).astype(np.int64)
    return y, operand_matrix, pair_a, pair_b


@_gpu_only
def test_batch_pair_usability_corr_cuda_uploads_y_and_operand_once_across_calls():
    """Two calls with the SAME (y, operand_matrix) but DIFFERENT pair chunks/form_ids (mirrors the
    pair-form-gate then single-form-gate calls inside batch_pair_tail_concentration_rankaware) must upload
    the y-shaped and operand_matrix-shaped arrays via cp.asarray only ONCE each."""
    import cupy as cp

    y, operand_matrix, pair_a, pair_b = _inputs(seed=1)
    y2 = y.copy()
    operand_matrix2 = operand_matrix.copy()

    upload_calls = {"y": 0, "operand": 0}
    orig_asarray = cp.asarray

    def _counting_asarray(arr, *a, **kw):
        """Counting asarray."""
        shp = getattr(arr, "shape", None)
        if shp == y.shape:
            upload_calls["y"] += 1
        elif shp == operand_matrix.shape:
            upload_calls["operand"] += 1
        return orig_asarray(arr, *a, **kw)

    cp.asarray = _counting_asarray
    try:
        out1 = batch_pair_usability_corr_cuda(y, operand_matrix, pair_a, pair_b, ALL_PAIR_FORM_IDS)
        out2 = batch_pair_usability_corr_cuda(y2, operand_matrix2, pair_a, pair_b, ALL_SINGLE_FORM_IDS)
    finally:
        cp.asarray = orig_asarray

    assert upload_calls["y"] == 1, f"y-shaped cp.asarray called {upload_calls['y']} times across 2 calls (expected 1)"
    assert upload_calls["operand"] == 1, f"operand_matrix-shaped cp.asarray called {upload_calls['operand']} times across 2 calls (expected 1)"

    ref1 = batch_pair_usability_corr_njit_parallel(
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(operand_matrix, dtype=np.float64),
        pair_a,
        pair_b,
        ALL_PAIR_FORM_IDS,
    )
    ref2 = batch_pair_usability_corr_njit_parallel(
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(operand_matrix, dtype=np.float64),
        pair_a,
        pair_b,
        ALL_SINGLE_FORM_IDS,
    )
    np.testing.assert_allclose(out1, ref1, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(out2, ref2, atol=1e-9, rtol=1e-9)


@_gpu_only
def test_batch_pair_usability_corr_cuda_bit_identical_to_prefix_raw_path():
    """The resident-upload path must be bit-identical to the pre-fix raw to_device(y)/to_device(operand_matrix) path."""
    from numba import cuda as nb_cuda

    y, operand_matrix, pair_a, pair_b = _inputs(seed=2)
    new_result = batch_pair_usability_corr_cuda(y, operand_matrix, pair_a, pair_b, ALL_FORM_IDS)

    # Reconstruct the exact pre-fix body (raw numba.cuda.to_device for both y and operand_matrix).
    from mlframe.feature_selection.filters.batch_pair_usability_corr_gpu import (
        _EPS_DENOM_FLOOR,
        _CV2_DEGENERATE_FLOOR,
        _cuda_kernel_factory,
    )
    import mlframe.feature_selection.filters.batch_pair_usability_corr_gpu as mod

    if mod._CUDA_KERNEL is None:
        mod._CUDA_KERNEL = _cuda_kernel_factory()
    kernel = mod._CUDA_KERNEL

    clear_fe_resident_operands()
    form_ids = np.asarray(ALL_FORM_IDS, dtype=np.int64)
    y_d = nb_cuda.to_device(np.ascontiguousarray(y, dtype=np.float64))
    operand_d = nb_cuda.to_device(np.ascontiguousarray(operand_matrix, dtype=np.float64))
    pair_a_d = nb_cuda.to_device(np.ascontiguousarray(pair_a, dtype=np.int64))
    pair_b_d = nb_cuda.to_device(np.ascontiguousarray(pair_b, dtype=np.int64))
    form_ids_d = nb_cuda.to_device(form_ids)
    n_pairs = pair_a.shape[0]
    n_forms = form_ids.shape[0]
    out_d = nb_cuda.device_array((n_pairs, n_forms), dtype=np.float64)
    total = n_pairs * n_forms
    threads_per_block = 128
    blocks = (total + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](y_d, operand_d, pair_a_d, pair_b_d, form_ids_d, _EPS_DENOM_FLOOR, _CV2_DEGENERATE_FLOOR, out_d)
    old_result = np.asarray(out_d.copy_to_host())

    np.testing.assert_array_equal(new_result, old_result)


@_gpu_only
def test_rankaware_style_two_dispatcher_calls_share_one_resident_upload():
    """batch_pair_tail_concentration_rankaware issues TWO dispatch_batch_pair_usability_corr calls (pair-form
    gate, then single-form gate when single_corr is None) on the SAME y/operand_matrix. The un-forced
    dispatcher default is CPU on this dev host (measured-correct per the module docstring) and
    batch_pair_tail_concentration_rankaware does not expose a force_backend passthrough, so this test
    reproduces its EXACT two-call sequence with force_backend="cuda" to exercise the real code path the
    fix targets -- both calls must resolve to the SAME resident cache entries."""
    from mlframe.feature_selection.filters._fe_resident_operands import _FE_RESIDENT_OPERANDS
    from mlframe.feature_selection.filters.batch_pair_usability_corr_gpu import dispatch_batch_pair_usability_corr

    y, operand_matrix, pair_a, pair_b = _inputs(n_pairs=20, n=300, seed=3)

    n_before = len(_FE_RESIDENT_OPERANDS)
    pair_corrs, backend1 = dispatch_batch_pair_usability_corr(
        y,
        operand_matrix,
        pair_a,
        pair_b,
        form_ids=ALL_PAIR_FORM_IDS,
        force_backend="cuda",
    )
    n_after_pair = len(_FE_RESIDENT_OPERANDS)
    single_corrs, backend2 = dispatch_batch_pair_usability_corr(
        y,
        operand_matrix,
        pair_a,
        pair_b,
        form_ids=ALL_SINGLE_FORM_IDS,
        force_backend="cuda",
    )
    n_after_single = len(_FE_RESIDENT_OPERANDS)

    assert backend1 == "cuda" and backend2 == "cuda"
    assert pair_corrs.shape == (20, len(ALL_PAIR_FORM_IDS))
    assert single_corrs.shape == (20, len(ALL_SINGLE_FORM_IDS))
    assert n_after_pair > n_before, "first call must populate the resident cache with y + operand_matrix"
    assert n_after_single == n_after_pair, "second call on the SAME y/operand_matrix must hit the cache, not add new entries"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
