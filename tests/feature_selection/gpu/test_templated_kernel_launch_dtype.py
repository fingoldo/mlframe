"""Launch-dtype tests for the ``{REAL}``-templated RawKernels.

Each kernel is launched once with float32 buffers (``MLFRAME_CRIT_DTYPE_RELAXED=1``, the default) and once with float64 buffers
(``MLFRAME_CRIT_DTYPE_RELAXED=0``) and compared to the host reference: 1e-6 for the float32 launch (a float32-vs-float64 comparison),
1e-9 for the float64 launch. The mismatch guard ``assert_kernel_buffers`` is checked directly. GPU tests skip without CUDA/cupy.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_gpu_batch._devices import assert_kernel_buffers
from mlframe.feature_selection.filters.info_theory._cmi_cuda import conditional_mi_batched_dispatch, cupy_available
from mlframe.feature_selection.filters.info_theory._entropy_kernels import conditional_mi

_HAS_GPU = cupy_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="no CUDA/cupy GPU available")
_ENV = pytest.mark.parametrize("relaxed, rtol", [("1", 1e-6), ("0", 1e-9)], ids=["float32", "float64"])


def test_guard_rejects_a_buffer_whose_dtype_disagrees_with_real_t():
    """A float32 buffer under real_t 'double' (or float64 under 'float') raises ValueError naming the kernel and buffer; no silent cast."""
    f32 = np.zeros(4, dtype=np.float32)
    f64 = np.zeros(4, dtype=np.float64)
    assert_kernel_buffers("k", "float", buf=f32)
    assert_kernel_buffers("k", "double", buf=f64)
    with pytest.raises(ValueError, match="k: buffer 'buf'"):
        assert_kernel_buffers("k", "double", buf=f32)
    with pytest.raises(ValueError, match="k: buffer 'buf'"):
        assert_kernel_buffers("k", "float", buf=f64)
    with pytest.raises(ValueError, match="unknown real_t"):
        assert_kernel_buffers("k", "half", buf=f32)


@_gpu_only
@_ENV
def test_cmi_from_joint_launch_matches_cpu(monkeypatch, relaxed, rtol):
    """cmi_from_joint (templated cmi_out) matches the CPU conditional_mi reference at the launch dtype's tolerance."""
    monkeypatch.setenv("MLFRAME_CRIT_DTYPE_RELAXED", relaxed)
    rng = np.random.RandomState(0)
    n, p = 3000, 12
    cols = [rng.randint(0, 4, n) for _ in range(p)]
    fd = np.column_stack([*cols, rng.randint(0, 3, n), rng.randint(0, 3, n)]).astype(np.int32)
    fb = np.array([4] * p + [3, 3], dtype=np.int64)
    gpu = np.asarray(conditional_mi_batched_dispatch(fd, np.arange(p), p, p + 1, fb, force="cuda"), dtype=np.float64)
    ref = np.array([conditional_mi(fd, np.array([c]), np.array([p]), np.array([p + 1]), None, fb) for c in range(p)])
    np.testing.assert_allclose(gpu, ref, rtol=rtol, atol=rtol)


@_gpu_only
@_ENV
def test_subset_loss_launch_matches_host(monkeypatch, relaxed, rtol):
    """subset_loss (templated buffers) returns the same best subset and loss as the numba search at the launch dtype's tolerance."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_gpu import brute_force_top_n_gpu
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n

    monkeypatch.setenv("MLFRAME_CRIT_DTYPE_RELAXED", relaxed)
    rng = np.random.default_rng(3)
    phi = rng.normal(size=(300, 8))
    base = np.full(300, 0.1)
    y = (1.0 / (1.0 + np.exp(-(base + phi[:, [1, 4]].sum(axis=1)))) > rng.random(300)).astype(float)
    cpu = brute_force_top_n(phi, base, y, classification=True, max_card=4, top_n=3)
    gpu = brute_force_top_n_gpu(phi, base, y, classification=True, max_card=4, top_n=3)
    assert set(cpu[0][1]) == set(gpu[0][1])
    np.testing.assert_allclose(cpu[0][0], gpu[0][0], rtol=rtol, atol=rtol)


@_gpu_only
@_ENV
def test_candidate_mi_build_kernels_match_row_and_column_major(monkeypatch, relaxed, rtol):
    """build_op_cands and build_op_cands_cm agree with each other (transposed) at the launch dtype's tolerance."""
    import cupy as cp

    from mlframe.feature_selection.filters._resident_candidate_mi import (
        _build_best_existing_op_candidates_gpu,
        _build_best_existing_op_candidates_gpu_cm,
    )

    monkeypatch.setenv("MLFRAME_CRIT_DTYPE_RELAXED", relaxed)
    rng = np.random.default_rng(5)
    cols = [cp.asarray(rng.normal(size=500)) for _ in range(4)]
    row = cp.asnumpy(_build_best_existing_op_candidates_gpu(cols, cp)).astype(np.float64)
    col = cp.asnumpy(_build_best_existing_op_candidates_gpu_cm(cols, cp)).astype(np.float64).T
    np.testing.assert_allclose(row, col, rtol=rtol, atol=rtol)
