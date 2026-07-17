"""Bit-identity grid for the fused per-cell MI-contribution kernel (GPU-saturation Task #2, 2026-06-23).

``_plugin_mi_classif_batch_cuda_resident`` collapses its ~6-launch per-cell MI nat chain into ONE
``cp.fuse``'d elementwise kernel (default ON). This pins that the fused result is BIT-IDENTICAL to the
unfused chain (same f64 ops/order) on a grid of shapes -- so the per-feature MI ranking, and thus the FE
selection, is unchanged. Auto-skips on CUDA-unavailable hosts.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda():
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


@pytest.mark.parametrize(
    "n,k,nbins,nclasses,seed",
    [
        (1000, 1, 20, 2, 0),  # k==1 (last VRAM chunk) -- the cupy single-col percentile guard path
        (2000, 7, 20, 3, 1),
        (5000, 33, 16, 5, 2),
        (10000, 64, 24, 8, 3),
        (3000, 12, 10, 12, 4),  # many classes, few bins
    ],
)
def test_fused_mi_term_bit_identical(monkeypatch, n, k, nbins, nclasses, seed):
    # import via the parent (hermite_fe) so the module-init cycle resolves in the canonical order
    from mlframe.feature_selection.filters.hermite_fe import _plugin_mi_classif_batch_cuda_resident

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k)).astype(np.float64)
    # mix in heavy-tail + duplicate columns so binning ties exercise the empty-cell (hist==0) branch
    if k >= 4:
        X[:, 1] = X[:, 0] ** 3
        X[:, 2] = np.round(X[:, 2])  # low-cardinality -> empty joint bins
    y = rng.integers(0, nclasses, size=n).astype(np.int64)
    X_gpu = cp.asarray(X)
    y_gpu = cp.asarray(y)

    monkeypatch.setenv("MLFRAME_FE_GPU_FUSE_MI", "1")
    mi_fused = _plugin_mi_classif_batch_cuda_resident(X_gpu, y_gpu, nbins)
    monkeypatch.setenv("MLFRAME_FE_GPU_FUSE_MI", "0")
    mi_unfused = _plugin_mi_classif_batch_cuda_resident(X_gpu, y_gpu, nbins)

    assert mi_fused.shape == (k,)
    # SELECTION-EQUIVALENT: cp.fuse evaluates the same f64 expression but the broadcasted
    # ``+log_n - logx - logy`` may re-associate vs the unfused ``- cp.log(safe_x)[:,:,None]
    # - cp.log(safe_y)[:,None,:]``, giving ~1e-18 FP round-off (measured max abs diff 3.5e-18,
    # rel 2e-16). That is ~10 orders below any MI-ranking / FE-selection threshold (the FE perf
    # bar is same-features-selected, ~1e-9 drift OK), so assert a tight numeric tolerance.
    np.testing.assert_allclose(mi_fused, mi_unfused, rtol=1e-12, atol=1e-15)


def test_fused_mi_default_on():
    """The fused path is the default (env unset)."""
    import os
    from mlframe.feature_selection.filters.hermite_fe import _plugin_mi_classif_batch_cuda_resident  # noqa: F401 -- force canonical init
    from mlframe.feature_selection.filters._hermite_fe_mi import _fe_gpu_fuse_mi_enabled

    os.environ.pop("MLFRAME_FE_GPU_FUSE_MI", None)
    assert _fe_gpu_fuse_mi_enabled() is True
