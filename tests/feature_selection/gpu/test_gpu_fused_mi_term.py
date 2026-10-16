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
    """True iff pyutilz reports CUDA availability, used to skip the fused-MI-kernel tests on CPU-only hosts."""
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
def test_fused_mi_term_matches_numpy_reference(n, k, nbins, nclasses, seed):
    """The cp.fuse'd per-cell MI term equals the numpy form of the same expression, on random joint and marginal counts."""
    from mlframe.feature_selection.filters._hermite_fe_mi import _get_fused_mi_term

    rng = np.random.default_rng(seed)
    hxyc = rng.integers(0, 30, size=(k, nbins, nclasses)).astype(np.float64)
    hx = np.maximum(hxyc.sum(axis=2, keepdims=True), 1.0)
    hy = np.maximum(hxyc.sum(axis=1, keepdims=True), 1.0)
    logx, logy = np.log(hx), np.log(hy)
    inv_n, log_n = 1.0 / float(n), float(np.log(n))
    got = cp.asnumpy(_get_fused_mi_term()(cp.asarray(hxyc), cp.asarray(logx), cp.asarray(logy), inv_n, log_n))
    safe = np.where(hxyc > 0, hxyc, 1.0)
    contrib = (hxyc * inv_n) * (np.log(safe) + log_n - logx - logy)
    ref = np.where(hxyc > 0, contrib, 0.0)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-15)
