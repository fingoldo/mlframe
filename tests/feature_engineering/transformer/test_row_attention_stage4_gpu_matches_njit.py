"""The GPU stage-4 row-attention kernel accumulates in double, so it matches the njit kernel on large-offset targets.

The njit kernel sums logits, weighted means and variances in float64. The RawKernel summed them in float32, and on a target near
1000 with unit-scale spread that drifted by about 8e-4 in the weighted mean and 2e-6 in the weighted std.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_DEVICE = False

pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


def test_gpu_stage4_matches_njit_on_large_offset_target():
    """GPU and njit stage-4 agree within float32 output precision on a target near 1000 with unit spread."""
    from mlframe.feature_engineering.transformer._kernels_cupy import row_attention_stage4_cupy
    from mlframe.feature_engineering.transformer._kernels_njit import row_attention_stage4_njit

    rng = np.random.default_rng(5)
    nq, nt, hd, K = 400, 3000, 32, 128
    q = rng.normal(size=(nq, hd)).astype(np.float32)
    k = rng.normal(size=(nt, hd)).astype(np.float32)
    y = (1000.0 + rng.normal(0.0, 1.0, nt)).astype(np.float32)
    ids = np.stack([rng.choice(nt, K, replace=False) for _ in range(nq)]).astype(np.int32)

    def run(fn):
        """Run one stage-4 backend on the shared inputs and return its three output arrays."""
        ym = np.empty(nq, np.float32)
        ys = np.empty(nq, np.float32)
        xm = np.empty((nq, hd), np.float32)
        fn(q, k, y, ids, 1.0, ym, ys, xm)
        return ym, ys, xm

    ref_mean, ref_std, ref_x = run(row_attention_stage4_njit)
    gpu_mean, gpu_std, gpu_x = run(row_attention_stage4_cupy)

    # Four float32 ulps at the target magnitude (about 1000): the output and the target storage round at that scale.
    assert float(np.max(np.abs(gpu_mean.astype(np.float64) - ref_mean))) <= 4 * float(np.spacing(np.float32(1000.0)))
    assert float(np.max(np.abs(gpu_std.astype(np.float64) - ref_std))) <= 5e-7
    assert float(np.max(np.abs(gpu_x.astype(np.float64) - ref_x))) <= 2e-6
