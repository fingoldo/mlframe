"""The GPU binning leg drops its resident (n, K) codes buffer when it does not fit free VRAM, and the codes stay bit-identical.

Before the fix the full resident buffer was allocated unconditionally. On a 4 GB card it oversubscribed VRAM and the blockwise
uploads that followed stalled. The resident copy is an optimisation, so the leg now keeps the codes on the host when the
cushion check fails.
"""

import logging

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_DEVICE = False

pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="no CUDA device")


def test_discretize_drops_resident_codes_when_vram_is_short_and_stays_bit_identical(monkeypatch, caplog):
    """With no VRAM cushion the leg logs the decision and returns codes equal to the CPU binning."""
    from mlframe.feature_selection.filters import _fe_gpu_vram
    from mlframe.feature_selection.filters._gpu_resident_fe import gpu_discretize_codes_host
    from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch

    monkeypatch.setenv("MLFRAME_FE_GPU_RESIDENT_CODES", "1")
    rng = np.random.default_rng(7)
    n, K = 20_000, 30
    cand = np.ascontiguousarray(rng.uniform(1, 5, size=(n, K)) * rng.uniform(1, 5, size=(n, K))).astype(np.float32)
    cpu = discretize_2d_quantile_batch(cand, n_bins=20, dtype=np.int8, assume_finite=True)
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda *_a, **_k: False)
    with caplog.at_level(logging.WARNING):
        gpu = gpu_discretize_codes_host(cand, 20, dtype=np.int8)
    assert np.array_equal(cpu, gpu)
    assert any("does not fit the free VRAM cushion" in r.getMessage() for r in caplog.records)
