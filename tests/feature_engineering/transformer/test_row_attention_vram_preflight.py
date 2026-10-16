"""The GPU row-attention launch warns, and still runs, when one head's working set exceeds the free VRAM."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.feature_engineering.transformer import _kernels_cupy as kc
from mlframe.feature_engineering.transformer._utils import is_gpu_available


@pytest.mark.skipif(not is_gpu_available(), reason="needs a CUDA GPU")
def test_stage4_warns_when_free_vram_is_short(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """With the free-VRAM figure forced to zero, the launch logs the estimate and still returns its outputs."""
    monkeypatch.setattr(kc, "gpu_available_bytes", lambda *a, **k: 0)
    rng = np.random.default_rng(0)
    n_train, n_q, d, k = 64, 16, 8, 4
    q = rng.normal(size=(n_q, d)).astype(np.float32)
    kp = rng.normal(size=(n_train, d)).astype(np.float32)
    y = rng.normal(size=n_train).astype(np.float32)
    ids = np.tile(np.arange(k, dtype=np.int32), (n_q, 1))
    ym, ys, xm = np.zeros(n_q, np.float32), np.ones(n_q, np.float32), np.zeros((n_q, d), np.float32)
    with caplog.at_level(logging.WARNING, logger=kc.logger.name):
        kc.row_attention_stage4_cupy(q, kp, y, ids, 1.0, ym, ys, xm)
    assert any("exceeds ~" in r.getMessage() for r in caplog.records)
    assert np.isfinite(ym).all()
