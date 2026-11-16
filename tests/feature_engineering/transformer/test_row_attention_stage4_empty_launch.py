"""The stage-4 row-attention launcher handles zero queries and rejects zero neighbours without an invalid CUDA launch."""

import numpy as np
import pytest

from mlframe.feature_engineering.transformer._kernels_cupy import row_attention_stage4_cupy


def _outputs(n):
    """Writable output arrays for n queries."""
    return np.zeros(n), np.zeros(n), np.zeros(n)


def test_zero_queries_returns_without_launching():
    """An empty batch is a no-op: no kernel launch with a zero-sized grid."""
    q = np.zeros((0, 4))
    k = np.zeros((10, 4))
    y = np.zeros(10)
    topk = np.zeros((0, 3), dtype=np.int64)
    y_mean, y_std, x_mean = _outputs(0)
    assert row_attention_stage4_cupy(q, k, y, topk, 1.0, y_mean, y_std, x_mean) is None


def test_zero_neighbours_is_rejected_with_a_clear_error():
    """K=0 is a caller error and must say so instead of failing inside CUDA."""
    q = np.zeros((2, 4))
    k = np.zeros((10, 4))
    y = np.zeros(10)
    topk = np.zeros((2, 0), dtype=np.int64)
    y_mean, y_std, x_mean = _outputs(2)
    with pytest.raises(ValueError, match="at least 1 neighbour"):
        row_attention_stage4_cupy(q, k, y, topk, 1.0, y_mean, y_std, x_mean)


def test_resident_float64_bank_is_rejected_not_reinterpreted(monkeypatch):
    """A device bank of the wrong dtype must raise before any kernel reads it as float32."""
    import mlframe.feature_engineering.transformer._kernels_cupy as kc

    monkeypatch.setattr(kc, "is_gpu_available", lambda: True)
    monkeypatch.setattr(kc, "_ensure_kernels_inited", lambda: None)
    q = np.zeros((2, 4), dtype=np.float32)
    k = np.zeros((10, 4), dtype=np.float32)
    y = np.zeros(10, dtype=np.float32)
    topk = np.zeros((2, 3), dtype=np.int64)
    y_mean, y_std, x_mean = _outputs(2)
    bank64 = np.zeros((10, 4), dtype=np.float64)
    with pytest.raises(TypeError, match="float32"):
        row_attention_stage4_cupy(q, k, y, topk, 1.0, y_mean, y_std, x_mean, k_proj_device=bank64)
