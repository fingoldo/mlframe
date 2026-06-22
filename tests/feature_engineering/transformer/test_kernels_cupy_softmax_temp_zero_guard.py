"""Regression test: ``row_attention_stage4_cupy`` must guard ``softmax_temp == 0`` (no inf), mirroring the njit twin.

The bug (fixed): the cupy kernel passed ``1.0 / softmax_temp`` with no zero guard, so ``softmax_temp == 0`` produced ``inf`` for the inverse temperature and NaN/inf
outputs. The njit twins guard ``softmax_temp > 1e-12``; the fix mirrors that guard at the cupy call site.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_engineering.transformer._kernels_cupy import (
    is_gpu_available,
    row_attention_stage4_cupy,
)

pytestmark = pytest.mark.fast


def test_cupy_stage4_softmax_temp_zero_no_inf():
    if not is_gpu_available():
        pytest.skip("GPU not available")
    rng = np.random.default_rng(0)
    n_queries, head_dim, k, n_train = 4, 8, 3, 10
    q_proj = rng.standard_normal((n_queries, head_dim)).astype(np.float32)
    k_proj = rng.standard_normal((n_train, head_dim)).astype(np.float32)
    y_train = rng.standard_normal(n_train).astype(np.float32)
    topk_ids = rng.integers(0, n_train, size=(n_queries, k)).astype(np.int32)
    y_mean = np.empty(n_queries, dtype=np.float32)
    y_std = np.empty(n_queries, dtype=np.float32)
    x_mean = np.empty((n_queries, head_dim), dtype=np.float32)

    row_attention_stage4_cupy(q_proj, k_proj, y_train, topk_ids, 0.0, y_mean, y_std, x_mean)

    assert np.isfinite(y_mean).all() and np.isfinite(y_std).all() and np.isfinite(x_mean).all(), (
        "softmax_temp=0 produced non-finite output; the cupy kernel must guard 1/softmax_temp like its njit twin."
    )
