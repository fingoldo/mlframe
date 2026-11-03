"""The float32 warp kernel of the usability correlation stays within float32 accuracy of the float64 kernel and picks the same best form per pair."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba.cuda")
from numba import cuda

if not cuda.is_available():  # pragma: no cover - CPU-only host
    pytest.skip("CUDA unavailable", allow_module_level=True)

from mlframe.feature_selection.filters.batch_pair_usability_corr_gpu import batch_pair_usability_corr_cuda_warp


def test_float32_kernel_matches_double(monkeypatch):
    """|corr| within 1e-5 of the double kernel on informative and noise pairs with ratio forms (near-zero divisors included)."""
    rng = np.random.default_rng(0)
    n, k = 20000, 12
    ops = rng.standard_normal((k, n)).astype(np.float32)
    ops[3, ::50] = 0.0  # zero divisors for the ratio forms
    y = (ops[0] * ops[1] + 0.3 * ops[2] + 0.5 * rng.standard_normal(n)).astype(np.float64)
    pa = np.array([0, 0, 1, 2, 3, 5], dtype=np.int64)
    pb = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    forms = np.arange(9, dtype=np.int64)
    monkeypatch.setenv("MLFRAME_USABILITY_CORR_F32", "0")
    want = batch_pair_usability_corr_cuda_warp(y, ops, pa, pb, forms)
    monkeypatch.setenv("MLFRAME_USABILITY_CORR_F32", "1")
    got = batch_pair_usability_corr_cuda_warp(y, ops, pa, pb, forms)
    np.testing.assert_allclose(got, want, atol=1e-5)
    assert np.array_equal(np.argmax(got, axis=1), np.argmax(want, axis=1))
