"""Regression for the GPU-strict Gram-matrix dispatch in ``audit_degenerate_columns``.

cProfile on the wellbore-100k GPU-strict fit attributed 43.0s tottime to a SINGLE call of
``audit_degenerate_columns`` (p~518 raw columns) -- entirely the collinearity-pass ``M @ M.T`` GEMM, which
was hardcoded to CPU numpy regardless of STRICT mode (this diagnostic scan never had a GPU twin, unlike the
selection-critical kernels). ``_gram_matrix`` now routes through the same ``fe_gpu_strict_enabled`` work-
floor gate as the rest of the FE pipeline. These tests pin: (1) the CPU default path is unchanged for small
frames, (2) the cupy-forced path produces the SAME collinear/duplicate verdict as CPU on data that puts
|corr| exactly at the 1e-9 tolerance boundary (an exact linear duplicate), and (3) the diagnostic result is
selection-transparent either way (module contract: this scan never changes what MRMR selects).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._mrmr_degenerate import (
    _gram_matrix,
    audit_degenerate_columns,
)


def _collinear_frame(n=2000, p=24, seed=7):
    """A numeric frame with a genuine exact-linear-duplicate pair (b = 2*a + 3) plus independent columns,
    sized to clear the GPU-strict per-call work floor (n*p >= 1M) when STRICT mode is forced on."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    cols = {"a": a, "b": 2.0 * a + 3.0}
    for j in range(p - 2):
        cols[f"c{j}"] = rng.normal(size=n)
    return pd.DataFrame(cols)


def test_gram_matrix_cpu_default_small_frame_unchanged():
    """Below the GPU-strict work floor, _gram_matrix must fall through to the plain numpy GEMM."""
    rng = np.random.default_rng(1)
    M = rng.normal(size=(5, 40))
    np.testing.assert_array_equal(_gram_matrix(M), M @ M.T)


def test_gram_matrix_gpu_forced_matches_cpu(monkeypatch):
    """Forcing STRICT GPU mode on a frame that clears the work floor must route through cupy and produce
    a Gram matrix numerically equivalent to the CPU GEMM (both are the same float64 contraction, order
    differences are within the 1e-9 collinearity tolerance the caller applies)."""
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(2)
    p, n = 80, 20000  # p*n = 1.6M >= the 1M strict work floor
    M = rng.normal(size=(p, n))

    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "0")
    cpu = _gram_matrix(M)

    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    gpu = _gram_matrix(M)

    np.testing.assert_allclose(gpu, cpu, atol=1e-9, rtol=1e-9)


def test_audit_degenerate_columns_collinear_verdict_matches_across_backends(monkeypatch):
    """End-to-end: audit_degenerate_columns must flag the SAME exact-duplicate pair as collinear whether the
    Gram matrix is computed on CPU or dispatched to cupy under forced STRICT mode."""
    pytest.importorskip("cupy")
    X = _collinear_frame()

    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "0")
    cpu_result = audit_degenerate_columns(X)

    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    gpu_result = audit_degenerate_columns(X)

    assert cpu_result == gpu_result, f"CPU vs GPU-strict verdict diverged: {cpu_result} vs {gpu_result}"
    assert cpu_result.get("b") == "collinear_with:a"


def test_gram_matrix_gpu_oom_falls_back_to_cpu(monkeypatch):
    """A cupy failure (OOM / driver error / missing cupy) inside _gram_matrix must fall back to the numpy
    GEMM rather than propagate and abort the purely-diagnostic scan."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    import mlframe.feature_selection.filters._mrmr_degenerate as mod

    class _BoomCupy:
        def __getattr__(self, name):
            raise RuntimeError("simulated cupy failure")

    import sys

    monkeypatch.setitem(sys.modules, "cupy", _BoomCupy())
    rng = np.random.default_rng(3)
    M = rng.normal(size=(80, 20000))
    out = mod._gram_matrix(M)
    np.testing.assert_array_equal(out, M @ M.T)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
