"""Regression test: per_column_rff fused-njit kernel matches the prior numpy broadcast path.

The fused kernel removes the (n, d_input, m) angles/cos_part/sin_part temporaries + Python copy
loop (2.3-6.8x; bench_per_column_rff_fused_njit.py). It is NOT bit-identical to the old numpy
float32 cos/sin ufunc — numba's np.cos/np.sin float32 path differs by a single ULP (~3e-8). This
test pins:
  1. output shape / column names / layout unchanged,
  2. the divergence from the reference numpy path is within float32 single-ULP (<= 1e-6), i.e.
     selection-equivalent for any downstream boosting (never the ~1e-3 that could flip a split),
so a future "rewrite the kernel" cannot silently introduce a real numeric drift.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer.per_column_rff import compute_per_column_rff


def _reference_numpy(X_std, W, b, m, dtype=np.float32):
    """Exact replica of the pre-fix per_column_rff numpy body (the OLD path)."""
    n, d_input = X_std.shape
    angles = X_std[:, :, None] * W[None, :, :] + b[None, :, :]
    scale = float(np.sqrt(1.0 / m))
    cos_part = (scale * np.cos(angles)).astype(dtype, copy=False)
    sin_part = (scale * np.sin(angles)).astype(dtype, copy=False)
    out = np.empty((n, d_input * 2 * m), dtype=dtype)
    for j in range(d_input):
        out[:, j * 2 * m : j * 2 * m + m] = cos_part[:, j, :]
        out[:, j * 2 * m + m : (j + 1) * 2 * m] = sin_part[:, j, :]
    return out


@pytest.mark.parametrize("n,d_input,m", [(500, 8, 4), (1000, 20, 4), (300, 5, 8)])
def test_per_column_rff_fused_njit_selection(n, d_input, m):
    rng = np.random.default_rng(123)
    X = rng.standard_normal((n, d_input)).astype(np.float32)

    df = compute_per_column_rff(X, seed=7, d_embed_per_column=m, standardize=False)
    got = df.to_numpy()

    # Reconstruct the reference using the SAME W/b the function draws (seed + draw order).
    rs = np.random.default_rng(7)
    W = (rs.standard_normal((d_input, m)) / 1.0).astype(np.float32)
    b = (rs.uniform(0, 2.0 * np.pi, size=(d_input, m))).astype(np.float32)
    ref = _reference_numpy(X.astype(np.float32), W, b, m)

    assert got.shape == ref.shape == (n, d_input * 2 * m)
    # Column-name / interleaved-layout contract preserved.
    assert df.columns[0] == "pcrff_c0_cos0"
    assert df.columns[m] == "pcrff_c0_sin0"
    # Single-ULP float32 divergence at most — selection-equivalent, never ~1e-3.
    max_abs = float(np.abs(got.astype(np.float64) - ref.astype(np.float64)).max())
    assert max_abs <= 1e-6, f"divergence {max_abs:.2e} exceeds float32 single-ULP band"


def test_per_column_rff_fused_njit_dtype_gate():
    """Non-float32 dtype path still produces the requested dtype + correct shape."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6)).astype(np.float32)
    df = compute_per_column_rff(X, seed=1, d_embed_per_column=4, standardize=False, dtype=np.float64)
    assert df.to_numpy().dtype == np.float64
    assert df.shape == (200, 6 * 2 * 4)
