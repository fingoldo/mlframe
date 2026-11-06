"""The four-loads-in-flight f32 radix select (v5) gives the order statistics of the compaction kernel (v4) bit for bit, for every row count and column shape."""

from __future__ import annotations

import numpy as np
import pytest

from tests._cuda_require import require_cuda_device


def _edges(cp, X, monkeypatch, v5: bool):
    """The interior quantile edges of ``X`` through the radix select with the v5 switch set."""
    from mlframe.feature_selection.filters import _gpu_resident_select as sel

    monkeypatch.setenv("MLFRAME_RADIX_F32_V5", "1" if v5 else "0")
    return cp.asnumpy(sel._radix_select_interior_edges(X.copy(), 10))


@pytest.mark.parametrize("n", [97, 1000, 4097, 20011, 100003])
@pytest.mark.parametrize("kind", ["normal", "uniform", "ties", "constant", "mixed_sign_zero"])
def test_v5_equals_v4(monkeypatch, n, kind):
    """Row counts that are and are not multiples of the four-element stride, and columns with many ties, a single value and signed zeros."""
    require_cuda_device()
    import cupy as cp

    rng = np.random.default_rng(n)
    k = 6
    if kind == "normal":
        raw = rng.standard_normal((n, k))
    elif kind == "uniform":
        raw = rng.random((n, k))
    elif kind == "ties":
        raw = np.round(rng.standard_normal((n, k)) * 3)
    elif kind == "constant":
        raw = np.full((n, k), 2.5)
    else:
        raw = np.where(rng.random((n, k)) < 0.3, 0.0, rng.standard_normal((n, k)))
        raw[::7] *= -1.0
        raw[raw == 0.0] = np.where(rng.random((raw == 0.0).sum()) < 0.5, -0.0, 0.0)
    X = cp.asarray(raw.astype(np.float32))
    ref = _edges(cp, X, monkeypatch, False)
    new = _edges(cp, X, monkeypatch, True)
    assert ref.shape == new.shape and ref.size > 0
    assert np.array_equal(ref, new, equal_nan=True)
