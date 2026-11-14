"""The float32-edge tiled binning kernel writes the same codes as the double-compare kernel, values on and next to the edges included."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._gpu_resident_discretize import _searchsorted_codes


def _codes(monkeypatch, cand, edges, out_dtype, flag):
    """Codes with the float32-edge kernel on (``flag="1"``) or off."""
    monkeypatch.setenv("MLFRAME_FE_BIN_CODES_F32EDGES", flag)
    return _searchsorted_codes(cand, edges, out_dtype=out_dtype)


@pytest.mark.parametrize("n,k,nbins", [(3000, 97, 10), (1000, 33, 4), (777, 1, 20), (500, 65, 64)])
@pytest.mark.parametrize("out_dtype", [np.int8, np.int32])
def test_codes_equal_the_double_compare_kernel(monkeypatch, n, k, nbins, out_dtype):
    """Ties (half-step grid), edges equal to a value, edges a hair above a value (rounded up to float32), +-inf edges, a ragged last tile."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((n, k)).astype(np.float32)
    x[:, ::3] = np.round(x[:, ::3] * 2) / 2
    edges = np.quantile(x.astype(np.float64), np.linspace(0, 1, nbins + 1)[1:-1], axis=0)
    edges[0] = x[5].astype(np.float64)
    if edges.shape[0] > 2:
        edges[1] = np.maximum(edges[1], x[7].astype(np.float64) + 1e-12)
        edges[-1, 0] = np.inf
    edges = np.sort(edges, axis=0)
    cand, ed = cp.asarray(x), cp.asarray(edges)
    od = np.dtype(out_dtype)
    assert cp.array_equal(_codes(monkeypatch, cand, ed, od, "1"), _codes(monkeypatch, cand, ed, od, "0"))


@pytest.mark.parametrize("flag", ["1", "0"])
def test_codes_written_in_place_into_a_column_slice(monkeypatch, flag):
    """``out`` as a column slice of a wider buffer is filled at its row stride (both kernels) and the other columns stay untouched."""
    rng = np.random.default_rng(3)
    n, k, width, start = 1500, 37, 100, 21
    x = cp.asarray(rng.standard_normal((n, k)).astype(np.float32))
    edges = cp.asarray(np.quantile(cp.asnumpy(x).astype(np.float64), np.linspace(0, 1, 11)[1:-1], axis=0))
    monkeypatch.setenv("MLFRAME_FE_BIN_CODES_F32EDGES", flag)
    ref = cp.asnumpy(_searchsorted_codes(x, edges, out_dtype=np.int8))
    buf = cp.full((n, width), -7, dtype=cp.int8)
    got = _searchsorted_codes(x, edges, out=buf[:, start : start + k])
    assert got.data.ptr == buf[:, start : start + k].data.ptr
    host = cp.asnumpy(buf)
    assert np.array_equal(host[:, start : start + k], ref)
    assert (host[:, :start] == -7).all() and (host[:, start + k :] == -7).all()
