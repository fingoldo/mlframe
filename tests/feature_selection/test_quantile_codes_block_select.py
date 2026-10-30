"""``quantile_codes_block`` reads only the order statistics its edges need (a multi-select) and must return the codes the full-sort form returns."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._numba_utils_cross_module import quantile_codes_block


def _reference(col: np.ndarray, n_bins: int) -> tuple[np.ndarray, int]:
    """Codes from numpy quantile edges: distinct values when there are at most ``n_bins`` of them, else the clipped edge bin."""
    levels = np.unique(col)
    if levels.size <= n_bins:
        return np.searchsorted(levels, col, side="left"), int(levels.size)
    raw = np.quantile(col, np.append(np.arange(n_bins) / n_bins, 1.0))
    edges = raw[np.r_[True, np.diff(raw) != 0]]
    codes = np.clip(np.searchsorted(edges, col, side="right") - 1, 0, max(edges.size - 2, 0))
    return codes, int(codes.max()) + 1


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("kind", ["normal", "ties", "lowcard", "constant", "heavy_tie", "skew", "tiny"])
@pytest.mark.parametrize("n_bins", [2, 10, 16])
def test_codes_equal_the_numpy_edge_reference(kind, n_bins, dtype):
    """Every column kind, bin count and dtype gives the reference codes and occupied-bin counts."""
    rng = np.random.default_rng(0)
    n = 13 if kind == "tiny" else 5000
    v = {
        "normal": rng.standard_normal((4, n)),
        "ties": np.round(rng.standard_normal((4, n)), 1),
        "lowcard": rng.integers(0, 7, (4, n)).astype(np.float64),
        "constant": np.ones((4, n)),
        "heavy_tie": np.where(rng.random((4, n)) < 0.6, 0.0, rng.standard_normal((4, n))),
        "skew": rng.lognormal(size=(4, n)),
        "tiny": rng.standard_normal((4, n)),
    }[kind].astype(dtype)
    codes = np.empty(v.shape, dtype=np.int16)
    nx = np.empty(v.shape[0], dtype=np.int64)
    quantile_codes_block(v, n_bins, codes, nx)
    for j in range(v.shape[0]):
        want, want_nx = _reference(v[j].astype(np.float64), n_bins)
        assert np.array_equal(codes[j], want), (kind, n_bins, j)
        assert nx[j] == want_nx
