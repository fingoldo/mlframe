"""The tiled radix quantile-edge kernel gives byte-identical edges for any tile width (columns are independent); ``tile_cols_for`` only narrows the tile so that every thread gets one."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.discretization import _quantile_radix as qr


@pytest.mark.parametrize("tile", [1, 3, 16, 20, 64, 500])
def test_every_tile_width_gives_identical_edges(tile):
    """Edges for 53 columns, with ties and NaN-free values, at several tile widths including one wider than the matrix."""
    rng = np.random.default_rng(0)
    arr = np.ascontiguousarray(rng.integers(0, 40, (3001, 53)).astype(np.float32) + rng.standard_normal((3001, 53)).astype(np.float32) * 0.01)
    q = np.linspace(0, 100, 11)[1:-1].astype(np.float64)
    ref = np.empty((q.size, 53))
    got = np.empty_like(ref)
    qr._quantile_edges_2d_radix_tiled_njit(arr, q, ref, 64)
    qr._quantile_edges_2d_radix_tiled_njit(arr, q, got, tile)
    np.testing.assert_array_equal(got, ref)


def test_the_default_tile_is_the_old_one():
    """Omitting the width keeps the constant the kernel always used."""
    arr = np.random.default_rng(1).standard_normal((500, 7)).astype(np.float32)
    q = np.array([25.0, 50.0, 75.0])
    a = np.empty((3, 7))
    b = np.empty((3, 7))
    qr._quantile_edges_2d_radix_tiled_njit(arr, q, a)
    qr._quantile_edges_2d_radix_tiled_njit(arr, q, b, qr._TILE_COLS)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("cols,threads,want", [(160, 8, 20), (1000, 8, 64), (20, 8, 16), (100, 1, 64), (1, 8, 16), (64, 8, 16)])
def test_tile_cols_for_narrows_until_every_thread_has_a_tile(cols, threads, want):
    """Never above 64, never under 16, and ``ceil(cols / threads)`` in between."""
    assert qr.tile_cols_for(cols, threads) == want
