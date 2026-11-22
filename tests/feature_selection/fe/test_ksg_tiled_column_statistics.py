"""The row-tiled column std and mean |x| of the KSG discrete-target path equal numpy's whole-block versions and hold no whole-block temporary."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from mlframe.feature_selection.filters import _ksg_cd_fast as K


@pytest.mark.parametrize("tile_elements", [97, 4_000_000])
def test_tiled_statistics_match_numpy(monkeypatch, tile_elements):
    """Several uneven tiles and a single tile both give numpy's std and mean |x| (to rounding), constant column included."""
    monkeypatch.setattr(K, "_NOISE_TILE_ELEMENTS", tile_elements)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 7)) * rng.uniform(0.1, 9.0, size=7) + rng.normal(size=7)
    X[:, 3] = 2.5
    np.testing.assert_allclose(K._column_std(X), X.std(axis=0), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(K._column_mean_abs(X), np.mean(np.abs(X), axis=0), rtol=1e-13, atol=0)


def test_no_temporary_of_the_whole_block(monkeypatch):
    """The std and the mean |x| allocate far less than one copy of the block (numpy's own versions allocate one)."""
    monkeypatch.setattr(K, "_NOISE_TILE_ELEMENTS", 20_000)
    X = np.random.default_rng(1).standard_normal((50_000, 20))
    block_bytes = X.nbytes
    tracemalloc.start()
    K._column_std(X)
    K._column_mean_abs(X)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < block_bytes // 8, (peak, block_bytes)
