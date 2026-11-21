"""The Kraskov noise added in row tiles gives the values of the single (n, p) draw, bit for bit."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _ksg_cd_fast as K


def _reference(X, y, n_neighbors, random_state):
    """The pre-tiling preparation: one (n, p) draw, one product, one add, then the kernel."""
    X = np.array(X, dtype=np.float64, copy=True)
    n = X.shape[0]
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    X /= std
    means = np.maximum(1.0, np.mean(np.abs(X), axis=0))
    rng = np.random.RandomState(int(random_state))
    X += 1e-10 * means * rng.standard_normal(size=(n, X.shape[1]))
    _, d = np.unique(np.asarray(y).ravel(), return_inverse=True)
    d = np.ascontiguousarray(d, dtype=np.int64)
    return K._mi_cd_columns_auto(np.ascontiguousarray(X), d, int(d.max()) + 1, int(n_neighbors))


@pytest.mark.parametrize("tile_elements", [4_000_000, 7000, 1000])
def test_tiled_noise_is_the_single_draw(monkeypatch, tile_elements):
    """Tiles of 1000 / 7000 elements (uneven tile rows) and the default give the MI of the one-draw reference exactly."""
    monkeypatch.setattr(K, "_NOISE_TILE_ELEMENTS", tile_elements)
    rng = np.random.default_rng(0)
    n, p = 3000, 9
    X = rng.standard_normal((n, p)).astype(np.float32)
    X[:, 2] = 5.0  # a constant column: std replaced by 1
    y = (X[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(int)
    got = K.ksg_mi_discrete_target(X, y, n_neighbors=3, random_state=7)
    ref = _reference(X, y, 3, 7)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("block_bytes", [128 * 1024 * 1024, 8 * 3000 * 4, 8 * 3000 * 1])
def test_column_blocks_give_the_same_mi_as_one_block(monkeypatch, block_bytes):
    """The per-column KSG over blocks of 4 columns, of 1 column and in one block: the same vector, bit for bit."""
    rng = np.random.default_rng(3)
    n, p = 3000, 11
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.standard_normal(n) > 0).astype(int)
    monkeypatch.setattr(K, "_COLUMNS_BLOCK_BYTES", 10**12)
    one = K.ksg_mi_discrete_target_columns(X, y, n_neighbors=3, random_state=5)
    monkeypatch.setattr(K, "_COLUMNS_BLOCK_BYTES", block_bytes)
    blocked = K.ksg_mi_discrete_target_columns(X, y, n_neighbors=3, random_state=5)
    np.testing.assert_array_equal(blocked, one)
    assert blocked.shape == (p,)
