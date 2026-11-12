"""The tiled coalesced ``binned_mi_from_codes_gpu`` path equals the host plug-in MI for every column width.

The tile groups G columns per block, so K values that G does not divide leave a partial last group; that remainder is what this covers.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters import _fe_batched_mi as M


def _host_mi(codes: np.ndarray, y: np.ndarray, kx: int, ky: int) -> np.ndarray:
    """Plug-in MI (nats) per column from the joint counts."""
    n = len(y)
    out = np.zeros(codes.shape[1])
    for k in range(codes.shape[1]):
        joint = np.zeros((kx, ky))
        np.add.at(joint, (codes[:, k], y), 1.0)
        pxy = joint / n
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        nz = pxy > 0
        out[k] = float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())
    return out


@pytest.mark.parametrize("n_cols", [1, 5, 8, 13, 100])
def test_tiled_path_matches_host_plugin_mi(n_cols):
    """The public call returns the host plug-in MI for a column count of ``n_cols``."""
    rng = np.random.default_rng(n_cols)
    n, kx, ky = 4000, 7, 3
    codes = rng.integers(0, kx, size=(n, n_cols)).astype(np.int64)
    y = rng.integers(0, ky, size=n).astype(np.int64)
    got = M.binned_mi_from_codes_gpu(codes, y, ky=ky, codes_trusted=True)
    assert got.shape == (n_cols,)
    np.testing.assert_allclose(got, _host_mi(codes, y, kx, ky), rtol=1e-10, atol=1e-12)


def test_tiled_width_respects_shared_cap():
    """The tile width is 0 when one histogram exceeds the shared cap, and otherwise between 1 and the tile maximum."""
    cells = M._MI_FROM_CODES_MAX_SHARED // 4 + 1
    assert M._mi_from_codes_tile_width(cells) == 0
    assert 1 <= M._mi_from_codes_tile_width(40) <= M._MI_FROM_CODES_TILE_COLS
