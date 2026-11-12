"""The global-scratch joint entropy equals the unfused count+reduce reference on both of its routes.

Joints up to ``_JOINT_ENTROPY_GLOBAL_TILE_MAX_CELLS`` run the tiled kernel (column groups of 8 for small joints, one column per block above
``_JOINT_ENTROPY_GLOBAL_G8_MAX_CELLS``); larger joints run the untiled kernel. Both routes are checked, with column counts that leave a partial group.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters import _fe_batched_mi_cmi as C


@pytest.mark.parametrize("route", ["tiled", "untiled"])
@pytest.mark.parametrize("K,Kb", [(1, 2), (9, 7), (13, 40)])
def test_global_entropy_equals_count_reduce(route, K, Kb, monkeypatch):
    """Entropies agree to round-off and the occupied-cell counts agree exactly with the unfused reference, on the route named."""
    cap = 6144 if route == "tiled" else 0
    monkeypatch.setattr(C, "_JOINT_ENTROPY_GLOBAL_TILE_MAX_CELLS", cap)
    rng = np.random.default_rng(K * 100 + Kb)
    n, Kx = 4000, 10
    X = cp.asarray(rng.integers(0, Kx, size=(n, K), dtype=np.int64))
    b = cp.asarray(rng.integers(0, Kb, size=n, dtype=np.int64))
    h, k = C.batched_joint_entropy_global(X, b, Kx, Kb, 1.0 / n, 3)
    ref_h, ref_k = C._rows_entropy_and_k(C._batched_joint_counts2(X, b, Kx, Kb).astype(cp.float64), 1.0 / n)
    assert float(cp.abs(h - ref_h).max()) < 1e-12
    assert cp.array_equal(k, ref_k)
