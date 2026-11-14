"""The per-stratum counting of the permutation null equals the dense batched CMI on the same within-stratum shuffles.

The ``(x, y, z)`` joint of a shuffled column is counted per stratum in shared memory (z-sorted rows, one thread per stratum) instead of through global atomics into a ``Kx * Kyz`` table;
the entropies, occupied-cell counts and so the CMI must agree with ``_batched_cmi_resident_chunked`` to rounding.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters import _cmi_null_strata_gpu as G
from mlframe.feature_selection.filters import _fe_cmi_perm_null_gpu as m


def _shuffles(n=6000, kz=700, kx=8, ky=3, nperm=9, seed=0):
    """Device inputs: support, target, candidate, z-sorted layout and ``nperm`` within-stratum shuffles of the candidate."""
    rng = np.random.default_rng(seed)
    z, x, y = rng.integers(0, kz, n), rng.integers(0, kx, n), rng.integers(0, ky, n)
    z_d, x_d, y_d = cp.asarray(z), cp.asarray(x), cp.asarray(y)
    z_for_cmi, order_d, z_rank_d = m._stratum_layout(cp, z_d, None, None, n)
    xs = x_d[order_d]
    flat = m._flat_within_stratum_shuffle(cp, cp.random.RandomState(seed + 1), xs, z_rank_d, nperm, kx, int(z_rank_d[-1, 0]) + 1)
    return z_d, x_d, y_d, y, z_for_cmi, order_d, z_rank_d, xs, flat


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("variant", ["direct", "rows", None])
@pytest.mark.parametrize("kz,kx,ky", [(700, 8, 3), (3000, 10, 2), (260, 4, 5), (6000, 8, 8)])
def test_strata_counting_equals_the_dense_batched_cmi(kz, kx, ky, variant, seed):
    """Observed shapes (small and large strata counts, tall and wide ``(x, y)`` tables), both kernels (direct, rows, and the shape rule), two datasets."""
    _z, _x, y_d, y_host, z_for_cmi, order_d, z_rank_d, xs, flat = _shuffles(kz=kz, kx=kx, ky=ky, seed=seed)
    got = G.strata_null_cmi(cp, flat, y_d[order_d], z_rank_d[:, 0], z_for_cmi, y_d, kx, order_d, xs, variant=variant)
    assert got is not None
    n, nperm = int(xs.shape[0]), int(flat.shape[0])
    xp = cp.empty((n, nperm), dtype=cp.int64)
    xp[order_d, :] = flat.T
    want = m._batched_cmi_resident_chunked(xp, y_host, z_for_cmi, shared_xz=True)
    assert np.allclose(cp.asnumpy(got), cp.asnumpy(want), rtol=0, atol=1e-12)


def test_routes_that_do_not_apply_return_none():
    """Too few strata (the shared-memory joint count is fast there) and a table past the private histogram size both decline, so the dense route runs."""
    _z, _x, y_d, _yh, z_for_cmi, order_d, z_rank_d, xs, flat = _shuffles(kz=40)
    assert G.strata_null_cmi(cp, flat, y_d[order_d], z_rank_d[:, 0], z_for_cmi, y_d, 8, order_d, xs) is None
    _z, _x, y_d, _yh, z_for_cmi, order_d, z_rank_d, xs, flat = _shuffles(kz=700, kx=10, ky=8)
    assert G.strata_null_cmi(cp, flat, y_d[order_d], z_rank_d[:, 0], z_for_cmi, y_d, 10, order_d, xs) is None
