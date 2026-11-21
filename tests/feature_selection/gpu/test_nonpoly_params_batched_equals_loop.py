"""The vectorised parameter build of the non-polynomial cells gives the per-degree loop's cells bit for bit."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters.hermite_fe import _nonpoly_cells_gpu as np_gpu


def _matrix(n=4000, k=12, seed=0, dtype=np.float32):
    """A finite matrix with ordinary columns, a constant column and a near-constant one (the degenerate branches)."""
    r = np.random.default_rng(seed)
    m = r.standard_normal((n, k)).astype(dtype) * r.uniform(0.5, 5.0, k).astype(dtype)
    m[:, 0] = 3.0
    m[:, 1] = 1.0 + 1e-12 * r.standard_normal(n)
    return cp.asarray(m)


@pytest.mark.parametrize("basis", sorted(np_gpu.NONPOLY_BASES))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_params_equal_the_loop(basis, dtype, monkeypatch):
    """Same cells and labels for every basis, both dtypes, with degenerate columns."""
    m = _matrix(dtype=dtype)
    degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    monkeypatch.setenv("MLFRAME_NONPOLY_PARAMS_LOOP", "0")
    got = np_gpu.nonpoly_block_gpu(cp, m, basis, degrees)
    monkeypatch.setenv("MLFRAME_NONPOLY_PARAMS_LOOP", "1")
    want = np_gpu.nonpoly_block_gpu(cp, m, basis, degrees)
    assert (got is None) == (want is None)
    if got is None:
        return
    assert got[1] == want[1]
    np.testing.assert_array_equal(cp.asnumpy(got[0]), cp.asnumpy(want[0]))
