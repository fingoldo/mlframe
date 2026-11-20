"""The one-launch pool fingerprints equal the per-candidate chain (astype, + 1, * weights, sum) for every integer code dtype."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _bin_fingerprint_gpu as fp
from mlframe.feature_selection.filters._mi_greedy_cmi_fe_steps import _fingerprint_weights


def _chain(cols):
    """The per-candidate reference: weighted wrap-around int64 sum of the codes plus one."""
    n = cols[0].size
    w = _fingerprint_weights(cp, n)
    return [int(v) for v in cp.asnumpy(cp.stack([cp.sum((c.astype(cp.int64, copy=False) + 1) * w, dtype=cp.int64) for c in cols])).tolist()]


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("n_k", [(1000, 5), (7, 3), (20000, 12)])
def test_pool_fingerprints_equal_chain(dtype, n_k):
    """Several candidates of every integer width, with a row count below and above the block width, fingerprint identically."""
    n, k = n_k
    rng = np.random.default_rng(0)
    cols = [cp.asarray(rng.integers(0, 20, n).astype(dtype)) for _ in range(k)]
    assert fp.pool_fingerprints(cp, cols) == _chain(cols)


def test_single_column_pool():
    """A pool of one candidate is fingerprinted as the chain does."""
    col = cp.asarray(np.random.default_rng(2).integers(0, 9, 500).astype(np.int16))
    assert fp.pool_fingerprints(cp, [col]) == _chain([col])
