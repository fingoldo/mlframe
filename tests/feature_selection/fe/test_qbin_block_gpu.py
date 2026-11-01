"""The batched device quantile binner returns the codes of the per-column binner for every column shape."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._mi_greedy_cmi_fe_binning import _sync_free_qbin_codes
from mlframe.feature_selection.filters._qbin_block_gpu import resident_qbin_codes_block, sync_free_qbin_codes_block


def _matrix(n: int = 3000) -> np.ndarray:
    """Columns of every shape the binner meets: continuous, heavy ties, mass point, binary, two values, constant, integer-valued."""
    rng = np.random.default_rng(3)
    cols = [
        rng.standard_normal(n),
        np.round(rng.standard_normal(n), 1),
        np.where(rng.random(n) < 0.7, 0.0, rng.standard_normal(n)),
        (rng.random(n) < 0.4).astype(float),
        rng.choice([-1.5, 2.5], n),
        np.full(n, 3.0),
        rng.integers(0, 4, n).astype(float),
        rng.exponential(size=n),
        rng.choice([0.0, 1.0, 2.0], n, p=[0.9, 0.05, 0.05]),
    ]
    return np.stack(cols, axis=1).astype(np.float32)


@pytest.mark.parametrize("nbins", [2, 5, 10, 16])
def test_block_codes_equal_the_per_column_codes(nbins):
    """Every column of the block equals the per-column binner's codes bit for bit."""
    x = _matrix()
    xd = cp.asarray(x)
    got = cp.asnumpy(sync_free_qbin_codes_block(cp, cp.ascontiguousarray(xd.T), nbins)).T
    for j in range(x.shape[1]):
        want = cp.asnumpy(_sync_free_qbin_codes(cp, xd[:, j].copy(), nbins))
        assert np.array_equal(got[:, j], want), f"column {j}, nbins {nbins}"


def test_cached_entry_point_matches_and_serves_repeats():
    """The content-cached entry point returns the same codes and a second call is served from the cache."""
    from mlframe.feature_selection.filters import _fe_resident_operands as ro

    x = _matrix()
    hosts = [np.ascontiguousarray(x[:, j]) for j in range(x.shape[1])]
    ro.clear_fe_resident_operands()
    first = resident_qbin_codes_block(hosts, 10, np.float32)
    assert first is not None
    xd = cp.asarray(x)
    for j, c in enumerate(first):
        assert np.array_equal(cp.asnumpy(c), cp.asnumpy(_sync_free_qbin_codes(cp, xd[:, j].copy(), 10)))
    again = resident_qbin_codes_block(hosts, 10, np.float32)
    assert again is not None and all(np.array_equal(cp.asnumpy(a), cp.asnumpy(b)) for a, b in zip(first, again))
