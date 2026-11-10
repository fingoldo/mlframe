"""The nogil member-rank kernel of the grouped-quantile family gives bit-identical ranks to its numpy twin."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _grouped_quantile_fe as G


def _cases():
    """Group value vectors covering ties, constants, NaN, infinities and single members."""
    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 37, 400, 5000):
        yield rng.standard_normal(n).astype(np.float32)
        yield rng.integers(0, 3, n).astype(np.float32)  # heavy ties
        yield np.zeros(n, dtype=np.float32)  # one run
        yield np.where(rng.random(n) < 0.2, np.nan, rng.standard_normal(n)).astype(np.float32)
        yield np.where(rng.random(n) < 0.1, np.inf, rng.standard_normal(n)).astype(np.float32)
        yield np.where(rng.random(n) < 0.1, -np.inf, rng.integers(0, 2, n)).astype(np.float32)


@pytest.mark.parametrize("vals", list(_cases()))
def test_kernel_ranks_equal_the_numpy_twin(vals):
    """Every member's rank matches the twin bit for bit, non-finite members included."""
    finite = np.isfinite(vals)
    fin = vals[finite]
    if fin.size == 0:
        pytest.skip("no finite member: the caller never reaches the rank path")
    order = np.argsort(fin, kind="stable")
    sv = fin[order]
    got = G._member_pct_ranks(order, sv, vals, finite)
    want = G._member_pct_ranks_numpy(order, sv, vals, finite)
    assert np.array_equal(got, want, equal_nan=True), np.abs(got - want).max()
    assert got.dtype == want.dtype
