"""Float32 outputs no longer come from float32 accumulation, and tiny sign changes are still counted (tracker 2026-11-02 wave 15, DTYPE15-1, DTYPE15-3, DTYPE15-4)."""

import numpy as np

from mlframe.feature_engineering._numerical_counts import compute_ncrossings
from mlframe.feature_engineering.grouped import per_group_rolling_reduce


def test_float32_rolling_mean_is_accurate_on_a_long_group():
    """200k rows near 100.37 in one group: the float32 prefix sum reached 2e7 where float32 steps by 2, so window means drifted by ~0.2%."""
    rng = np.random.default_rng(0)
    v = 100.37 + rng.normal(scale=0.01, size=200_000)
    g = np.zeros(v.size, dtype=np.int64)
    got = per_group_rolling_reduce(v, g, 5, "mean", output_dtype=np.float32)
    ref = np.convolve(v, np.ones(5) / 5, mode="valid")
    np.testing.assert_allclose(got[4:], ref, rtol=2e-6)


def test_tiny_alternating_differences_still_cross():
    """Differences of +-1e-24 around the mark: their float32 product is 1e-48, which underflowed to 0 and counted no crossing."""
    mark = 1.0
    arr = np.array([mark + 1e-24 if i % 2 == 0 else mark - 1e-24 for i in range(10)], dtype=np.float64)
    arr = (arr - mark) * 1.0 + mark  # keep float64; the +-1e-24 vanishes at 1.0, so use a mark of 0 instead
    arr = np.array([1e-24 if i % 2 == 0 else -1e-24 for i in range(10)])
    assert compute_ncrossings(arr, np.array([0.0]))[0] == 9
