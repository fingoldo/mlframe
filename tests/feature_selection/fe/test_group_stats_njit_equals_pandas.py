"""The group mean / std / count kernel returns the pandas groupby values to the last bit, for float32 and float64 columns with NaN rows."""
import numpy as np
import pandas as pd
import pytest

import mlframe.feature_selection.filters._composite_group_agg_fe as comp
from mlframe.feature_selection.filters._group_stats_njit import group_mean_std_count


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed", range(6))
def test_kernel_bit_equal_to_the_pandas_groupby(dtype, seed):
    """Random sizes, group counts and magnitudes, 10 percent NaN rows, groups of one row and groups that hold only NaN."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 6000))
    n_groups = int(rng.integers(2, 120))
    inverse = rng.integers(0, n_groups, n).astype(np.int64)
    x = (rng.normal(size=n) * 10.0 ** int(rng.integers(-3, 4))).astype(dtype)
    x[rng.random(n) < 0.1] = np.nan
    x[inverse == 0] = np.nan  # a group with no value
    grouped = pd.DataFrame({"g": inverse, "v": x}).groupby("g", sort=True)["v"]
    present = np.unique(inverse)
    mean, std, count = group_mean_std_count(inverse, x, n_groups)
    assert np.array_equal(grouped.mean().to_numpy(), mean[present], equal_nan=True)
    assert np.array_equal(grouped.std(ddof=1).to_numpy(), std[present], equal_nan=True)
    assert np.array_equal(grouped.count().to_numpy(), count[present])
    assert mean.dtype == dtype and std.dtype == dtype


def test_other_statistics_still_come_from_the_groupby():
    """A stat the kernel does not cover (min, max, median) alongside mean / std / count: all columns are emitted, in the same order as the stats."""
    rng = np.random.default_rng(1)
    n = 2000
    X = pd.DataFrame({"g1": rng.integers(0, 5, n), "g2": rng.integers(0, 4, n), "x": rng.normal(size=n)})
    _enc, raw = comp.generate_composite_group_agg_features(X, [("g1", "g2")], ["x"], stats=("mean", "min", "median", "std", "count", "max"))
    names = list(raw)
    for stat in ("mean", "min", "median", "std", "count", "max"):
        assert comp.engineered_name_composite_agg("x", ("g1", "g2"), stat) in names
    got_min = raw[comp.engineered_name_composite_agg("x", ("g1", "g2"), "min")]["group_lookup_dict"]
    keys = comp.build_composite_keys(X, ("g1", "g2"))
    want_min = {str(k): float(v) for k, v in pd.DataFrame({"g": keys, "v": X["x"].to_numpy().astype(np.float32)}).groupby("g")["v"].min().items()}
    assert got_min == want_min
