"""The memoised whole-column fallback of the composite aggregation equals the direct computation and follows the column content."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _composite_group_agg_fe as C
from mlframe.feature_selection.filters._agg_stat_helpers import global_value_for_stat

STATS = ["mean", "std", "min", "max", "median", "nunique", "count", "skew"]


@pytest.mark.parametrize("stat", STATS)
def test_memo_equals_direct_value_and_follows_content(stat):
    """A second call and a copy return the direct value; a changed column returns its own value."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000).astype(np.float32)
    x[3] = np.nan
    direct = global_value_for_stat(stat, x, "composite_group_agg")
    assert C._global_value_for_stat(x, stat) == direct
    assert C._global_value_for_stat(x.copy(), stat) == direct
    y = x + 1.0
    assert C._global_value_for_stat(y, stat) == global_value_for_stat(stat, y, "composite_group_agg")
