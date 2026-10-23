"""The binned-aggregate feature block is sized for the pairs the fit visits, and the capped fit equals the uncapped fit restricted to those pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._binned_numeric_agg_fe import _feature_block_rows, fit_binned_numeric_agg


def test_block_rows_follow_the_visited_pairs_not_the_full_grid():
    """16 groups x 16 aggs x 4 stats would be 1024 rows; 64 given pairs need 256."""
    pairs = {(f"g{i}", f"a{j}") for i in range(8) for j in range(8)}
    assert _feature_block_rows(16, 16, 4, pairs, False) == 256
    assert _feature_block_rows(16, 16, 4, None, False) == 1024
    assert _feature_block_rows(2, 2, 4, pairs, False) == 16  # never above the full grid
    assert _feature_block_rows(16, 16, 4, pairs, True) == 0


def test_capped_fit_equals_the_uncapped_fit_on_those_pairs():
    """Sizing the block by ``pairs`` changes no stored value."""
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame(rng.standard_normal((n, 6)).astype(np.float32), columns=[f"c{i}" for i in range(6)])
    y = (X["c0"].to_numpy() + rng.standard_normal(n) > 0).astype(np.int64)
    groups, aggs = ["c0", "c1", "c2"], ["c3", "c4", "c5"]
    pairs = {("c0", "c3"), ("c1", "c4")}
    capped, _ = fit_binned_numeric_agg(X, y, group_num_cols=groups, agg_num_cols=aggs, pairs=pairs)
    full, _ = fit_binned_numeric_agg(X, y, group_num_cols=groups, agg_num_cols=aggs)
    assert 0 < capped.shape[1] < full.shape[1]
    pd.testing.assert_frame_equal(capped, full[list(capped.columns)])
