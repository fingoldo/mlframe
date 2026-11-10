"""A categorisation served from the edge memo gives the codes of a cold one, and a column with other values or labels is not served from it."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._adaptive_nbins_columns import clear_edge_memo
from mlframe.feature_selection.filters.mrmr import categorize_dataset


def _frame(n=4000, seed=0):
    """A frame of six Gaussian columns and a binary label that depends on the first two."""
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, 6))
    df = pd.DataFrame(x, columns=[f"x{i}" for i in range(6)])
    y = (x[:, 0] * x[:, 1] + 0.3 * r.standard_normal(n) > 0).astype(np.int64)
    return df, y


def _cat(df, y):
    """``categorize_dataset`` with the MDLP strategy and the label ``y``."""
    return categorize_dataset(df=df, method="quantile", n_bins=10, dtype=np.int16, missing_strategy="separate_bin", nbins_strategy="mdlp", nbins_strategy_kwargs=None, y_for_strategy=y)


def test_memo_hit_gives_the_cold_codes():
    """Cold run, then a warm-up on the raw frame, then the real run on the frame with extra columns: same codes for every shared column."""
    df, y = _frame()
    clear_edge_memo()
    cold_data, cold_cols, cold_nb = _cat(df.assign(targ_0=y), y)
    clear_edge_memo()
    _cat(df, y)  # the warm-up pass over the raw frame
    warm_data, warm_cols, warm_nb = _cat(df.assign(targ_0=y), y)
    assert list(warm_cols) == list(cold_cols)
    np.testing.assert_array_equal(warm_data, cold_data)
    np.testing.assert_array_equal(warm_nb, cold_nb)


def test_other_labels_are_not_served_from_the_memo():
    """The same columns under permuted labels are searched again (the supervised edges differ)."""
    df, y = _frame()
    clear_edge_memo()
    _cat(df, y)
    y2 = np.random.default_rng(1).permutation(y)
    data_memo, _, _ = _cat(df.assign(targ_0=y2), y2)
    clear_edge_memo()
    data_cold, _, _ = _cat(df.assign(targ_0=y2), y2)
    np.testing.assert_array_equal(data_memo, data_cold)
