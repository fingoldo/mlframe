"""Regression: MRMR.transform on a polars frame whose base selection is empty but whose engineered recipes replay.

When every signal is folded into engineered recipes, ``support_`` is empty and the polars base frame is
``X.select([])``. That frame has height 0 on polars 2.x, so attaching the replayed engineered columns with
``with_columns`` raised ``ShapeError: can't broadcast Series ... of length 810 to length 0``. This is the
failure in CI on ``TestFeatureSelectorsWithPolarsPipeline::test_mrmr_with_polars_input_dataframe``
(polars 2.0.0). The fix builds the engineered frame directly, so its height is the replayed row count, as
the pandas branch already does.
"""

import numpy as np
import pandas as pd
import polars as pl

from mlframe.feature_selection.filters.mrmr import MRMR


def _fitted_mrmr_with_recipes_only(n_rows: int = 810):
    """Fit MRMR on pandas so that engineered recipes exist, then clear the base support to force the empty-selection path."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, 10))
    y = np.log(np.abs(X[:, 0]) + 0.1) + X[:, 2] + rng.standard_normal(1000) * 0.1
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    m = MRMR(verbose=0, max_runtime_mins=0.5)
    m.fit(df.iloc[:n_rows], y[:n_rows])
    assert len(m._engineered_recipes_) > 0, "fixture must produce engineered recipes"
    m.support_ = np.array([], dtype=np.int64)
    return m, df


def test_polars_transform_with_empty_base_selection_keeps_row_count():
    """Polars transform with recipes but zero base columns returns one row per input row, not a zero-height frame."""
    m, df = _fitted_mrmr_with_recipes_only(n_rows=810)
    pl_df = pl.from_pandas(df)
    out = m.transform(pl_df.slice(0, 810))
    assert isinstance(out, pl.DataFrame)
    assert out.height == 810
    assert out.width == len(m._engineered_recipes_)


def test_polars_transform_with_empty_base_selection_matches_pandas_values():
    """Polars and pandas transforms under an empty base selection produce the same engineered values."""
    m, df = _fitted_mrmr_with_recipes_only(n_rows=810)
    pd_out = m.transform(df.iloc[:810])
    pl_out = m.transform(pl.from_pandas(df).slice(0, 810))
    assert pl_out.height == pd_out.shape[0] == 810
    np.testing.assert_allclose(pl_out.to_numpy(), pd_out.to_numpy(), rtol=1e-9, atol=1e-12)
