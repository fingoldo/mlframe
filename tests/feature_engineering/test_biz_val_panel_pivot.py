"""biz_value test for ``feature_engineering.panel_pivot.pivot_time_indexed_panel``.

Source: 5th_amex-default-prediction.md -- "Pivot: Combine all features horizontally" (reshape statement-level
panel rows into one wide row per customer). Plain pandas ``.pivot()`` left-aligns by absolute time_step value,
so entities with different history lengths get their MOST RECENT value in different columns -- a GBDT split
on a fixed column position then sees an inconsistent signal across entities. Right-alignment (this module) puts
the most recent value in the same column (``lag_0``) for every entity regardless of history length, which
should make a "depends only on the most recent value" target far easier to learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlframe.feature_engineering.panel_pivot import pivot_time_indexed_panel


def _make_variable_length_panel(n_entities: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    targets = {}
    for e in range(n_entities):
        hist_len = rng.integers(3, 13)
        vals = rng.normal(size=hist_len).cumsum() + 50
        for t in range(hist_len):
            rows.append({"id": e, "t": t, "x": vals[t]})
        targets[e] = vals[-1] * 2 + rng.normal(scale=0.5)  # target depends only on the MOST RECENT value.
    return pd.DataFrame(rows), pd.Series(targets)


def test_biz_val_right_aligned_pivot_beats_left_aligned_pivot():
    df, y = _make_variable_length_panel(n_entities=800, seed=0)

    right_aligned = pivot_time_indexed_panel(df, "id", "t", ["x"], max_lags=13)
    left_aligned = df.pivot(index="id", columns="t", values="x")
    left_aligned.columns = [f"x_t{c}" for c in left_aligned.columns]

    y_aligned = y.reindex(right_aligned.index)
    train_idx, test_idx = train_test_split(right_aligned.index, test_size=0.3, random_state=0)

    def _rmse(X: pd.DataFrame) -> float:
        Xtr, Xte = X.loc[train_idx], X.loc[test_idx]
        ytr, yte = y_aligned.loc[train_idx], y_aligned.loc[test_idx]
        model = LGBMRegressor(n_estimators=200, num_leaves=15, random_state=0, verbose=-1)
        model.fit(Xtr, ytr)
        return float(mean_squared_error(yte, model.predict(Xte)) ** 0.5)

    rmse_right = _rmse(right_aligned)
    rmse_left = _rmse(left_aligned)

    assert rmse_right < rmse_left * 0.7, (
        f"expected right-aligned pivot to beat left-aligned pivot by >=30% RMSE, got right={rmse_right:.4f} left={rmse_left:.4f}"
    )


def test_pivot_time_indexed_panel_right_alignment_hand_computed():
    df = pd.DataFrame({"id": [1, 1, 1, 2, 2], "t": [0, 1, 2, 0, 1], "x": [10, 11, 12, 20, 21]})
    out = pivot_time_indexed_panel(df, "id", "t", ["x"], max_lags=3)

    assert out.loc[1, "x_lag_0"] == 12  # entity 1's most recent value.
    assert out.loc[2, "x_lag_0"] == 21  # entity 2's most recent value -- SAME column as entity 1's.
    assert out.loc[1, "x_lag_2"] == 10
    assert pd.isna(out.loc[2, "x_lag_2"])  # entity 2 only has 2 rows -- lag_2 is unavailable.


def test_pivot_time_indexed_panel_truncates_to_max_lags():
    df = pd.DataFrame({"id": [1] * 5, "t": range(5), "x": range(100, 105)})
    out = pivot_time_indexed_panel(df, "id", "t", ["x"], max_lags=3)
    assert list(out.columns) == ["x_lag_0", "x_lag_1", "x_lag_2"]
    assert out.loc[1, "x_lag_0"] == 104  # most recent of the 5 rows.
