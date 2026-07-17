"""biz_value test for ``pivot_time_indexed_panel(..., agg_stats=...)``.

Entities have history much LONGER than ``max_lags``. The target depends on the per-entity mean level of the
OLDER statements that fall past the lag cutoff -- the most recent ``max_lags`` values are pure noise centered
at zero, uncorrelated with that mean level. A raw lag-only pivot truncates the older statements away entirely,
so it has no column that can recover the signal; ``agg_stats`` adds trailing summary columns computed over
exactly the truncated-away rows, which should let a GBDT recover it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlframe.feature_engineering.panel_pivot import pivot_time_indexed_panel

MAX_LAGS = 5


def _make_long_history_panel(n_entities: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    targets = {}
    for e in range(n_entities):
        hist_len = int(rng.integers(20, 40))  # far longer than MAX_LAGS.
        mu_e = rng.normal(scale=3.0)
        older_len = hist_len - MAX_LAGS
        older_vals = rng.normal(loc=mu_e, scale=1.0, size=older_len)
        recent_vals = rng.normal(loc=0.0, scale=1.0, size=MAX_LAGS)  # noise only, no signal about mu_e.
        vals = np.concatenate([older_vals, recent_vals])
        for t in range(hist_len):
            rows.append({"id": e, "t": t, "x": vals[t]})
        # target depends ONLY on the truncated-away older history's mean level.
        targets[e] = mu_e * 4.0 + rng.normal(scale=0.5)
    return pd.DataFrame(rows), pd.Series(targets)


def _rmse(X: pd.DataFrame, y: pd.Series, train_idx, test_idx) -> float:
    Xtr, Xte = X.loc[train_idx], X.loc[test_idx]
    ytr, yte = y.loc[train_idx], y.loc[test_idx]
    model = LGBMRegressor(n_estimators=200, num_leaves=15, random_state=0, verbose=-1)
    model.fit(Xtr, ytr)
    return float(mean_squared_error(yte, model.predict(Xte)) ** 0.5)


def test_biz_val_pivot_time_indexed_panel_agg_stats_recovers_truncated_signal():
    df, y = _make_long_history_panel(n_entities=1000, seed=0)

    raw_only = pivot_time_indexed_panel(df, "id", "t", ["x"], max_lags=MAX_LAGS)
    with_agg = pivot_time_indexed_panel(df, "id", "t", ["x"], max_lags=MAX_LAGS, agg_stats=("mean", "std", "min", "max"))

    y_aligned = y.reindex(with_agg.index)
    train_idx, test_idx = train_test_split(with_agg.index, test_size=0.3, random_state=0)

    rmse_raw = _rmse(raw_only, y_aligned, train_idx, test_idx)
    rmse_agg = _rmse(with_agg, y_aligned, train_idx, test_idx)

    assert rmse_agg < rmse_raw * 0.5, (
        f"expected agg_stats-augmented pivot to beat raw-only pivot by >=50% RMSE (truncated history carries "
        f"the target signal), got with_agg={rmse_agg:.4f} raw_only={rmse_raw:.4f}"
    )


def test_pivot_time_indexed_panel_agg_stats_hand_computed():
    df = pd.DataFrame({"id": [1] * 8 + [2] * 2, "t": list(range(8)) + list(range(2)), "x": list(range(100, 108)) + [20, 21]})
    out = pivot_time_indexed_panel(df, "id", "t", ["x"], max_lags=3, agg_stats=("mean", "std", "min", "max"))

    # entity 1: 8 rows, max_lags=3 keeps lag_0..2 (values 107,106,105); truncated-away are 100..104.
    assert out.loc[1, "x_trail_mean"] == 102.0
    assert out.loc[1, "x_trail_min"] == 100.0
    assert out.loc[1, "x_trail_max"] == 104.0
    # entity 2 only has 2 rows -- nothing truncated away, so trailing stats are NaN.
    assert pd.isna(out.loc[2, "x_trail_mean"])


def test_pivot_time_indexed_panel_agg_stats_default_unchanged():
    df = pd.DataFrame({"id": [1] * 8 + [2] * 2, "t": list(range(8)) + list(range(2)), "x": list(range(100, 108)) + [20, 21]})
    baseline = pivot_time_indexed_panel(df, "id", "t", ["x"], max_lags=3)
    with_none = pivot_time_indexed_panel(df, "id", "t", ["x"], max_lags=3, agg_stats=None)
    pd.testing.assert_frame_equal(baseline, with_none)
    assert not any(c.startswith("x_trail_") for c in baseline.columns)
