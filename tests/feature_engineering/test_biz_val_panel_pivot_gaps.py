"""biz_value test for ``pivot_time_indexed_panel(..., add_time_gaps=True)``.

Entities have IRREGULAR (non-uniformly-spaced) observation timing. The target depends on how RECENT in
real elapsed time -- not lag-rank -- ``lag_2`` was: a big jump if the gap to "now" is small, near zero if
stale. Two entities can land on the same ``x_lag_2`` VALUE while one observed it yesterday and the other
six months ago; the lag-rank-only pivot has no column that can distinguish them, so a GBDT trained on it
can only see the row-position rank, not the elapsed time. The gap column supplies exactly that signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlframe.feature_engineering.panel_pivot import pivot_time_indexed_panel


def _make_irregular_panel(n_entities: int, seed: int):
    """Helper: Make irregular panel."""
    rng = np.random.default_rng(seed)
    rows = []
    targets = {}
    for e in range(n_entities):
        hist_len = int(rng.integers(6, 13))
        # irregular gaps: each step's real elapsed time varies a lot, uncorrelated with row-count rank.
        step_gaps = rng.uniform(1, 180, size=hist_len)
        timestamps = np.cumsum(step_gaps)
        vals = rng.normal(size=hist_len)
        for t in range(hist_len):
            rows.append({"id": e, "t": timestamps[t], "x": vals[t]})
        # target reacts to x_lag_2 ONLY if that slot's real elapsed gap to "now" is small (recent event).
        gap_lag2 = timestamps[-1] - timestamps[-3]
        recency_weight = np.exp(-gap_lag2 / 30.0)
        targets[e] = vals[-3] * 5.0 * recency_weight + rng.normal(scale=0.3)
    return pd.DataFrame(rows), pd.Series(targets)


def test_biz_val_time_gaps_beat_lag_rank_only_pivot():
    """Biz val time gaps beat lag rank only pivot."""
    df, y = _make_irregular_panel(n_entities=1200, seed=0)

    with_gaps = pivot_time_indexed_panel(df, "id", "t", ["x"], max_lags=6, add_time_gaps=True)
    rank_only = pivot_time_indexed_panel(df, "id", "t", ["x"], max_lags=6, add_time_gaps=False)

    y_aligned = y.reindex(with_gaps.index)
    train_idx, test_idx = train_test_split(with_gaps.index, test_size=0.3, random_state=0)

    def _rmse(X: pd.DataFrame) -> float:
        """Helper: Rmse."""
        Xtr, Xte = X.loc[train_idx], X.loc[test_idx]
        ytr, yte = y_aligned.loc[train_idx], y_aligned.loc[test_idx]
        model = LGBMRegressor(n_estimators=200, num_leaves=15, random_state=0, verbose=-1)
        model.fit(Xtr, ytr)
        return float(mean_squared_error(yte, model.predict(Xte)) ** 0.5)

    rmse_with_gaps = _rmse(with_gaps)
    rmse_rank_only = _rmse(rank_only)

    assert rmse_with_gaps < rmse_rank_only * 0.85, (
        f"expected gap-augmented pivot to beat lag-rank-only pivot by >=15% RMSE, got with_gaps={rmse_with_gaps:.4f} rank_only={rmse_rank_only:.4f}"
    )


def test_pivot_time_indexed_panel_gap_hand_computed():
    """Pivot time indexed panel gap hand computed."""
    df = pd.DataFrame({"id": [1, 1, 1, 2, 2], "t": [0, 5, 30, 0, 100], "x": [10, 11, 12, 20, 21]})
    out = pivot_time_indexed_panel(df, "id", "t", ["x"], max_lags=3, add_time_gaps=True)

    assert out.loc[1, "t_gap_lag_0"] == 0  # most recent row is always zero elapsed gap.
    assert out.loc[1, "t_gap_lag_1"] == 25  # 30 - 5.
    assert out.loc[1, "t_gap_lag_2"] == 30  # 30 - 0.
    assert out.loc[2, "t_gap_lag_0"] == 0
    assert out.loc[2, "t_gap_lag_1"] == 100  # 100 - 0.
    assert pd.isna(out.loc[2, "t_gap_lag_2"])  # entity 2 only has 2 rows.


def test_pivot_time_indexed_panel_no_gaps_by_default():
    """Pivot time indexed panel no gaps by default."""
    df = pd.DataFrame({"id": [1, 1], "t": [0, 5], "x": [10, 11]})
    out = pivot_time_indexed_panel(df, "id", "t", ["x"], max_lags=2)
    assert not any(c.startswith("t_gap_lag_") for c in out.columns)
