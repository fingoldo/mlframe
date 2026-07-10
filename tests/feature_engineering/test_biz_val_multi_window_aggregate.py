"""biz_value test for ``feature_engineering.multi_window_aggregate``.

The win: a genuine behavioral shift concentrated in the RECENT window (e.g. a worsening trend) is diluted
away by a single all-history aggregate but clearly visible in a short-lookback window -- so a classifier
using the short-window aggregate as a feature should beat one using only the all-history aggregate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from mlframe.feature_engineering.multi_window_aggregate import multi_window_aggregate


def _make_data(seed: int):
    rng = np.random.default_rng(seed)
    n_entities = 400
    # a long history (~4 years) so a small recent-90-day shift is heavily diluted in the all-history mean,
    # but still clearly visible in a 90-day window -- the realistic scenario the source technique targets.
    cutoff = 1500.0
    rows = []
    labels = {}
    for entity_id in range(n_entities):
        worsening = rng.random() < 0.5
        labels[entity_id] = int(worsening)
        n_events = rng.integers(80, 120)
        times = np.sort(rng.uniform(0, cutoff - 1, n_events))
        if worsening:
            # a modest elevated level ONLY in the last 90 days; earlier history is identical to the stable class.
            values = np.where(times > cutoff - 90, rng.normal(7, 1.5, n_events), rng.normal(5, 1.5, n_events))
        else:
            values = rng.normal(5, 1.5, n_events)
        for t, v in zip(times, values):
            rows.append({"entity": entity_id, "t": float(t), "amount": float(v)})

    history_df = pd.DataFrame(rows)
    query_df = pd.DataFrame({"entity": list(labels.keys()), "as_of": cutoff})
    y = np.array([labels[e] for e in query_df["entity"]])
    return history_df, query_df, y


def test_biz_val_multi_window_aggregate_recent_window_beats_all_history_aggregate():
    history_df, query_df, y = _make_data(seed=0)

    result = multi_window_aggregate(
        history_df, entity_col="entity", time_col="t", as_of=query_df,
        agg_funcs={"amount": ["sum", "count", "mean"]}, lookback_horizons=[90, 10_000],
    )

    recent_mean = result[["amount_mean_last_90"]].fillna(0.0)
    all_history_mean = result[["amount_mean_last_10000"]].fillna(0.0)

    auc_recent = cross_val_score(LogisticRegression(), recent_mean, y, cv=5, scoring="roc_auc").mean()
    auc_all_history = cross_val_score(LogisticRegression(), all_history_mean, y, cv=5, scoring="roc_auc").mean()

    assert auc_recent > auc_all_history + 0.2, (
        f"the recent-window aggregate should detect the concentrated shift far better than the all-history aggregate: "
        f"recent={auc_recent:.4f} all_history={auc_all_history:.4f}"
    )
    assert auc_recent > 0.9


def test_multi_window_aggregate_matches_manual_windowed_sum():
    history_df = pd.DataFrame({"entity": ["a"] * 5, "t": [1, 5, 10, 15, 20], "amount": [10.0, 20.0, 30.0, 40.0, 50.0]})
    query_df = pd.DataFrame({"entity": ["a"], "as_of": [21]})

    result = multi_window_aggregate(
        history_df, entity_col="entity", time_col="t", as_of=query_df, agg_funcs={"amount": ["sum", "count", "mean"]}, lookback_horizons=[10, 100]
    )
    row = result.iloc[0]
    assert row["amount_sum_last_10"] == 90.0  # rows at t=15,20 (window [11,21))
    assert row["amount_count_last_10"] == 2.0
    assert row["amount_mean_last_10"] == 45.0
    assert row["amount_sum_last_100"] == 150.0  # all 5 rows


def test_multi_window_aggregate_empty_horizons_raises():
    import pytest

    history_df = pd.DataFrame({"entity": ["a"], "t": [1.0], "amount": [1.0]})
    query_df = pd.DataFrame({"entity": ["a"], "as_of": [2.0]})
    with pytest.raises(ValueError):
        multi_window_aggregate(history_df, "entity", "t", query_df, {"amount": ["sum"]}, lookback_horizons=[])
