"""biz_value test for ``feature_engineering.windowed_edge_diff.windowed_edge_aggregate_diff``.

The win: an entity's OVERALL mean value can be identical whether its behavior is trending up or trending
down over its observed history -- a plain groupby-mean aggregate carries zero signal about trend direction.
``windowed_edge_aggregate_diff``'s first-N-vs-last-N diff/ratio directly captures that trend, giving a
downstream classifier real separating power a level-only aggregate cannot provide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_engineering.windowed_edge_diff import windowed_edge_aggregate_diff


def _make_trending_entities_dataset(n_entities: int, seed: int):
    """Helper: Make trending entities dataset."""
    rng = np.random.default_rng(seed)
    rows = []
    labels = np.zeros(n_entities, dtype=int)
    for entity in range(n_entities):
        label = rng.integers(0, 2)
        labels[entity] = label
        n_obs = rng.integers(8, 15)
        base = rng.uniform(5, 15)
        # "growing" (label=1): values ramp up from base-delta to base+delta; "shrinking" (label=0): ramp down.
        # The overall MEAN across the whole trajectory is ~base regardless of label -- the trend direction is
        # invisible to a plain mean, only visible by comparing early vs late records.
        delta = 4.0
        trend = np.linspace(-delta, delta, n_obs) if label == 1 else np.linspace(delta, -delta, n_obs)
        values = base + trend + rng.normal(scale=0.5, size=n_obs)
        for t, v in enumerate(values):
            rows.append({"entity": entity, "t": t, "value": v})
    return pd.DataFrame(rows), labels


def test_biz_val_windowed_edge_diff_separates_trend_when_mean_cannot():
    """Biz val windowed edge diff separates trend when mean cannot."""
    df, labels = _make_trending_entities_dataset(n_entities=150, seed=0)
    entities = pd.unique(df["entity"])
    y = labels[entities]

    mean_feats = df.groupby("entity", sort=False)["value"].mean().reindex(entities).to_numpy().reshape(-1, 1)
    auc_mean = cross_val_score(LogisticRegression(max_iter=500), mean_feats, y, cv=5, scoring="roc_auc").mean()

    edge = windowed_edge_aggregate_diff(df, entity_col="entity", time_col="t", value_col="value", n=3, agg="mean").set_index("entity").reindex(entities)
    diff_feats = edge[["value_edge_diff_3_mean"]].to_numpy()
    auc_diff = cross_val_score(LogisticRegression(max_iter=500), diff_feats, y, cv=5, scoring="roc_auc").mean()

    assert auc_mean < 0.6, f"expected the plain-mean baseline to be near-chance (trend is invisible to it), got AUC={auc_mean:.4f}"
    assert auc_diff > 0.9, f"expected the edge-diff feature to strongly separate trend direction, got AUC={auc_diff:.4f}"


def test_windowed_edge_diff_short_group_uses_available_records():
    """Windowed edge diff short group uses available records."""
    df = pd.DataFrame({"entity": [1], "t": [0], "value": [5.0]})
    out = windowed_edge_aggregate_diff(df, "entity", "t", "value", n=3, agg="mean")
    assert out["value_edge_diff_3_mean"].item() == 0.0
    assert out["value_edge_ratio_3_mean"].item() == 1.0


def test_windowed_edge_diff_respects_time_ordering():
    """Windowed edge diff respects time ordering."""
    df = pd.DataFrame({"entity": [1, 1, 1, 1], "t": [3, 1, 2, 0], "value": [40.0, 20.0, 30.0, 10.0]})
    out = windowed_edge_aggregate_diff(df, "entity", "t", "value", n=2, agg="mean")
    assert out["value_first2_mean"].item() == 15.0  # t=0,1 -> (10+20)/2
    assert out["value_last2_mean"].item() == 35.0  # t=2,3 -> (30+40)/2
