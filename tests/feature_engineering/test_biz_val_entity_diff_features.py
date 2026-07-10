"""biz_value test for ``feature_engineering.entity_diff_features``.

The win: when the true signal is a statement-to-statement CHANGE (a worsening trend) rather than the raw
level itself, a model using only the raw value performs at chance, while a model using the diff feature
(computed here in bulk, boundary-safe per entity) recovers the signal cleanly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_engineering.entity_diff_features import entity_diff_features


def test_biz_val_entity_diff_features_recover_trend_signal_raw_level_misses():
    rng = np.random.default_rng(0)
    n_entities = 400
    rows_per_entity = 6

    rows = []
    labels = {}
    for entity_id in range(n_entities):
        worsening = rng.random() < 0.5
        labels[entity_id] = int(worsening)
        base_level = rng.normal(50, 10)
        values = base_level + rng.normal(0, 1, rows_per_entity)
        if worsening:
            # dip the SECOND-TO-LAST value down (not the last), so the final row's raw level keeps the same
            # distribution as the stable class -- only the statement-to-statement DELTA carries the signal.
            values[-2] -= rng.uniform(15, 25)
        for t, v in enumerate(values):
            rows.append({"entity": entity_id, "t": t, "amount": float(v)})

    df = pd.DataFrame(rows)
    diffed = entity_diff_features(df, entity_col="entity", feature_cols=["amount"], n=1)

    last_rows = diffed.sort_values(["entity", "t"]).groupby("entity", as_index=False).tail(1)
    y = last_rows["entity"].map(labels).to_numpy()

    raw_auc = cross_val_score(LogisticRegression(), last_rows[["amount"]], y, cv=5, scoring="roc_auc").mean()
    diff_auc = cross_val_score(LogisticRegression(), last_rows[["amount_diff"]], y, cv=5, scoring="roc_auc").mean()

    assert raw_auc < 0.6, f"the raw level should carry no signal (chance-level AUC), got {raw_auc:.4f}"
    assert diff_auc > 0.9, f"the diff feature should cleanly recover the trend signal, got {diff_auc:.4f}"


def test_entity_diff_features_never_bleeds_across_entities():
    df = pd.DataFrame({"entity": ["a", "a", "b", "b"], "t": [0, 1, 0, 1], "x": [10.0, 20.0, 1000.0, 1005.0]})
    result = entity_diff_features(df, entity_col="entity", feature_cols=["x"], n=1)
    assert pd.isna(result["x_diff"].iloc[0])  # entity a's first row
    assert result["x_diff"].iloc[1] == 10.0  # a: 20-10
    assert pd.isna(result["x_diff"].iloc[2])  # entity b's first row -- NOT 1000-20 (would be a cross-entity leak)
    assert result["x_diff"].iloc[3] == 5.0  # b: 1005-1000
