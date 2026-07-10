"""biz_value test for ``feature_engineering.two_step_recency_weighted_target_encode``.

The win: when an entity's true label is driven by its RECENT behavior (a feature-combo pattern used in its
most recent events) while its older events used an unrelated random pattern (representing a prior, now-
irrelevant behavioral era), a recency-weighted two-step aggregate recovers the entity label far better than
an unweighted aggregate, which gets diluted by the older, uninformative majority of events.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_engineering.two_step_target_encode import two_step_recency_weighted_target_encode


def _make_data(seed: int):
    rng = np.random.default_rng(seed)
    n_entities = 300
    rows = []
    entity_label = {}
    for e in range(n_entities):
        label = int(rng.random() < 0.5)
        entity_label[e] = label
        n_old = rng.integers(20, 30)
        n_recent = rng.integers(2, 4)
        for t in range(n_old):
            # old era: pattern is RANDOM, uncorrelated with the entity's current label.
            pat = rng.choice(["A", "B"])
            y_event = 1.0 if pat == "A" else 0.0
            y_event = y_event if rng.random() > 0.1 else 1 - y_event
            rows.append({"entity": e, "t": float(t), "cat1": pat, "y": y_event})
        for t in range(n_old, n_old + n_recent):
            # recent era: pattern IS the entity's current label.
            pat = "A" if label == 1 else "B"
            y_event = 1.0 if pat == "A" else 0.0
            y_event = y_event if rng.random() > 0.1 else 1 - y_event
            rows.append({"entity": e, "t": float(t), "cat1": pat, "y": y_event})

    events_df = pd.DataFrame(rows)
    return events_df, entity_label


def test_biz_val_two_step_recency_weighted_encoding_beats_unweighted_aggregate():
    events_df, entity_label = _make_data(seed=0)
    y_all = events_df["y"].to_numpy()

    weighted = two_step_recency_weighted_target_encode(events_df, "entity", ["cat1"], y_all, "t", decay_half_life=2.0)
    unweighted = two_step_recency_weighted_target_encode(events_df, "entity", ["cat1"], y_all, "t", decay_half_life=1e6)

    entity_ids = events_df["entity"].to_numpy()
    labels_arr = np.array([entity_label[e] for e in entity_ids])
    first_idx = events_df.groupby("entity").head(1).index

    X_weighted = weighted[first_idx].reshape(-1, 1)
    X_unweighted = unweighted[first_idx].reshape(-1, 1)
    y_entity = labels_arr[first_idx]

    auc_weighted = cross_val_score(LogisticRegression(), X_weighted, y_entity, cv=5, scoring="roc_auc").mean()
    auc_unweighted = cross_val_score(LogisticRegression(), X_unweighted, y_entity, cv=5, scoring="roc_auc").mean()

    assert auc_weighted > auc_unweighted + 0.1, (
        f"recency-weighted two-step encoding should recover the entity label far better than unweighted aggregation: "
        f"weighted={auc_weighted:.4f} unweighted={auc_unweighted:.4f}"
    )
    assert auc_weighted > 0.95


def test_two_step_recency_weighted_encode_same_value_per_entity():
    events_df, _ = _make_data(seed=1)
    y_all = events_df["y"].to_numpy()
    encoded = two_step_recency_weighted_target_encode(events_df, "entity", ["cat1"], y_all, "t", decay_half_life=2.0)
    result_df = events_df.assign(enc=encoded)
    per_entity_unique_counts = result_df.groupby("entity")["enc"].nunique()
    assert (per_entity_unique_counts == 1).all()
