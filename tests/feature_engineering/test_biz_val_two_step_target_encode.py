"""biz_value test for ``feature_engineering.two_step_recency_weighted_target_encode``.

The win: when an entity's true label is driven by its RECENT behavior (a feature-combo pattern used in its
most recent events) while its older events used an unrelated random pattern (representing a prior, now-
irrelevant behavioral era), a recency-weighted two-step aggregate recovers the entity label far better than
an unweighted aggregate, which gets diluted by the older, uninformative majority of events.

The second win (``causal=True``): the default (``causal=False``) aggregate is constant per entity -- every
row of an entity, including its OLDEST rows, gets the identical full-history aggregate. Scored "as of" an
early row (before the entity's recent, label-defining era has even happened), that constant already contains
hindsight about the entity's future behavior -- a real deployment could never have known it at that point in
time. ``causal=True`` restricts each row to only its own entity's PAST events, so a row scored before the
recent era exists correctly reflects "nothing is knowable yet" instead of leaking the future answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from mlframe.feature_engineering.two_step_target_encode import two_step_recency_weighted_target_encode


def _make_data(seed: int):
    """Helper: Make data."""
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
    """Biz val two step recency weighted encoding beats unweighted aggregate."""
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


def _make_data_with_boundary(seed: int):
    """Same generative process as ``_make_data``, plus each entity's last OLD-era row index.

    That boundary row is the last point in time at which the entity's future (label-defining) recent era is
    still unknown -- the natural "score me now" point for a leakage check.
    """
    rng = np.random.default_rng(seed)
    n_entities = 300
    rows = []
    entity_label = {}
    boundary_idx = []
    row_i = 0
    for e in range(n_entities):
        label = int(rng.random() < 0.5)
        entity_label[e] = label
        n_old = rng.integers(20, 30)
        n_recent = rng.integers(2, 4)
        for t in range(n_old):
            pat = rng.choice(["A", "B"])
            y_event = 1.0 if pat == "A" else 0.0
            y_event = y_event if rng.random() > 0.1 else 1 - y_event
            rows.append({"entity": e, "t": float(t), "cat1": pat, "y": y_event})
            row_i += 1
        boundary_idx.append(row_i - 1)  # last old-era row just appended
        for t in range(n_old, n_old + n_recent):
            pat = "A" if label == 1 else "B"
            y_event = 1.0 if pat == "A" else 0.0
            y_event = y_event if rng.random() > 0.1 else 1 - y_event
            rows.append({"entity": e, "t": float(t), "cat1": pat, "y": y_event})
            row_i += 1

    events_df = pd.DataFrame(rows)
    return events_df, entity_label, np.array(boundary_idx)


def test_biz_val_two_step_causal_avoids_future_leakage_at_early_scoring_point():
    """Biz val two step causal avoids future leakage at early scoring point."""
    events_df, entity_label, boundary_idx = _make_data_with_boundary(seed=2)
    y_all = events_df["y"].to_numpy()
    entity_ids = events_df["entity"].to_numpy()
    labels_arr = np.array([entity_label[e] for e in entity_ids])

    naive = two_step_recency_weighted_target_encode(events_df, "entity", ["cat1"], y_all, "t", decay_half_life=2.0, causal=False)
    causal = two_step_recency_weighted_target_encode(events_df, "entity", ["cat1"], y_all, "t", decay_half_life=2.0, causal=True)

    y_boundary = labels_arr[boundary_idx]
    naive_auc = roc_auc_score(y_boundary, naive[boundary_idx])
    causal_auc = roc_auc_score(y_boundary, causal[boundary_idx])

    # naive leaks the entity's not-yet-happened recent era into every earlier row (constant-per-entity
    # aggregate), so scoring "as of" the last old-era row looks almost perfectly predictive -- impossible in
    # real deployment, since that future hasn't occurred yet at that point in time.
    assert naive_auc > 0.85, f"expected naive (causal=False) to show hindsight leakage at the boundary row, got AUC={naive_auc:.4f}"
    # causal only sees the entity's own past (pure old-era noise) at that row, so it should be near chance.
    assert causal_auc < 0.62, f"expected causal=True to be near-chance (no future info yet) at the boundary row, got AUC={causal_auc:.4f}"
    assert naive_auc - causal_auc > 0.3, (
        f"causal=True should close most of the future-leakage gap at the boundary row: naive={naive_auc:.4f} causal={causal_auc:.4f}"
    )


def test_two_step_recency_weighted_encode_causal_matches_default_at_last_event():
    """Regression pin: at an entity's LAST event, causal (expanding to entity's full history) must equal
    the default full-history aggregate -- both average over exactly the same set of events at that point."""
    events_df, _ = _make_data(seed=3)
    y_all = events_df["y"].to_numpy()
    default = two_step_recency_weighted_target_encode(events_df, "entity", ["cat1"], y_all, "t", decay_half_life=2.0, causal=False)
    causal = two_step_recency_weighted_target_encode(events_df, "entity", ["cat1"], y_all, "t", decay_half_life=2.0, causal=True)
    last_idx = events_df.groupby("entity").tail(1).index
    np.testing.assert_allclose(causal[last_idx], default[last_idx], rtol=1e-10)


def test_two_step_recency_weighted_encode_default_unchanged_when_causal_not_passed():
    """Regression pin: omitting ``causal`` (the new opt-in param) must be bit-identical to ``causal=False``,
    and both must be bit-identical to the pre-extension behavior."""
    events_df, _ = _make_data(seed=4)
    y_all = events_df["y"].to_numpy()
    implicit_default = two_step_recency_weighted_target_encode(events_df, "entity", ["cat1"], y_all, "t", decay_half_life=2.0)
    explicit_default = two_step_recency_weighted_target_encode(events_df, "entity", ["cat1"], y_all, "t", decay_half_life=2.0, causal=False)
    np.testing.assert_array_equal(implicit_default, explicit_default)


def test_two_step_recency_weighted_encode_same_value_per_entity():
    """Two step recency weighted encode same value per entity."""
    events_df, _ = _make_data(seed=1)
    y_all = events_df["y"].to_numpy()
    encoded = two_step_recency_weighted_target_encode(events_df, "entity", ["cat1"], y_all, "t", decay_half_life=2.0)
    result_df = events_df.assign(enc=encoded)
    per_entity_unique_counts = result_df.groupby("entity")["enc"].nunique()
    assert (per_entity_unique_counts == 1).all()
