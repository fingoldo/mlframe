"""biz_value + unit tests for ``feature_engineering.entity_inter_event.entity_inter_event_features``.

The win: on synthetic transaction data where the TARGET is driven by an entity's behavioral tempo (bursty
entities with short inter-event gaps are fraud-like) rather than any raw per-row value, the engineered
``group_mean_time_delta`` feature carries strong correlation with the target while the raw timestamp column
alone carries essentially none — quantifying that the group-level aggregate genuinely extracts entity-level
behavioral signal a raw per-row column cannot express.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from mlframe.feature_engineering.entity_inter_event import entity_inter_event_features


def _make_entity_tempo_data(n_entities: int, events_per_entity: int, seed: int):
    rng = np.random.default_rng(seed)
    entity_ids = np.repeat(np.arange(n_entities), events_per_entity)
    # half the entities are "bursty" (small inter-event gaps, fraud-like), half "steady" (large gaps).
    is_bursty_entity = (np.arange(n_entities) % 2 == 0)
    mean_gap_per_entity = np.where(is_bursty_entity, 2.0, 50.0)

    timestamps = np.empty(entity_ids.shape[0], dtype=np.float64)
    y = np.empty(entity_ids.shape[0], dtype=np.float64)
    pos = 0
    for e in range(n_entities):
        gaps = rng.exponential(scale=mean_gap_per_entity[e], size=events_per_entity)
        # random per-entity start offset, range >> the cumulative drift (at most ~events*mean_gap) so the
        # RAW timestamp position carries no residual tempo signal -- only the inter-event GAPS do.
        ts = np.cumsum(gaps) + rng.uniform(0, 200_000)
        timestamps[pos : pos + events_per_entity] = ts
        y[pos : pos + events_per_entity] = float(is_bursty_entity[e])
        pos += events_per_entity
    return entity_ids, timestamps, y


def test_entity_inter_event_features_returns_expected_keys_and_shapes():
    entity_ids, timestamps, _ = _make_entity_tempo_data(5, 4, seed=0)
    out = entity_inter_event_features(entity_ids, timestamps)
    for key in ("time_since_prev_event", "time_to_next_event", "group_mean_time_delta", "group_std_time_delta", "group_median_time_delta"):
        assert key in out
        assert out[key].shape == entity_ids.shape


def test_entity_inter_event_features_first_and_last_row_of_group_are_nan():
    entity_ids = np.array([1, 1, 1, 2, 2])
    timestamps = np.array([0.0, 5.0, 12.0, 100.0, 108.0])
    out = entity_inter_event_features(entity_ids, timestamps)
    assert np.isnan(out["time_since_prev_event"][0])  # first row of entity 1
    assert np.isnan(out["time_since_prev_event"][3])  # first row of entity 2
    assert np.isnan(out["time_to_next_event"][2])  # last row of entity 1
    assert np.isnan(out["time_to_next_event"][4])  # last row of entity 2
    assert out["time_since_prev_event"][1] == 5.0
    assert out["time_since_prev_event"][2] == 7.0


def test_entity_inter_event_features_value_col_adds_value_stats():
    entity_ids = np.array([1, 1, 2, 2])
    timestamps = np.array([0.0, 1.0, 0.0, 1.0])
    values = np.array([10.0, 20.0, 100.0, 300.0])
    out = entity_inter_event_features(entity_ids, timestamps, value_col=values)
    assert np.allclose(out["group_mean_value"], [15.0, 15.0, 200.0, 200.0])
    assert "group_std_value" in out and "group_median_value" in out


def test_entity_inter_event_features_group_stat_constant_within_entity():
    entity_ids, timestamps, _ = _make_entity_tempo_data(6, 8, seed=1)
    out = entity_inter_event_features(entity_ids, timestamps)
    for e in np.unique(entity_ids):
        mask = entity_ids == e
        vals = out["group_mean_time_delta"][mask]
        assert np.allclose(vals, vals[0])  # same broadcast value for every row of the entity


def test_biz_val_entity_group_mean_gap_predicts_target_while_raw_timestamp_does_not():
    entity_ids, timestamps, y = _make_entity_tempo_data(n_entities=200, events_per_entity=10, seed=42)
    out = entity_inter_event_features(entity_ids, timestamps)

    corr_group_mean, _ = spearmanr(out["group_mean_time_delta"], y)
    corr_raw_timestamp, _ = spearmanr(timestamps, y)

    # group_mean_time_delta directly encodes entity tempo (small = bursty = target 1), so it should
    # correlate strongly and NEGATIVELY with y. Floor set well below the measured value (~-0.85).
    assert corr_group_mean < -0.5, f"group_mean_time_delta should strongly predict target, got corr={corr_group_mean:.3f}"
    # raw per-row timestamp carries no entity-tempo information by construction (each entity's start
    # offset is random and uncorrelated with its bursty/steady label) -- near-zero correlation.
    assert abs(corr_raw_timestamp) < 0.15, f"raw timestamp should NOT predict target on its own, got corr={corr_raw_timestamp:.3f}"
