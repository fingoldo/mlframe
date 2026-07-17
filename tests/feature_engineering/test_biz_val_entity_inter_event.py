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
    is_bursty_entity = np.arange(n_entities) % 2 == 0
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


def _make_entity_tempo_shift_data(n_entities: int, events_per_entity: int, switch_idx: int, seed: int):
    """Each entity has a baseline tempo (shared across all entities, same distribution) for its first
    ``switch_idx`` events, then abruptly switches to its OWN recent tempo for the remaining events -- a
    churning/accelerating-customer regime change. ``recent_gap_row`` is the ground-truth per-row recent-regime
    mean gap (what a windowed feature should recover); the shared baseline carries zero information about it.
    """
    rng = np.random.default_rng(seed)
    entity_ids = np.repeat(np.arange(n_entities), events_per_entity)
    baseline_gap = 10.0
    recent_gap = rng.uniform(2.0, 60.0, size=n_entities)

    timestamps = np.empty(entity_ids.shape[0], dtype=np.float64)
    recent_gap_row = np.empty(entity_ids.shape[0], dtype=np.float64)
    eval_mask = np.zeros(entity_ids.shape[0], dtype=bool)
    pos = 0
    for e in range(n_entities):
        gaps = np.empty(events_per_entity)
        gaps[:switch_idx] = rng.exponential(baseline_gap, switch_idx)
        gaps[switch_idx:] = rng.exponential(recent_gap[e], events_per_entity - switch_idx)
        ts = np.cumsum(gaps) + rng.uniform(0, 200_000)
        timestamps[pos : pos + events_per_entity] = ts
        recent_gap_row[pos : pos + events_per_entity] = recent_gap[e]
        # evaluate only on the tail, once the trailing window is fully inside the new regime -- the
        # rows right at the switch boundary still legitimately mix both regimes for ANY causal feature.
        eval_mask[pos + events_per_entity - 5 : pos + events_per_entity] = True
        pos += events_per_entity
    return entity_ids, timestamps, recent_gap_row, eval_mask


def test_biz_val_entity_windowed_time_delta_tracks_recent_tempo_shift_vs_whole_history():
    entity_ids, timestamps, recent_gap_row, eval_mask = _make_entity_tempo_shift_data(n_entities=200, events_per_entity=60, switch_idx=50, seed=42)
    out = entity_inter_event_features(entity_ids, timestamps, window_size=5)

    mae_whole = np.mean(np.abs(out["group_mean_time_delta"][eval_mask] - recent_gap_row[eval_mask]))
    mae_windowed = np.mean(np.abs(out["group_mean_time_delta_windowed"][eval_mask] - recent_gap_row[eval_mask]))

    # measured mae_whole ~= 17.9, mae_windowed ~= 10.5 (windowed/whole ratio ~= 0.59); threshold set with
    # headroom above the measured ratio so the test isn't pinned to the exact synthetic seed.
    assert mae_windowed < mae_whole * 0.75, (
        f"windowed feature should track the recent regime much more tightly, got mae_whole={mae_whole:.2f} mae_windowed={mae_windowed:.2f}"
    )
    assert mae_whole > 15.0, f"whole-history feature should stay diluted by the long baseline history, got mae_whole={mae_whole:.2f}"
    assert mae_windowed < 12.0, f"windowed feature should closely track the true recent-regime gap, got mae_windowed={mae_windowed:.2f}"


def test_biz_val_entity_windowed_time_delta_default_omitted_matches_whole_history_only():
    entity_ids, timestamps, _, _ = _make_entity_tempo_shift_data(n_entities=20, events_per_entity=20, switch_idx=15, seed=7)
    out_default = entity_inter_event_features(entity_ids, timestamps)
    out_windowed = entity_inter_event_features(entity_ids, timestamps, window_size=4)
    # opt-in must be strictly additive: every default key stays bit-identical when window_size/window_time is omitted.
    for key in out_default:
        assert np.array_equal(out_default[key], out_windowed[key], equal_nan=True), key
    assert "group_mean_time_delta_windowed" not in out_default
    assert "group_mean_time_delta_windowed" in out_windowed


def test_entity_inter_event_features_window_size_matches_hand_computed_trailing_mean():
    entity_ids = np.array([1, 1, 1, 1, 1])
    timestamps = np.array([0.0, 10.0, 22.0, 27.0, 47.0])  # gaps: nan, 10, 12, 5, 20
    out = entity_inter_event_features(entity_ids, timestamps, window_size=2)
    # row index 2 (gap=12): trailing window of last 2 gaps ending here = [10, 12] -> mean 11
    assert out["group_mean_time_delta_windowed"][2] == 11.0
    # row index 4 (gap=20): trailing window of last 2 gaps ending here = [5, 20] -> mean 12.5
    assert out["group_mean_time_delta_windowed"][4] == 12.5
    # row 0 has no gap (first of entity) -> nan
    assert np.isnan(out["group_mean_time_delta_windowed"][0])


def test_entity_inter_event_features_window_time_excludes_future_events():
    entity_ids = np.array([1, 1, 1, 1])
    timestamps = np.array([0.0, 5.0, 6.0, 20.0])  # gaps: nan, 5, 1, 14
    out = entity_inter_event_features(entity_ids, timestamps, window_time=3.0)
    # at row 2 (ts=6), window is (3, 6]: row 1 (ts=5, gap=5) and row 2 (ts=6, gap=1) both qualify -> mean 3.
    # row 0 (ts=0) does NOT qualify (0 <= 3) -- proving the window is bounded, not whole-history-to-date.
    assert out["group_mean_time_delta_windowed"][2] == 3.0
    # at row 3 (ts=20), window is (17, 20] -> only its own gap (14) qualifies.
    assert out["group_mean_time_delta_windowed"][3] == 14.0


def test_entity_inter_event_features_window_size_and_window_time_mutually_exclusive():
    entity_ids = np.array([1, 1])
    timestamps = np.array([0.0, 1.0])
    try:
        entity_inter_event_features(entity_ids, timestamps, window_size=2, window_time=1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


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
