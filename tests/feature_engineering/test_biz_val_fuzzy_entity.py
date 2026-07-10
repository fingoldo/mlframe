"""biz_value + unit tests for ``feature_engineering.fuzzy_entity.fuzzy_entity_group_features``.

The win: on synthetic fraud-style data where each entity has a "usual" attribute value it uses most of the
time, and fraud rows are exactly the ones where the value deviates from that usual pattern, BOTH
``group_mode_match`` (does this row match the group's usual value) and ``value_occurrence_count_in_group``
(has this exact value been seen in this group before) carry strong, quantifiable correlation with the fraud
label — while the raw value/group columns alone (without the within-group context) carry essentially none.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from mlframe.feature_engineering.fuzzy_entity import fuzzy_entity_group_features


def _make_fraud_style_data(n_entities: int, events_per_entity: int, seed: int, deviation_rate: float = 0.15):
    rng = np.random.default_rng(seed)
    entity_ids = np.repeat(np.arange(n_entities), events_per_entity)
    n = entity_ids.shape[0]

    # each entity has a "usual" device/domain value, drawn from a shared pool so raw value alone has no
    # meaning (the SAME value is "normal" for one entity and "anomalous" for another).
    value_pool = np.arange(1000, 1000 + n_entities * 3)
    usual_value_per_entity = rng.choice(value_pool, size=n_entities, replace=False)

    values = np.empty(n, dtype=np.int64)
    y = np.empty(n, dtype=np.float64)
    pos = 0
    for e in range(n_entities):
        deviates = rng.random(events_per_entity) < deviation_rate
        vals = np.where(deviates, rng.choice(value_pool, size=events_per_entity), usual_value_per_entity[e])
        values[pos : pos + events_per_entity] = vals
        y[pos : pos + events_per_entity] = deviates.astype(np.float64)
        pos += events_per_entity

    order = np.arange(n, dtype=np.float64)
    return entity_ids, values, order, y


def test_fuzzy_entity_group_features_returns_expected_keys_and_shapes():
    entity_ids, values, order, _ = _make_fraud_style_data(5, 6, seed=0)
    out = fuzzy_entity_group_features(entity_ids, values, time_order=order)
    for key in ("group_mode_match", "value_occurrence_count_in_group", "days_since_value_last_seen_in_group"):
        assert key in out
        assert out[key].shape == entity_ids.shape


def test_fuzzy_entity_group_features_mode_match_correctness_small_case():
    entity_ids = np.array([1, 1, 1, 2, 2])
    values = np.array(["A", "A", "B", "X", "Y"])
    out = fuzzy_entity_group_features(entity_ids, values)
    # entity 1: mode is "A" (2 occurrences) -> rows 0,1 match, row 2 (B) doesn't.
    assert out["group_mode_match"][0] and out["group_mode_match"][1] and not out["group_mode_match"][2]


def test_fuzzy_entity_group_features_occurrence_count_is_causal():
    entity_ids = np.array([1, 1, 1])
    values = np.array(["A", "A", "A"])
    order = np.array([0.0, 1.0, 2.0])
    out = fuzzy_entity_group_features(entity_ids, values, time_order=order)
    assert out["value_occurrence_count_in_group"][0] == 0.0  # first occurrence -- never seen before
    assert out["value_occurrence_count_in_group"][1] == 1.0
    assert out["value_occurrence_count_in_group"][2] == 2.0


def test_fuzzy_entity_group_features_days_since_last_seen_nan_on_first_occurrence():
    entity_ids = np.array([1, 1, 1])
    values = np.array(["A", "B", "A"])
    order = np.array([0.0, 5.0, 12.0])
    out = fuzzy_entity_group_features(entity_ids, values, time_order=order)
    assert np.isnan(out["days_since_value_last_seen_in_group"][0])  # first "A"
    assert np.isnan(out["days_since_value_last_seen_in_group"][1])  # first "B"
    assert out["days_since_value_last_seen_in_group"][2] == 12.0  # second "A", gap = 12 - 0


def test_biz_val_fuzzy_entity_features_predict_deviation_while_raw_value_does_not():
    entity_ids, values, order, y = _make_fraud_style_data(n_entities=300, events_per_entity=15, seed=42)
    out = fuzzy_entity_group_features(entity_ids, values, time_order=order)

    # group_mode_match=False should coincide with y=1 (deviation) far more than chance.
    mismatch_rate_when_fraud = float((~out["group_mode_match"][y == 1]).mean())
    mismatch_rate_when_clean = float((~out["group_mode_match"][y == 0]).mean())
    assert mismatch_rate_when_fraud > mismatch_rate_when_clean + 0.5, (
        f"mode mismatch should be far more common on fraud rows: fraud={mismatch_rate_when_fraud:.3f} "
        f"clean={mismatch_rate_when_clean:.3f}"
    )

    # novel-in-group values (occurrence_count==0) should correlate with fraud.
    corr_occurrence, _ = spearmanr(out["value_occurrence_count_in_group"], y)
    assert corr_occurrence < -0.15, f"occurrence count should negatively predict fraud, got corr={corr_occurrence:.3f}"

    # the raw VALUE alone (no group context) carries no meaning -- same values are normal for one entity
    # and anomalous for another, by construction. A crude "is this a large id" proxy should show ~no signal.
    corr_raw_value, _ = spearmanr(values.astype(np.float64), y)
    assert abs(corr_raw_value) < 0.1, f"raw value alone should not predict fraud, got corr={corr_raw_value:.3f}"
