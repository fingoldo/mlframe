"""biz_value + unit tests for ``feature_engineering.fuzzy_entity.fuzzy_entity_group_features``.

The win: on synthetic fraud-style data where each entity has a "usual" attribute value it uses most of the
time, and fraud rows are exactly the ones where the value deviates from that usual pattern, BOTH
``group_mode_match`` (does this row match the group's usual value) and ``value_occurrence_count_in_group``
(has this exact value been seen in this group before) carry strong, quantifiable correlation with the fraud
label — while the raw value/group columns alone (without the within-group context) carry essentially none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mlframe.feature_engineering.fuzzy_entity import _cluster_fuzzy_keys, fuzzy_entity_group_features


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
        f"mode mismatch should be far more common on fraud rows: fraud={mismatch_rate_when_fraud:.3f} clean={mismatch_rate_when_clean:.3f}"
    )

    # novel-in-group values (occurrence_count==0) should correlate with fraud.
    corr_occurrence, _ = spearmanr(out["value_occurrence_count_in_group"], y)
    assert corr_occurrence < -0.15, f"occurrence count should negatively predict fraud, got corr={corr_occurrence:.3f}"

    # the raw VALUE alone (no group context) carries no meaning -- same values are normal for one entity
    # and anomalous for another, by construction. A crude "is this a large id" proxy should show ~no signal.
    corr_raw_value, _ = spearmanr(values.astype(np.float64), y)
    assert abs(corr_raw_value) < 0.1, f"raw value alone should not predict fraud, got corr={corr_raw_value:.3f}"


def _make_noisy_key_fraud_data(n_entities: int, events_per_entity: int, seed: int, key_typo_rate: float = 0.4, deviation_rate: float = 0.15):
    """Same fraud-style generative process as ``_make_fraud_style_data``, but the entity KEY itself is noisy:
    each event's key is the entity's canonical id string (``"cust_NNNNNN"``, 11 chars), occasionally
    perturbed by a single-character typo in its LAST character only, simulating a trailing OCR/checksum-digit
    error. Only the last char is perturbable so ``fuzzy_block_prefix_len=10`` (all but the last char) blocks
    keys into small buckets of entities sharing the same first 10 chars, keeping clustering sub-quadratic even
    as ``n_entities`` grows -- a fixed prefix length that ignores where the id's distinguishing digits actually
    live (e.g. blocking on ``"cust_0"`` for all 6-digit zero-padded ids) would put ~all keys in one giant block.
    """
    rng = np.random.default_rng(seed)
    value_pool = np.arange(1000, 1000 + n_entities * 3)
    usual_value_per_entity = rng.choice(value_pool, size=n_entities, replace=False)

    keys: list[str] = []
    values: list[int] = []
    y: list[float] = []
    for e in range(n_entities):
        # spaced out (x97, not sequential) so a single-last-digit typo lands on unused id space instead of
        # colliding with another real entity's canonical key -- sequential ids densely cover all last-digit
        # values within a decade, which would make typos silently relink to the WRONG existing entity instead
        # of fragmenting into a genuinely novel key.
        canonical_key = f"cust_{(e * 97) % 1_000_000:06d}"
        deviates = rng.random(events_per_entity) < deviation_rate
        vals = np.where(deviates, rng.choice(value_pool, size=events_per_entity), usual_value_per_entity[e])
        for i in range(events_per_entity):
            if rng.random() < key_typo_rate:
                chars = list(canonical_key)
                chars[-1] = str(rng.integers(0, 10))
                keys.append("".join(chars))
            else:
                keys.append(canonical_key)
            values.append(int(vals[i]))
            y.append(float(deviates[i]))
    return np.array(keys), np.array(values, dtype=np.int64), np.array(y, dtype=np.float64)


def test_biz_val_fuzzy_entity_group_features_fuzzy_key_matching_recovers_split_groups():
    keys, values, y = _make_noisy_key_fraud_data(n_entities=150, events_per_entity=12, seed=7)
    order = np.arange(len(keys), dtype=np.float64)

    out_exact = fuzzy_entity_group_features(keys, values, time_order=order)
    out_fuzzy = fuzzy_entity_group_features(keys, values, time_order=order, fuzzy_key_matching=True, fuzzy_max_distance=1, fuzzy_block_prefix_len=10)

    # exact matching sees each typo'd key as a brand-new, distinct group -- true entity count is 150, but
    # noisy identifiers fragment it into far more exact-match groups.
    n_exact_groups = len(np.unique(keys))
    assert n_exact_groups > 150, f"expected typo noise to fragment groups past 150, got {n_exact_groups}"

    # fuzzy clustering should recover (near) the true 150-entity structure.
    n_fuzzy_groups = len(pd.unique(_cluster_fuzzy_keys(np.asarray(keys), max_distance=1, block_prefix_len=10)))
    assert n_fuzzy_groups <= 160, f"fuzzy clustering should recover close to 150 true entities, got {n_fuzzy_groups} groups"

    # under-populated fragmented groups make "group mode" and "seen this value before" far less informative,
    # so occurrence-count's correlation with the fraud label should be measurably weaker under exact matching
    # than under fuzzy matching, which reunites an entity's events into one group.
    corr_exact, _ = spearmanr(out_exact["value_occurrence_count_in_group"], y)
    corr_fuzzy, _ = spearmanr(out_fuzzy["value_occurrence_count_in_group"], y)
    assert corr_fuzzy < corr_exact - 0.05, (
        f"fuzzy key matching should strengthen occurrence-count's fraud signal vs exact matching: exact corr={corr_exact:.3f} fuzzy corr={corr_fuzzy:.3f}"
    )
    assert corr_fuzzy < -0.2, f"fuzzy-matched occurrence count should strongly predict fraud, got corr={corr_fuzzy:.3f}"


def test_fuzzy_entity_group_features_fuzzy_key_matching_default_off_is_bit_identical():
    entity_ids, values, order, _ = _make_fraud_style_data(20, 8, seed=3)
    out_default = fuzzy_entity_group_features(entity_ids, values, time_order=order)
    out_explicit_off = fuzzy_entity_group_features(entity_ids, values, time_order=order, fuzzy_key_matching=False)
    for key in out_default:
        np.testing.assert_array_equal(out_default[key], out_explicit_off[key])
