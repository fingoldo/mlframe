"""biz_value tests for ``mlframe.competition.known_label_override``.

Covers both competition-only override tricks:

* ``monotonic_entity_override`` -- on a synthetic "once fraud always fraud" entity panel
  where the raw model misses some past occurrences of known-eventually-fraud entities,
  the override must recover those missed rows and produce a measurable recall/AUC gain.
* ``known_label_override`` -- on a synthetic partial-known-label recovery scenario, the
  safe one-directional override must improve the target metric, and must NEVER hurt it,
  while a naive bidirectional override (applied for comparison only) demonstrably CAN hurt
  it when some "negative-direction" recovered labels are wrong -- concretely illustrating
  the asymmetric-safety rationale documented in the module.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from mlframe.competition.known_label_override import known_label_override, monotonic_entity_override


def _make_monotonic_entity_dataset(n_entities: int, rows_per_entity: int, seed: int):
    rng = np.random.default_rng(seed)
    n = n_entities * rows_per_entity
    entity_ids = np.repeat(np.arange(n_entities), rows_per_entity)

    # half the entities eventually become fraud; once fraud, ALL their rows are fraud (domain monotonicity)
    is_fraud_entity = np.zeros(n_entities, dtype=bool)
    is_fraud_entity[: n_entities // 2] = True
    rng.shuffle(is_fraud_entity)
    y_true = is_fraud_entity[entity_ids].astype(float)

    # raw model: strong on non-fraud entities, but MISSES a large fraction of true-fraud rows
    # (simulates a model that only reliably detects fraud once late-stage signal appears)
    raw_preds = np.where(
        y_true == 1.0,
        rng.uniform(0.0, 0.55, size=n),  # many false negatives baked in (mostly below a 0.5 threshold)
        rng.uniform(0.0, 0.3, size=n),
    )

    known_positive_entity_ids = set(np.where(is_fraud_entity)[0].tolist())
    return raw_preds, entity_ids, known_positive_entity_ids, y_true


def test_biz_val_monotonic_entity_override_recovers_missed_past_fraud_rows():
    raw_preds, entity_ids, known_positive_entity_ids, y_true = _make_monotonic_entity_dataset(n_entities=200, rows_per_entity=10, seed=0)

    baseline_recall = float(np.mean(raw_preds[y_true == 1.0] >= 0.5))
    baseline_auc = roc_auc_score(y_true, raw_preds)

    overridden = monotonic_entity_override(raw_preds, entity_ids, known_positive_entity_ids)
    overridden_recall = float(np.mean(overridden[y_true == 1.0] >= 0.5))
    overridden_auc = roc_auc_score(y_true, overridden)

    assert baseline_recall < 0.6, f"expected raw model to miss a lot of fraud rows, got recall={baseline_recall}"
    assert overridden_recall == 1.0, f"expected override to recover ALL known-positive-entity rows, got {overridden_recall}"

    recall_gain = overridden_recall - baseline_recall
    assert recall_gain > 0.3, f"expected recall gain > 0.3, got {recall_gain:.4f}"

    auc_gain = overridden_auc - baseline_auc
    assert auc_gain > 0.05, f"expected AUC gain > 0.05, got {auc_gain:.4f} (baseline={baseline_auc:.4f}, overridden={overridden_auc:.4f})"

    # non-fraud entities' predictions must be untouched
    non_fraud_mask = ~np.isin(entity_ids, list(known_positive_entity_ids))
    assert np.array_equal(overridden[non_fraud_mask], raw_preds[non_fraud_mask])


def test_biz_val_monotonic_entity_override_noop_when_no_known_positives():
    raw_preds, entity_ids, _known_positive_entity_ids, _y_true = _make_monotonic_entity_dataset(n_entities=50, rows_per_entity=5, seed=1)
    overridden = monotonic_entity_override(raw_preds, entity_ids, known_positive_entity_ids=set())
    assert np.array_equal(overridden, raw_preds)
    assert overridden is not raw_preds  # must not mutate/alias the input


def _make_known_label_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n).astype(float)
    # base rate skewed toward negative (rare-positive scenario, as in the fraud writeup)
    y_true[: int(n * 0.85)] = 0.0
    rng.shuffle(y_true)

    # model with false negatives (misses some true positives) AND some false positives it's overconfident about
    raw_preds = np.where(y_true == 1.0, rng.uniform(0.1, 0.6, size=n), rng.uniform(0.0, 0.4, size=n))

    positive_idx = np.where(y_true == 1.0)[0]
    negative_idx = np.where(y_true == 0.0)[0]

    # high-confidence recovered POSITIVE labels: near-certain (all correct), covering half the true positives
    recovered_positive_idx = rng.choice(positive_idx, size=max(1, len(positive_idx) // 2), replace=False)

    # "recovered negative" labels: noisy -- some are WRONG (actually true positives), simulating the
    # asymmetric-uncertainty rationale (evidence for "not the rare class" is much weaker/noisier)
    recovered_negative_idx = rng.choice(negative_idx, size=max(1, len(negative_idx) // 3), replace=False)
    n_wrong = max(1, len(recovered_negative_idx) // 5)
    wrong_recovered_negative_idx = rng.choice(positive_idx, size=n_wrong, replace=False)

    known_label_map_safe_only = {int(i): 1.0 for i in recovered_positive_idx}

    known_label_map_bidirectional = dict(known_label_map_safe_only)
    known_label_map_bidirectional.update({int(i): 0.0 for i in recovered_negative_idx})
    known_label_map_bidirectional.update({int(i): 0.0 for i in wrong_recovered_negative_idx})  # noisy/wrong evidence

    return raw_preds, y_true, known_label_map_safe_only, known_label_map_bidirectional


def test_biz_val_known_label_override_safe_direction_improves_without_ever_hurting():
    raw_preds, y_true, known_label_map_safe_only, _bidir_map = _make_known_label_dataset(n=3000, seed=3)

    baseline_auc = roc_auc_score(y_true, raw_preds)
    overridden = known_label_override(raw_preds, known_label_map_safe_only, asymmetric_safe_direction="positive")
    overridden_auc = roc_auc_score(y_true, overridden)

    assert overridden_auc >= baseline_auc, f"safe-direction override must never hurt AUC: baseline={baseline_auc:.4f}, overridden={overridden_auc:.4f}"
    auc_gain = overridden_auc - baseline_auc
    assert auc_gain > 0.02, f"expected a real AUC gain from recovering missed positives, got {auc_gain:.4f}"

    # every recovered-positive row must now read as positive_value
    for idx in known_label_map_safe_only:
        assert overridden[idx] == 1.0


def test_biz_val_known_label_override_bidirectional_can_hurt_vs_safe_direction():
    raw_preds, y_true, known_label_map_safe_only, bidir_map = _make_known_label_dataset(n=3000, seed=3)

    baseline_auc = roc_auc_score(y_true, raw_preds)

    safe_overridden = known_label_override(raw_preds, known_label_map_safe_only, asymmetric_safe_direction="positive")
    safe_auc = roc_auc_score(y_true, safe_overridden)

    # naive bidirectional override applied directly (NOT via known_label_override's one-sided
    # safety gate) -- this is exactly the unsafe pattern the module's docstring warns against.
    bidir_overridden = raw_preds.copy()
    for idx, label in bidir_map.items():
        bidir_overridden[idx] = label
    bidir_auc = roc_auc_score(y_true, bidir_overridden)

    assert safe_auc >= baseline_auc, f"safe override regressed: baseline={baseline_auc:.4f}, safe={safe_auc:.4f}"
    assert bidir_auc < safe_auc, (
        f"expected naive bidirectional override to underperform the safe one-directional override "
        f"(demonstrating asymmetric-safety rationale): safe={safe_auc:.4f}, bidirectional={bidir_auc:.4f}"
    )
    gap = safe_auc - bidir_auc
    assert gap > 0.01, f"expected a materially worse bidirectional result, got gap={gap:.4f}"


def test_biz_val_known_label_override_negative_direction_and_bounds_check():
    n = 500
    rng = np.random.default_rng(9)
    raw_preds = rng.uniform(0.4, 0.9, size=n)  # model over-predicts positive broadly
    y_true = np.zeros(n)
    y_true[:50] = 1.0
    rng.shuffle(y_true)

    # high-confidence recovered NEGATIVE labels for rows the (overconfident) model marks positive
    negative_idx = np.where(y_true == 0.0)[0]
    known_map = {int(i): 0.0 for i in rng.choice(negative_idx, size=100, replace=False)}

    baseline_auc = roc_auc_score(y_true, raw_preds)
    overridden = known_label_override(raw_preds, known_map, asymmetric_safe_direction="negative")
    overridden_auc = roc_auc_score(y_true, overridden)

    assert overridden_auc >= baseline_auc
    for idx in known_map:
        assert overridden[idx] == 0.0

    try:
        known_label_override(raw_preds, {n + 5: 1.0}, asymmetric_safe_direction="positive")
        assert False, "expected IndexError for out-of-bounds known_label_map index"
    except IndexError:
        pass
