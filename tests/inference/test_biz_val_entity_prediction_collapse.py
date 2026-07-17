"""biz_value test for ``inference.entity_prediction_collapse.collapse_predictions_by_group``.

The win (6th_ieee-cis-fraud-detection.md): when the true label is genuinely an ENTITY-level property (e.g.
fraud risk is a customer attribute, not independent per transaction) but a per-row model produces noisy,
inconsistent predictions within the same entity, collapsing to a group statistic and broadcasting it back
should recover a materially better score than the noisy per-row predictions.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from mlframe.inference.entity_prediction_collapse import collapse_predictions_by_group


def _make_entity_level_label_dataset(n_entities: int, rows_per_entity: int, seed: int):
    """Helper that make entity level label dataset."""
    rng = np.random.default_rng(seed)
    entity_ids = np.repeat(np.arange(n_entities), rows_per_entity)
    n = len(entity_ids)

    entity_risk = rng.uniform(0, 1, n_entities)
    y = (rng.uniform(size=n_entities) < entity_risk).astype(int)
    y_row = np.repeat(y, rows_per_entity)

    # per-row model prediction: correlated with the TRUE entity risk but with substantial per-row noise
    # (the model sees per-transaction features that only partially reflect the entity's underlying risk).
    entity_risk_row = np.repeat(entity_risk, rows_per_entity)
    row_predictions = np.clip(entity_risk_row + rng.normal(scale=0.3, size=n), 0, 1)

    return row_predictions, y_row, entity_ids


def test_biz_val_collapse_predictions_by_group_beats_noisy_per_row():
    """Collapse predictions by group beats noisy per row."""
    row_predictions, y_row, entity_ids = _make_entity_level_label_dataset(n_entities=300, rows_per_entity=8, seed=0)

    auc_per_row = roc_auc_score(y_row, row_predictions)

    collapsed_mean = collapse_predictions_by_group(row_predictions, entity_ids, stat="mean")
    auc_collapsed = roc_auc_score(y_row, collapsed_mean)

    assert auc_collapsed > auc_per_row, (
        f"expected group-mean collapse to beat noisy per-row predictions when the label is an entity-level property, got collapsed={auc_collapsed:.4f} per_row={auc_per_row:.4f}"
    )


def test_collapse_predictions_by_group_broadcasts_consistently():
    """Collapse predictions by group broadcasts consistently."""
    predictions = np.array([0.1, 0.9, 0.5, 0.5, 0.5])
    group = np.array(["a", "a", "b", "b", "b"])
    out = collapse_predictions_by_group(predictions, group, stat="mean")
    np.testing.assert_allclose(out, [0.5, 0.5, 0.5, 0.5, 0.5])
    assert out[0] == out[1]  # all rows of the same group share the same collapsed value
    assert out[2] == out[3] == out[4]


def test_collapse_predictions_by_group_quantile():
    """Collapse predictions by group quantile."""
    predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    group = np.array(["a", "a", "a", "a", "a"])
    out = collapse_predictions_by_group(predictions, group, stat="quantile", quantile=0.9)
    expected = np.quantile(predictions, 0.9)
    np.testing.assert_allclose(out, [expected] * 5)


def test_collapse_predictions_by_group_invalid_stat_raises():
    """Collapse predictions by group invalid stat raises."""
    import pytest

    with pytest.raises(ValueError):
        collapse_predictions_by_group(np.array([1.0]), np.array(["a"]), stat="median")


def _make_mixed_reliability_dataset(n_entities: int, recent_per_entity: int, stale_per_entity: int, seed: int):
    """Each entity has RECENT rows whose predictions track the true entity risk, and STALE rows that carry
    no signal at all (pure noise unrelated to the label) -- e.g. predictions computed from an old feature
    snapshot before something material changed for that customer. An unweighted mean lets the noisy stale
    rows dilute the informative recent ones; a recency-weighted mean should down-weight the noise instead.
    """
    rng = np.random.default_rng(seed)
    entity_risk = rng.uniform(0, 1, n_entities)
    y = (rng.uniform(size=n_entities) < entity_risk).astype(int)

    rows_per_entity = recent_per_entity + stale_per_entity
    entity_ids = np.repeat(np.arange(n_entities), rows_per_entity)
    y_row = np.repeat(y, rows_per_entity)
    entity_risk_row = np.repeat(entity_risk, rows_per_entity)

    is_recent = np.tile(np.array([True] * recent_per_entity + [False] * stale_per_entity), n_entities)

    row_predictions = np.empty(entity_ids.shape[0])
    # recent rows: mildly noisy but genuinely correlated with the entity's true risk.
    row_predictions[is_recent] = np.clip(entity_risk_row[is_recent] + rng.normal(scale=0.15, size=is_recent.sum()), 0, 1)
    # stale rows: pure noise, independent of the true entity risk entirely.
    row_predictions[~is_recent] = rng.uniform(0, 1, size=(~is_recent).sum())

    weights = np.where(is_recent, 5.0, 0.2)  # recency/confidence score: recent rows count far more.

    return row_predictions, y_row, entity_ids, weights


def test_biz_val_collapse_predictions_by_group_weighted_beats_unweighted_with_stale_noise():
    """Collapse predictions by group weighted beats unweighted with stale noise."""
    row_predictions, y_row, entity_ids, weights = _make_mixed_reliability_dataset(n_entities=300, recent_per_entity=3, stale_per_entity=5, seed=1)

    unweighted = collapse_predictions_by_group(row_predictions, entity_ids, stat="mean")
    auc_unweighted = roc_auc_score(y_row, unweighted)

    weighted = collapse_predictions_by_group(row_predictions, entity_ids, stat="mean", weights=weights)
    auc_weighted = roc_auc_score(y_row, weighted)

    assert auc_weighted > auc_unweighted + 0.05, (
        f"expected recency-weighted collapse to clearly beat unweighted collapse when most rows per entity are "
        f"stale noise, got weighted={auc_weighted:.4f} unweighted={auc_unweighted:.4f}"
    )


def test_collapse_predictions_by_group_weights_none_matches_original_unweighted_path():
    # opt-in contract: omitting `weights` must reproduce the exact original unweighted output.
    """Collapse predictions by group weights none matches original unweighted path."""
    predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    group = np.array(["a", "a", "b", "b", "b", "c"])

    mean_out = collapse_predictions_by_group(predictions, group, stat="mean")
    mean_out_explicit_none = collapse_predictions_by_group(predictions, group, stat="mean", weights=None)
    np.testing.assert_array_equal(mean_out, mean_out_explicit_none)

    quantile_out = collapse_predictions_by_group(predictions, group, stat="quantile", quantile=0.9)
    quantile_out_explicit_none = collapse_predictions_by_group(predictions, group, stat="quantile", quantile=0.9, weights=None)
    np.testing.assert_array_equal(quantile_out, quantile_out_explicit_none)


def test_collapse_predictions_by_group_weighted_quantile_equal_weights_between_group_extremes():
    """Collapse predictions by group weighted quantile equal weights between group extremes."""
    predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    group = np.array(["a", "a", "a", "a", "a"])
    weights = np.ones(5)
    out = collapse_predictions_by_group(predictions, group, stat="quantile", quantile=0.9, weights=weights)
    assert predictions.min() <= out[0] <= predictions.max()


def test_collapse_predictions_by_group_weighted_rejects_all_zero_weight_group():
    """Collapse predictions by group weighted rejects all zero weight group."""
    import pytest

    predictions = np.array([1.0, 2.0])
    group = np.array(["a", "a"])
    weights = np.array([0.0, 0.0])
    with pytest.raises(ValueError):
        collapse_predictions_by_group(predictions, group, stat="mean", weights=weights)
