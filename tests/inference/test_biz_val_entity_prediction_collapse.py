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
    row_predictions, y_row, entity_ids = _make_entity_level_label_dataset(n_entities=300, rows_per_entity=8, seed=0)

    auc_per_row = roc_auc_score(y_row, row_predictions)

    collapsed_mean = collapse_predictions_by_group(row_predictions, entity_ids, stat="mean")
    auc_collapsed = roc_auc_score(y_row, collapsed_mean)

    assert auc_collapsed > auc_per_row, f"expected group-mean collapse to beat noisy per-row predictions when the label is an entity-level property, got collapsed={auc_collapsed:.4f} per_row={auc_per_row:.4f}"


def test_collapse_predictions_by_group_broadcasts_consistently():
    predictions = np.array([0.1, 0.9, 0.5, 0.5, 0.5])
    group = np.array(["a", "a", "b", "b", "b"])
    out = collapse_predictions_by_group(predictions, group, stat="mean")
    np.testing.assert_allclose(out, [0.5, 0.5, 0.5, 0.5, 0.5])
    assert out[0] == out[1]  # all rows of the same group share the same collapsed value
    assert out[2] == out[3] == out[4]


def test_collapse_predictions_by_group_quantile():
    predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    group = np.array(["a", "a", "a", "a", "a"])
    out = collapse_predictions_by_group(predictions, group, stat="quantile", quantile=0.9)
    expected = np.quantile(predictions, 0.9)
    np.testing.assert_allclose(out, [expected] * 5)


def test_collapse_predictions_by_group_invalid_stat_raises():
    import pytest

    with pytest.raises(ValueError):
        collapse_predictions_by_group(np.array([1.0]), np.array(["a"]), stat="median")
