"""biz_value test for ``training.composite.compute_row_level_then_average_predictions``.

The win: when an entity's true label depends on a per-row INTERACTION pattern (fraction of child rows where
``x1*x2 > 0``) rather than the child rows' marginal means, hand-aggregating child rows into per-entity mean
features BEFORE modeling destroys that signal entirely (the mean of ``x1``/``x2`` separately carries no
information about how often they share a sign). Training directly on the raw child rows (with the parent
label broadcast down) and averaging the row-level model's predictions back to the entity level recovers the
interaction-driven signal instead -- mirroring the Home Credit 3rd place's "skip aggregation, average
predictions" technique.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from mlframe.training.composite import compute_row_level_then_average_predictions


def _make_interaction_panel_dataset(n_entities: int, k_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    x1_rows, x2_rows, entity_ids_list = [], [], []
    y_entity = np.zeros(n_entities)
    for e in range(n_entities):
        x1 = rng.normal(size=k_rows)
        x2 = rng.normal(size=k_rows)
        positive = (x1 * x2) > 0
        y_entity[e] = 1.0 if positive.mean() > 0.5 else 0.0
        x1_rows.extend(x1)
        x2_rows.extend(x2)
        entity_ids_list.extend([e] * k_rows)
    entity_ids = np.array(entity_ids_list)
    X_rows = pd.DataFrame({"x1": x1_rows, "x2": x2_rows})
    return X_rows, y_entity, entity_ids


def test_biz_val_row_level_then_average_beats_mean_aggregation_baseline_auc():
    X_rows, y_entity, entity_ids = _make_interaction_panel_dataset(n_entities=600, k_rows=10, seed=0)
    y_row_broadcast = y_entity[entity_ids]
    n_entities = y_entity.shape[0]

    # Baseline: hand-aggregate child rows into per-entity mean features BEFORE modeling.
    df = pd.DataFrame({"entity_id": entity_ids, "x1": X_rows["x1"], "x2": X_rows["x2"]})
    agg = df.groupby("entity_id")[["x1", "x2"]].mean().reindex(range(n_entities)).to_numpy()
    oof_baseline = np.zeros(n_entities)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, val_idx in kf.split(agg):
        reg = GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=3)
        reg.fit(agg[train_idx], y_entity[train_idx])
        oof_baseline[val_idx] = reg.predict(agg[val_idx])
    auc_baseline = roc_auc_score(y_entity, oof_baseline)

    result = compute_row_level_then_average_predictions(
        X_rows, y_row_broadcast, entity_ids,
        model_factory=lambda: GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=3),
        n_splits=5, random_state=0,
    )
    result_sorted = result.sort("entity_id")
    avg_pred = result_sorted["row_level_avg_pred"].to_numpy()
    auc_row_level = roc_auc_score(y_entity, avg_pred)

    assert auc_row_level > 0.8, f"expected row-level-then-average AUC > 0.8, got {auc_row_level:.4f}"
    assert auc_row_level - auc_baseline > 0.25, (
        f"expected row-level-then-average to beat the mean-aggregation baseline by >0.25 AUC, "
        f"got row_level={auc_row_level:.4f} vs baseline={auc_baseline:.4f}"
    )


def test_row_level_then_average_mode_b_external_query():
    X_rows, y_entity, entity_ids = _make_interaction_panel_dataset(n_entities=200, k_rows=5, seed=1)
    y_row_broadcast = y_entity[entity_ids]
    X_query, y_query_entity, query_entity_ids = _make_interaction_panel_dataset(n_entities=50, k_rows=5, seed=2)

    result = compute_row_level_then_average_predictions(
        X_rows, y_row_broadcast, entity_ids,
        model_factory=lambda: GradientBoostingRegressor(random_state=0, n_estimators=50),
        X_query_rows=X_query, query_entity_ids=query_entity_ids,
    )
    assert result.shape[0] == 50
    assert set(result["entity_id"].to_list()) == set(range(50))


def test_biz_val_row_level_agg_stats_max_beats_mean_for_outlier_driven_label():
    """Home Credit 5th place's multi-stat extension: when an entity's true label depends on the PRESENCE of
    a single extreme child row (not the average), max-aggregation of row-level OOF scores should recover it
    far better than mean-aggregation, which dilutes one outlier among many normal rows."""
    rng = np.random.default_rng(0)
    n_entities = 600
    k_rows = 30
    x_rows: list[float] = []
    entity_ids_list: list[int] = []
    y_entity = np.zeros(n_entities)
    for e in range(n_entities):
        vals = rng.normal(size=k_rows)
        has_outlier = rng.random() < 0.4
        if has_outlier:
            vals[rng.integers(0, k_rows)] = rng.uniform(2.5, 4)
        y_entity[e] = 1.0 if has_outlier else 0.0
        x_rows.extend(vals)
        entity_ids_list.extend([e] * k_rows)

    entity_ids = np.array(entity_ids_list)
    X_rows = pd.DataFrame({"x": x_rows})
    y_row_broadcast = y_entity[entity_ids]

    result = compute_row_level_then_average_predictions(
        X_rows, y_row_broadcast, entity_ids,
        model_factory=lambda: GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=3),
        n_splits=5, random_state=0, agg_stats=("mean", "max"),
    )
    result_sorted = result.sort("entity_id")
    assert set(result_sorted.columns) == {"entity_id", "row_level_avg_pred_mean", "row_level_avg_pred_max"}

    auc_mean = roc_auc_score(y_entity, result_sorted["row_level_avg_pred_mean"].to_numpy())
    auc_max = roc_auc_score(y_entity, result_sorted["row_level_avg_pred_max"].to_numpy())
    assert auc_max > 0.9, f"expected max-aggregation AUC > 0.9, got {auc_max:.4f}"
    assert auc_max - auc_mean > 0.25, f"expected max-aggregation to beat mean-aggregation by >0.25 AUC, got max={auc_max:.4f} vs mean={auc_mean:.4f}"


def test_row_level_then_average_entity_order_matches_first_seen():
    X_rows = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    entity_ids = np.array([5, 5, 2, 2])
    y_row_broadcast = np.array([1.0, 1.0, 0.0, 0.0])
    result = compute_row_level_then_average_predictions(
        X_rows, y_row_broadcast, entity_ids,
        model_factory=lambda: GradientBoostingRegressor(random_state=0, n_estimators=5),
        n_splits=2, random_state=0,
    )
    assert result["entity_id"].to_list() == [5, 2]
