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
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from mlframe.training.composite import compute_row_level_then_average_predictions


def _make_interaction_panel_dataset(n_entities: int, k_rows: int, seed: int):
    """Make interaction panel dataset."""
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
    """Biz val row level then average beats mean aggregation baseline auc."""
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
        X_rows,
        y_row_broadcast,
        entity_ids,
        model_factory=lambda: GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=3),
        n_splits=5,
        random_state=0,
    )
    result_sorted = result.sort("entity_id")
    avg_pred = result_sorted["row_level_avg_pred"].to_numpy()
    auc_row_level = roc_auc_score(y_entity, avg_pred)

    assert auc_row_level > 0.8, f"expected row-level-then-average AUC > 0.8, got {auc_row_level:.4f}"
    assert auc_row_level - auc_baseline > 0.25, (
        f"expected row-level-then-average to beat the mean-aggregation baseline by >0.25 AUC, got row_level={auc_row_level:.4f} vs baseline={auc_baseline:.4f}"
    )


def test_row_level_then_average_mode_b_external_query():
    """Row level then average mode b external query."""
    X_rows, y_entity, entity_ids = _make_interaction_panel_dataset(n_entities=200, k_rows=5, seed=1)
    y_row_broadcast = y_entity[entity_ids]
    X_query, _y_query_entity, query_entity_ids = _make_interaction_panel_dataset(n_entities=50, k_rows=5, seed=2)

    result = compute_row_level_then_average_predictions(
        X_rows,
        y_row_broadcast,
        entity_ids,
        model_factory=lambda: GradientBoostingRegressor(random_state=0, n_estimators=50),
        X_query_rows=X_query,
        query_entity_ids=query_entity_ids,
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
        X_rows,
        y_row_broadcast,
        entity_ids,
        model_factory=lambda: GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=3),
        n_splits=5,
        random_state=0,
        agg_stats=("mean", "max"),
    )
    result_sorted = result.sort("entity_id")
    assert set(result_sorted.columns) == {"entity_id", "row_level_avg_pred_mean", "row_level_avg_pred_max"}

    auc_mean = roc_auc_score(y_entity, result_sorted["row_level_avg_pred_mean"].to_numpy())
    auc_max = roc_auc_score(y_entity, result_sorted["row_level_avg_pred_max"].to_numpy())
    assert auc_max > 0.9, f"expected max-aggregation AUC > 0.9, got {auc_max:.4f}"
    assert auc_max - auc_mean > 0.25, f"expected max-aggregation to beat mean-aggregation by >0.25 AUC, got max={auc_max:.4f} vs mean={auc_mean:.4f}"


def test_biz_val_row_level_low_confidence_flag_identifies_less_reliable_entities():
    """Two entity populations, both averaging to the same class-1 prediction rate, but built from either
    highly-agreeing rows (all children individually predictive) or highly-disagreeing rows (children carry
    contradictory row-level signal, only the entity-level base rate is informative). The low-confidence flag
    should concentrate almost entirely on the disagreeing population, and the flagged group's aggregate
    prediction error should be materially worse than the unflagged group's -- proving the spread signal
    identifies genuinely less-trustworthy group aggregates, not just noise."""
    rng = np.random.default_rng(3)
    n_per_group = 300
    k_rows = 20
    x_rows: list[float] = []
    entity_ids_list: list[int] = []
    y_entity = np.zeros(2 * n_per_group)
    entity = 0

    # Agreeing entities: x's sign consistently matches the label -> every row individually predictive.
    for _ in range(n_per_group):
        label = rng.integers(0, 2)
        y_entity[entity] = label
        sign = 1.0 if label == 1 else -1.0
        vals = sign * rng.uniform(1.0, 2.0, size=k_rows) + rng.normal(scale=0.1, size=k_rows)
        x_rows.extend(vals)
        entity_ids_list.extend([entity] * k_rows)
        entity += 1

    # Disagreeing entities: rows are near-pure noise around 0, uninformative individually -- only a faint
    # entity-level base-rate shift in the mean carries any signal, so all children look ambiguous.
    for _ in range(n_per_group):
        label = rng.integers(0, 2)
        y_entity[entity] = label
        shift = 0.15 if label == 1 else -0.15
        vals = rng.normal(loc=shift, scale=1.5, size=k_rows)
        x_rows.extend(vals)
        entity_ids_list.extend([entity] * k_rows)
        entity += 1

    entity_ids = np.array(entity_ids_list)
    X_rows = pd.DataFrame({"x": x_rows})
    y_row_broadcast = y_entity[entity_ids]
    is_disagreeing_entity = np.arange(2 * n_per_group) >= n_per_group

    result = compute_row_level_then_average_predictions(
        X_rows,
        y_row_broadcast,
        entity_ids,
        model_factory=lambda: GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=3),
        n_splits=5,
        random_state=0,
        flag_low_confidence_quantile=0.6,
    )
    result_sorted = result.sort("entity_id")
    assert set(result_sorted.columns) == {"entity_id", "row_level_avg_pred", "row_level_avg_pred_low_confidence"}

    flagged = result_sorted["row_level_avg_pred_low_confidence"].to_numpy()
    pred = result_sorted["row_level_avg_pred"].to_numpy()

    # The flag should concentrate heavily on the disagreeing (harder, noisier-row) population.
    frac_flagged_from_disagreeing = flagged[is_disagreeing_entity].mean()
    frac_flagged_from_agreeing = flagged[~is_disagreeing_entity].mean()
    assert frac_flagged_from_disagreeing - frac_flagged_from_agreeing > 0.5, (
        f"expected the low-confidence flag to concentrate on disagreeing entities, got "
        f"disagreeing={frac_flagged_from_disagreeing:.3f} vs agreeing={frac_flagged_from_agreeing:.3f}"
    )

    # The flagged group's aggregate predictions should be materially less accurate than the unflagged group's.
    err_flagged = np.abs(pred[flagged] - y_entity[flagged])
    err_unflagged = np.abs(pred[~flagged] - y_entity[~flagged])
    assert err_flagged.mean() - err_unflagged.mean() > 0.15, (
        f"expected flagged-entity error to exceed unflagged-entity error by >0.15, got flagged={err_flagged.mean():.4f} vs unflagged={err_unflagged.mean():.4f}"
    )


def test_row_level_then_average_default_flag_off_is_bit_identical():
    """flag_low_confidence_quantile=None (default) must not alter the pre-existing mean-only output."""
    X_rows, y_entity, entity_ids = _make_interaction_panel_dataset(n_entities=150, k_rows=8, seed=7)
    y_row_broadcast = y_entity[entity_ids]
    kwargs = dict(
        model_factory=lambda: GradientBoostingRegressor(random_state=0, n_estimators=30, max_depth=3),
        n_splits=5,
        random_state=0,
    )
    result_default = compute_row_level_then_average_predictions(X_rows, y_row_broadcast, entity_ids, **kwargs)
    result_explicit_none = compute_row_level_then_average_predictions(X_rows, y_row_broadcast, entity_ids, flag_low_confidence_quantile=None, **kwargs)
    assert result_default.columns == ["entity_id", "row_level_avg_pred"]
    assert result_default.equals(result_explicit_none)


def test_biz_val_row_level_then_average_feature_importance_identifies_informative_subset():
    """A KNOWN subset of child-row features (``x1``, ``x2``) drives the true entity label via their
    interaction; the remaining columns (``noise0``..``noise4``) are pure Gaussian noise carrying zero signal.
    The opt-in ``return_row_feature_importance`` passthrough should rank the two informative features above
    all five noise features -- proving the diagnostic surfaces WHICH row-level features actually drove the
    aggregated OOF score, not just that the score moved."""
    rng = np.random.default_rng(11)
    n_entities = 400
    k_rows = 10
    n_noise = 5
    informative = {"x1", "x2"}

    x1_rows, x2_rows, entity_ids_list = [], [], []
    noise_rows = {f"noise{i}": [] for i in range(n_noise)}
    y_entity = np.zeros(n_entities)
    for e in range(n_entities):
        x1 = rng.normal(size=k_rows)
        x2 = rng.normal(size=k_rows)
        y_entity[e] = 1.0 if ((x1 * x2) > 0).mean() > 0.5 else 0.0
        x1_rows.extend(x1)
        x2_rows.extend(x2)
        entity_ids_list.extend([e] * k_rows)
        for i in range(n_noise):
            noise_rows[f"noise{i}"].extend(rng.normal(size=k_rows))

    entity_ids = np.array(entity_ids_list)
    data = {"x1": x1_rows, "x2": x2_rows, **noise_rows}
    X_rows = pd.DataFrame(data)
    y_row_broadcast = y_entity[entity_ids]

    entity_df, importance_df = compute_row_level_then_average_predictions(
        X_rows,
        y_row_broadcast,
        entity_ids,
        model_factory=lambda: GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=3),
        n_splits=5,
        random_state=0,
        return_row_feature_importance=True,
    )
    assert entity_df.columns == ["entity_id", "row_level_avg_pred"]
    assert set(importance_df.columns) == {"feature", "importance"}
    assert set(importance_df["feature"].to_list()) == informative | set(noise_rows.keys())

    ranked = importance_df.sort("importance", descending=True)["feature"].to_list()
    top_k = set(ranked[: len(informative)])
    precision = len(top_k & informative) / len(informative)
    assert precision == 1.0, f"expected top-{len(informative)} importance to be exactly {informative}, got {ranked}"

    # Per-feature MEAN, not min/max: a GBM splits an x1*x2 interaction signal across many nodes, so
    # individual noise features can still pick up moderate importance from spurious splits -- the ranking
    # (asserted above) is the precise claim, this is a coarser magnitude sanity check on top of it.
    informative_importance = importance_df.filter(pl.col("feature").is_in(list(informative)))["importance"].to_numpy()
    noise_importance = importance_df.filter(~pl.col("feature").is_in(list(informative)))["importance"].to_numpy()
    assert informative_importance.mean() > noise_importance.mean() * 1.5, (
        f"expected informative-feature importance to clearly dominate noise on average, got "
        f"mean_informative={informative_importance.mean():.4f} vs mean_noise={noise_importance.mean():.4f}"
    )


def test_row_level_then_average_default_return_shape_unaffected_by_importance_param_omission():
    """``return_row_feature_importance`` omitted (default False) must not alter the pre-existing single-df
    return contract -- bit-identical to a call that doesn't know the parameter exists."""
    X_rows, y_entity, entity_ids = _make_interaction_panel_dataset(n_entities=120, k_rows=6, seed=13)
    y_row_broadcast = y_entity[entity_ids]
    kwargs = dict(
        model_factory=lambda: GradientBoostingRegressor(random_state=0, n_estimators=20, max_depth=3),
        n_splits=5,
        random_state=0,
    )
    result_omitted = compute_row_level_then_average_predictions(X_rows, y_row_broadcast, entity_ids, **kwargs)
    result_explicit_false = compute_row_level_then_average_predictions(X_rows, y_row_broadcast, entity_ids, return_row_feature_importance=False, **kwargs)
    assert isinstance(result_omitted, pl.DataFrame)
    assert result_omitted.equals(result_explicit_false)


def test_row_level_then_average_entity_order_matches_first_seen():
    """Row level then average entity order matches first seen."""
    X_rows = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    entity_ids = np.array([5, 5, 2, 2])
    y_row_broadcast = np.array([1.0, 1.0, 0.0, 0.0])
    result = compute_row_level_then_average_predictions(
        X_rows,
        y_row_broadcast,
        entity_ids,
        model_factory=lambda: GradientBoostingRegressor(random_state=0, n_estimators=5),
        n_splits=2,
        random_state=0,
    )
    assert result["entity_id"].to_list() == [5, 2]
