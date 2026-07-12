"""biz_value test for ``preprocessing.augment_temporal_drift``.

The win: a model trained ONLY on each entity's true-last-statement (full-history-standardized) features
generalizes poorly when queried at an earlier point in an entity's lifecycle (production reality -- entities
get scored before their full history is in), because the standardization stats it saw at train time never
matched the truncated-history stats it faces at query time. Augmenting training data with truncated-history,
re-standardized copies (same real label) should measurably improve accuracy on early-vintage queries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from mlframe.preprocessing.temporal_drift_augment import augment_temporal_drift


def _make_panel(n_entities: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    labels = {}
    for entity_id in range(n_entities):
        n_periods = rng.integers(4, 7)
        base = rng.normal(0, 1)
        drift = rng.normal(0, 0.4)
        series = base + drift * np.arange(n_periods) + rng.normal(0, 0.3, size=n_periods)
        # real label driven by how far the TRUE last value stands relative to that entity's full history.
        z_last = (series[-1] - series.mean()) / (series.std(ddof=1) + 1e-9)
        label = int(z_last > 0.3)
        labels[entity_id] = label
        for t, val in enumerate(series):
            rows.append({"entity_id": entity_id, "t": t, "x": val, "y": label})
    return pd.DataFrame(rows), labels


def test_biz_val_temporal_drift_augment_improves_early_vintage_generalization():
    train_df, _ = _make_panel(n_entities=600, seed=0)
    test_df, _ = _make_panel(n_entities=300, seed=1)

    # baseline training set: one row per entity, the TRUE last statement, standardized against its OWN full
    # history (the label itself is defined this way) -- typical panel-to-tabular usage.
    train_last_rows = []
    for entity_id, grp in train_df.sort_values(["entity_id", "t"]).groupby("entity_id"):
        z = (grp["x"].iloc[-1] - grp["x"].mean()) / (grp["x"].std(ddof=1) + 1e-9)
        train_last_rows.append({"x": z, "y": grp["y"].iloc[0]})
    train_last = pd.DataFrame(train_last_rows)

    augmented = augment_temporal_drift(train_df, entity_col="entity_id", time_col="t", feature_cols=["x"], n_drop_options=(1, 2))
    assert augmented["_temporal_drift_augmented"].sum() > 0
    augmented_only = augmented.loc[augmented["_temporal_drift_augmented"], ["x", "y"]]
    train_augmented = pd.concat([train_last, augmented_only], axis=0, ignore_index=True)

    # early-vintage query set: for each test entity, drop the last 2 periods and standardize against the
    # truncated history only -- the production scenario of scoring an entity before its full history exists.
    # This truncated z-score is a NOISIER version of the full-history z-score the label was defined from, so
    # a model that only ever saw full-history-standardized inputs at train time is miscalibrated against it.
    early_rows = []
    for entity_id, grp in test_df.sort_values(["entity_id", "t"]).groupby("entity_id"):
        if len(grp) <= 2:
            continue
        truncated = grp.iloc[:-2]
        if len(truncated) < 2:
            continue
        z = (truncated["x"].iloc[-1] - truncated["x"].mean()) / (truncated["x"].std(ddof=1) + 1e-9)
        early_rows.append({"x": z, "y": grp["y"].iloc[0]})
    early_test = pd.DataFrame(early_rows)

    model_baseline = LogisticRegression().fit(train_last[["x"]], train_last["y"])
    model_augmented = LogisticRegression().fit(train_augmented[["x"]], train_augmented["y"])

    loss_baseline = log_loss(early_test["y"], model_baseline.predict_proba(early_test[["x"]])[:, 1])
    loss_augmented = log_loss(early_test["y"], model_augmented.predict_proba(early_test[["x"]])[:, 1])

    assert loss_augmented < loss_baseline, (
        f"temporal-drift augmentation should improve calibration on early-vintage (truncated-history) queries: "
        f"augmented_log_loss={loss_augmented:.4f} baseline_log_loss={loss_baseline:.4f}"
    )


def _make_normal_history_panel(n_entities: int, seed: int):
    """Same generative process as the module-level ``_make_panel`` above (typical entities)."""
    rng = np.random.default_rng(seed)
    rows = []
    for entity_id in range(n_entities):
        n_periods = rng.integers(4, 7)
        base = rng.normal(0, 1)
        drift = rng.normal(0, 0.4)
        series = base + drift * np.arange(n_periods) + rng.normal(0, 0.3, size=n_periods)
        z_last = (series[-1] - series.mean()) / (series.std(ddof=1) + 1e-9)
        label = int(z_last > 0.3)
        for t, val in enumerate(series):
            rows.append({"entity_id": entity_id, "t": t, "x": val, "y": label})
    return pd.DataFrame(rows)


def _make_long_noisy_history_panel(n_entities: int, n_periods: int, seed: int, id_offset: int):
    """A minority of entities with MUCH longer history and a much noisier per-period signal.

    Same label rule (``z_last > 0.3`` on the entity's own full history) so real rows are legitimate,
    but with a long history each of these entities is eligible for many ``n_drop`` depths, and at
    heavy truncation the expanding z-score of such a noisy series is a poor proxy for the label --
    exactly the "small subset of entities with much longer, noisier history dominates augmented-row
    volume" case ``weight_by_recency`` targets.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_entities):
        entity_id = id_offset + i
        base = rng.normal(0, 1)
        series = base + rng.normal(0, 3.0, size=n_periods)
        z_last = (series[-1] - series.mean()) / (series.std(ddof=1) + 1e-9)
        label = int(z_last > 0.3)
        for t, val in enumerate(series):
            rows.append({"entity_id": entity_id, "t": t, "x": val, "y": label})
    return pd.DataFrame(rows)


def _last_row_standardized(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for entity_id, grp in df.sort_values(["entity_id", "t"]).groupby("entity_id"):
        z = (grp["x"].iloc[-1] - grp["x"].mean()) / (grp["x"].std(ddof=1) + 1e-9)
        out.append({"x": z, "y": grp["y"].iloc[0]})
    return pd.DataFrame(out)


def test_biz_val_temporal_drift_augment_weight_by_recency_beats_naive_equal_weight():
    normal_entities = _make_normal_history_panel(n_entities=500, seed=0)
    # a small minority (100 of 600, ~17%) of entities with a much longer, much noisier history --
    # they numerically dominate the naive augmented pool (many eligible n_drop depths each) despite
    # being a small fraction of the true entity population.
    long_noisy_entities = _make_long_noisy_history_panel(n_entities=100, n_periods=40, seed=5, id_offset=100_000)
    train_df = pd.concat([normal_entities, long_noisy_entities], axis=0, ignore_index=True)
    test_df = _make_normal_history_panel(n_entities=300, seed=1)

    train_last = _last_row_standardized(train_df)

    n_drop_options = tuple(range(1, 38))

    naive = augment_temporal_drift(
        train_df, entity_col="entity_id", time_col="t", feature_cols=["x"], n_drop_options=n_drop_options, weight_by_recency=False
    )
    weighted = augment_temporal_drift(
        train_df,
        entity_col="entity_id",
        time_col="t",
        feature_cols=["x"],
        n_drop_options=n_drop_options,
        weight_by_recency=True,
        min_augmented_weight=0.02,
    )
    assert "_sample_weight" not in naive.columns
    assert "_sample_weight" in weighted.columns

    naive_augmented_only = naive.loc[naive["_temporal_drift_augmented"], ["x", "y"]]
    train_naive = pd.concat([train_last, naive_augmented_only], axis=0, ignore_index=True)

    weighted_augmented_only = weighted.loc[weighted["_temporal_drift_augmented"], ["x", "y", "_sample_weight"]]
    train_weighted = pd.concat(
        [train_last.assign(_sample_weight=1.0), weighted_augmented_only], axis=0, ignore_index=True
    )

    # realistic production query: moderate truncation (drop last 1 period) on the NORMAL entity
    # population, not the long-noisy-history minority that pollutes the naive augmented pool.
    early_rows = []
    for entity_id, grp in test_df.sort_values(["entity_id", "t"]).groupby("entity_id"):
        truncated = grp.iloc[:-1]
        if len(truncated) < 2:
            continue
        z = (truncated["x"].iloc[-1] - truncated["x"].mean()) / (truncated["x"].std(ddof=1) + 1e-9)
        early_rows.append({"x": z, "y": grp["y"].iloc[0]})
    early_test = pd.DataFrame(early_rows)

    model_naive = LogisticRegression().fit(train_naive[["x"]], train_naive["y"])
    model_weighted = LogisticRegression().fit(
        train_weighted[["x"]], train_weighted["y"], sample_weight=train_weighted["_sample_weight"]
    )

    loss_naive = log_loss(early_test["y"], model_naive.predict_proba(early_test[["x"]])[:, 1])
    loss_weighted = log_loss(early_test["y"], model_weighted.predict_proba(early_test[["x"]])[:, 1])

    assert loss_weighted < loss_naive * 0.97, (
        f"weight_by_recency should down-weight the noisy long-history minority's heavily-truncated synthetic "
        f"rows and measurably beat naive equal-weight augmentation on the normal-entity query distribution: "
        f"weighted_log_loss={loss_weighted:.4f} naive_log_loss={loss_naive:.4f}"
    )


def test_augment_temporal_drift_weight_by_recency_default_off_is_bit_identical():
    df = pd.DataFrame(
        {
            "entity_id": [1, 1, 1, 1, 2, 2, 2, 2, 2],
            "t": [0, 1, 2, 3, 0, 1, 2, 3, 4],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 6.0, 7.0, 8.0],
            "y": [0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
    default_result = augment_temporal_drift(df, entity_col="entity_id", time_col="t", feature_cols=["x"], n_drop_options=(1, 2))
    explicit_off_result = augment_temporal_drift(
        df, entity_col="entity_id", time_col="t", feature_cols=["x"], n_drop_options=(1, 2), weight_by_recency=False
    )
    pd.testing.assert_frame_equal(default_result, explicit_off_result)
    assert "_sample_weight" not in default_result.columns


def test_augment_temporal_drift_marks_augmented_rows_and_preserves_originals():
    df = pd.DataFrame(
        {
            "entity_id": [1, 1, 1, 2, 2, 2, 2],
            "t": [0, 1, 2, 0, 1, 2, 3],
            "x": [1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0],
            "y": [0, 0, 0, 1, 1, 1, 1],
        }
    )
    result = augment_temporal_drift(df, entity_col="entity_id", time_col="t", feature_cols=["x"], n_drop_options=(1,))
    assert (~result["_temporal_drift_augmented"]).sum() == len(df)
    assert result["_temporal_drift_augmented"].sum() > 0


def test_augment_temporal_drift_no_eligible_entities_returns_originals_only():
    df = pd.DataFrame({"entity_id": [1, 2], "t": [0, 0], "x": [1.0, 2.0], "y": [0, 1]})
    result = augment_temporal_drift(df, entity_col="entity_id", time_col="t", feature_cols=["x"], n_drop_options=(1,), min_history=2)
    assert len(result) == len(df)
    assert not result["_temporal_drift_augmented"].any()
