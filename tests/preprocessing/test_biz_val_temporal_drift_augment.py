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
