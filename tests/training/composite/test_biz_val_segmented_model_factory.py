"""biz_value test for ``training.composite.SegmentedModelFactory``.

The win: when segments (e.g. airports) have genuinely different feature-target relationships, a single
global model with the segment as a one-hot categorical feature must learn a compromise/interaction that a
linear model can't represent cleanly. Per-segment models recover each segment's own relationship exactly.
Also verifies the lifecycle claim the source technique was built for: adding/removing one segment must not
disturb any other segment's already-fitted model (no full-set retrain needed on entity churn).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlframe.training.composite import SegmentedModelFactory


def _make_airport_dataset(n_per_segment: int, seed: int):
    rng = np.random.default_rng(seed)
    weights = {"JFK": np.array([3.0, 1.0]), "LAX": np.array([-2.0, 0.5]), "ORD": np.array([1.0, -3.0])}
    rows = []
    for airport, w in weights.items():
        x1 = rng.normal(size=n_per_segment)
        x2 = rng.normal(size=n_per_segment)
        y = x1 * w[0] + x2 * w[1] + rng.normal(scale=0.3, size=n_per_segment)
        for i in range(n_per_segment):
            rows.append({"airport": airport, "x1": x1[i], "x2": x2[i], "y": y[i]})
    return pd.DataFrame(rows)


def test_biz_val_segmented_model_factory_beats_global_one_hot_model_mse():
    df = _make_airport_dataset(300, seed=0)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(df))
    train_df = df.iloc[perm[:700]].reset_index(drop=True)
    test_df = df.iloc[perm[700:]].reset_index(drop=True)

    X_global_train = pd.get_dummies(train_df[["airport", "x1", "x2"]], columns=["airport"])
    X_global_test = pd.get_dummies(test_df[["airport", "x1", "x2"]], columns=["airport"]).reindex(columns=X_global_train.columns, fill_value=0)
    global_model = LinearRegression().fit(X_global_train, train_df["y"])
    mse_global = mean_squared_error(test_df["y"], global_model.predict(X_global_test))

    factory = SegmentedModelFactory(estimator_factory=lambda: LinearRegression(), segment_keys=["airport"])
    factory.fit(train_df[["airport", "x1", "x2"]], train_df["y"])
    mse_segmented = mean_squared_error(test_df["y"], factory.predict(test_df[["airport", "x1", "x2"]]))

    improvement = 1.0 - mse_segmented / mse_global
    assert improvement > 0.9, f"expected >90% MSE reduction vs. a global one-hot model, got {improvement:.4f} (global={mse_global:.4f}, segmented={mse_segmented:.4f})"


def test_segmented_model_factory_add_segment_does_not_disturb_other_segments():
    df = _make_airport_dataset(50, seed=2)
    factory = SegmentedModelFactory(estimator_factory=lambda: LinearRegression(), segment_keys=["airport"])
    factory.fit(df[["airport", "x1", "x2"]], df["y"])
    model_jfk_before = factory.segment_models_[("JFK",)]
    model_lax_before = factory.segment_models_[("LAX",)]

    rng = np.random.default_rng(3)
    new_df = pd.DataFrame({"airport": ["DFW"] * 30, "x1": rng.normal(size=30), "x2": rng.normal(size=30)})
    new_y = rng.normal(size=30)
    factory.add_segment(new_df, new_y)

    assert factory.segment_models_[("JFK",)] is model_jfk_before
    assert factory.segment_models_[("LAX",)] is model_lax_before
    assert ("DFW",) in factory.segment_models_

    factory.remove_segment(("DFW",))
    assert ("DFW",) not in factory.segment_models_
    assert factory.segment_models_[("JFK",)] is model_jfk_before


def test_segmented_model_factory_unseen_segment_falls_back_to_global_model():
    df = _make_airport_dataset(50, seed=4)
    train_df = df[df["airport"] != "ORD"].reset_index(drop=True)
    factory = SegmentedModelFactory(estimator_factory=lambda: LinearRegression(), segment_keys=["airport"])
    factory.fit(train_df[["airport", "x1", "x2"]], train_df["y"])
    assert ("ORD",) not in factory.segment_models_

    ord_rows = df[df["airport"] == "ORD"][["airport", "x1", "x2"]].reset_index(drop=True)
    pred = factory.predict(ord_rows)
    expected = factory.global_model_.predict(ord_rows[["x1", "x2"]])
    np.testing.assert_allclose(pred, expected)
