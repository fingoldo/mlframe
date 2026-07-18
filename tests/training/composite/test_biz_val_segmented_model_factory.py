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
    """Make airport dataset."""
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
    """Biz val segmented model factory beats global one hot model mse."""
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
    assert (
        improvement > 0.9
    ), f"expected >90% MSE reduction vs. a global one-hot model, got {improvement:.4f} (global={mse_global:.4f}, segmented={mse_segmented:.4f})"


def test_segmented_model_factory_add_segment_does_not_disturb_other_segments():
    """Segmented model factory add segment does not disturb other segments."""
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


def _make_regional_mixed_size_dataset(n_large: int, n_tiny: int, n_tiny_per_region: int, seed: int):
    """Three regions, each with its OWN distinct linear relationship: one large, well-populated segment per
    region plus several tiny ones (too few rows to fit a reliable per-segment model) that share the SAME
    underlying relationship as their region's large segment (small per-region noise on top). This is the
    shape the hierarchical-shrinkage fallback targets: raw per-segment fitting overfits the tiny segments
    (too few rows), a blunt full-GLOBAL fallback averages across regions with genuinely different
    relationships (loses all region-specific signal), but pooling a tiny segment with its own region's
    large-segment model (``shrinkage_parent_keys=["region"]``) recovers the right relationship from a
    well-estimated neighbor instead of either extreme."""
    rng = np.random.default_rng(seed)
    region_weights = {"A": np.array([3.0, 1.0]), "B": np.array([-2.0, 0.5]), "C": np.array([1.0, -3.0])}
    rows = []
    for region, w in region_weights.items():
        x1 = rng.normal(size=n_large)
        x2 = rng.normal(size=n_large)
        y = x1 * w[0] + x2 * w[1] + rng.normal(scale=0.3, size=n_large)
        for i in range(n_large):
            rows.append({"region": region, "airport": f"{region}_BIG", "x1": x1[i], "x2": x2[i], "y": y[i]})
        for t in range(n_tiny_per_region):
            w_tiny = w + rng.normal(scale=0.1, size=2)  # same relationship as the region, plus small per-segment noise
            x1 = rng.normal(size=n_tiny)
            x2 = rng.normal(size=n_tiny)
            y = x1 * w_tiny[0] + x2 * w_tiny[1] + rng.normal(scale=0.3, size=n_tiny)
            for i in range(n_tiny):
                rows.append({"region": region, "airport": f"{region}_TINY{t}", "x1": x1[i], "x2": x2[i], "y": y[i]})
    return pd.DataFrame(rows)


def test_biz_val_segmented_model_factory_hierarchical_shrinkage_beats_raw_and_global_fallback_mse():
    """Biz val segmented model factory hierarchical shrinkage beats raw and global fallback mse."""
    train_df = _make_regional_mixed_size_dataset(n_large=300, n_tiny=3, n_tiny_per_region=2, seed=10)
    test_df = _make_regional_mixed_size_dataset(
        n_large=50, n_tiny=100, n_tiny_per_region=2, seed=11
    )  # more held-out rows per tiny segment to score it reliably

    def _fit_predict(**kwargs):
        """Fit predict."""
        kwargs.setdefault("min_segment_rows", 2)
        factory = SegmentedModelFactory(estimator_factory=lambda: LinearRegression(), segment_keys=["region", "airport"], **kwargs)
        factory.fit(train_df[["region", "airport", "x1", "x2"]], train_df["y"])
        return factory.predict(test_df[["region", "airport", "x1", "x2"]])

    mse_raw = mean_squared_error(test_df["y"], _fit_predict())  # per-segment fit-anyway: overfits on 3-row tiny segments
    mse_global = mean_squared_error(test_df["y"], _fit_predict(min_segment_rows=5))  # blunt fallback: tiny segments lose their region-specific signal entirely
    mse_shrinkage = mean_squared_error(test_df["y"], _fit_predict(shrinkage_min_rows=30, shrinkage_parent_keys=["region"], shrinkage_k=10.0))

    improvement_vs_raw = 1.0 - mse_shrinkage / mse_raw
    improvement_vs_global = 1.0 - mse_shrinkage / mse_global
    assert (
        improvement_vs_raw > 0.3
    ), f"expected >30% MSE reduction vs. raw per-segment fit-anyway, got {improvement_vs_raw:.4f} (raw={mse_raw:.4f}, shrinkage={mse_shrinkage:.4f})"
    assert (
        improvement_vs_global > 0.3
    ), f"expected >30% MSE reduction vs. blunt full-global fallback, got {improvement_vs_global:.4f} (global={mse_global:.4f}, shrinkage={mse_shrinkage:.4f})"


def test_segmented_model_factory_shrinkage_disabled_by_default_is_bit_identical():
    """Segmented model factory shrinkage disabled by default is bit identical."""
    df = _make_airport_dataset(150, seed=5)
    factory_default = SegmentedModelFactory(estimator_factory=lambda: LinearRegression(), segment_keys=["airport"])
    factory_default.fit(df[["airport", "x1", "x2"]], df["y"])
    pred_default = factory_default.predict(df[["airport", "x1", "x2"]])

    factory_explicit_none = SegmentedModelFactory(estimator_factory=lambda: LinearRegression(), segment_keys=["airport"], shrinkage_min_rows=None)
    factory_explicit_none.fit(df[["airport", "x1", "x2"]], df["y"])
    pred_explicit_none = factory_explicit_none.predict(df[["airport", "x1", "x2"]])

    np.testing.assert_array_equal(pred_default, pred_explicit_none)
    assert factory_default.parent_models_ == {}
    assert factory_default.shrinkage_weights_ == {}


def test_segmented_model_factory_unseen_segment_falls_back_to_global_model():
    """Segmented model factory unseen segment falls back to global model."""
    df = _make_airport_dataset(50, seed=4)
    train_df = df[df["airport"] != "ORD"].reset_index(drop=True)
    factory = SegmentedModelFactory(estimator_factory=lambda: LinearRegression(), segment_keys=["airport"])
    factory.fit(train_df[["airport", "x1", "x2"]], train_df["y"])
    assert ("ORD",) not in factory.segment_models_

    ord_rows = df[df["airport"] == "ORD"][["airport", "x1", "x2"]].reset_index(drop=True)
    pred = factory.predict(ord_rows)
    expected = factory.global_model_.predict(ord_rows[["x1", "x2"]])
    np.testing.assert_allclose(pred, expected)
