"""Preprocessing edge cases from audit 2026-11-20 cluster 26 (findings 26-F1..F5)."""

import numpy as np
import pandas as pd

from mlframe.preprocessing.cleaning import apply_features_cleaning
from mlframe.preprocessing.cleaning_helpers import map_elementwise_dedup
from mlframe.preprocessing.missing_indicator_pairing import impute_with_missing_indicator
from mlframe.preprocessing.outlier_policy import apply_outlier_policy
from mlframe.preprocessing.temporal_drift_augment import augment_temporal_drift


def test_apply_features_cleaning_accepts_integer_column_labels():
    """Non-string labels used to reach ``df.assign(**kwargs)`` and raise TypeError (26-F1)."""
    df = pd.DataFrame({0: ["a", "rare", "a"], 1: [1, 2, 3]})
    fc = {
        "features_transforms": {0: {"rare": "other"}},
        "constant_features": [],
    }
    out = apply_features_cleaning(df, fc)
    assert list(out[0]) == ["a", "other", "a"]
    assert list(df[0]) == ["a", "rare", "a"]


def test_dedup_map_handles_pd_na_elements():
    """``pd.NA != pd.NA`` is ambiguous under bool(); the dedup gate must not raise (26-F2)."""
    values = np.array(["x", "y", pd.NA] * 2, dtype=object)
    s = pd.Series(values)
    out = map_elementwise_dedup(s, lambda v: "NA" if v is pd.NA else str(v).upper(), sample=2, dup_ratio=0.1)
    assert list(out) == ["X", "Y", "NA", "X", "Y", "NA"]


def test_tree_outlier_score_uses_supplied_train_bounds():
    """Scoring a test frame against train bounds must not re-derive quantiles from the test frame (26-F3)."""
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(size=500)})
    model = RandomForestRegressor(n_estimators=3, random_state=0).fit(train, rng.normal(size=500))
    lo = pd.Series({"a": -0.5})
    hi = pd.Series({"a": 0.5})
    shifted = pd.DataFrame({"a": np.full(50, 5.0)})  # every value far outside the train range
    out = apply_outlier_policy(shifted, model, columns=["a"], bounds=(lo, hi))
    np.testing.assert_allclose(out["outlier_score"].to_numpy(), 1.0)


def test_temporal_augment_with_missing_history_stays_finite():
    """A NaN in an entity's history used to bias the expanding statistics; the augmented rows must stay finite (26-F4).

    Weak check: the function exposes only the augmented rows, not the expanding mean itself.
    """
    df = pd.DataFrame({"e": ["A"] * 4, "t": [1, 2, 3, 4], "x": [10.0, 20.0, np.nan, 30.0]})
    out = augment_temporal_drift(df, entity_col="e", time_col="t", feature_cols=["x"], min_history=2)
    augmented = out[out["_temporal_drift_augmented"]]
    assert len(augmented) >= 1
    assert np.isfinite(augmented["x"].to_numpy()).all()


def test_impute_auto_discovered_string_column_uses_mode_not_median():
    """A string column with a missing value used to reach pandas' median and raise (26-F5)."""
    df = pd.DataFrame({"num": [1.0, np.nan, 3.0], "cat": ["a", None, "a"]})
    out = impute_with_missing_indicator(df)
    assert out["cat"].isna().sum() == 0
    assert out.loc[1, "cat"] == "a"
