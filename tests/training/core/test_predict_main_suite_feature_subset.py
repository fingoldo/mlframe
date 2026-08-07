"""Regression: predict_mlframe_models_suite must subset+reorder the predict frame to the model's
own expected feature schema before calling predict, mirroring predict_from_models's equivalent step.

Bug: a raw column that a training-pipeline stage (e.g. fairness_features tracking) reads but never
adds to the model's actual feature matrix survived unchanged into the predict-time frame handed to
CatBoost's bare predict_proba(X). CatBoost's positional Pool auto-detection from a DataFrame then
misread a later NUMERIC feature's slot as that extra string column -- 'Cannot convert 'A' to float'
-- with nothing pointing back to the real cause (an unsubset predict frame), reproduced live via
tests/training/feature_selection/test_bizvalue_fairness_weights.py::test_fairness_features_emits_per_group_path[cb-*].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training import OutputConfig
from mlframe.training.core import predict_mlframe_models_suite, train_mlframe_models_suite

from tests.training.shared import SimpleFeaturesAndTargetsExtractor


def _make_grouped_classification(n_train=300, n_test=80, n_features=4, seed=0):
    """Small binary classification frame with an extra 'group' string column."""
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test
    X = rng.randn(n_total, n_features)
    logits = X @ np.array([1.5, -1.0, 0.8, -0.5])
    group = rng.choice(["A", "B"], size=n_total, p=[0.7, 0.3])
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n_total) < p).astype(int)
    cols = [f"f_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["group"] = group
    df["target"] = y
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def test_predict_suite_survives_row_wise_extension_plus_untracked_extra_column(tmp_path):
    """predict_mlframe_models_suite must not crash when the predict-time frame carries an extra
    column (here: 'group', read only for fairness tracking) beyond what the fitted model's own
    feature_names_/feature_names_in_ declares -- the frame must be subset+reordered first."""
    pytest.importorskip("catboost")
    train_df, test_df = _make_grouped_classification()
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    data_dir = str(tmp_path / "data")

    _models, _metadata = train_mlframe_models_suite(
        df=train_df,
        target_name="test_target",
        model_name="cb_subset_test",
        features_and_targets_extractor=fte,
        mlframe_models=["cb"],
        behavior_config={"fairness_features": []},
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=data_dir, models_dir="models"),
        verbose=0,
        hyperparams_config={"iterations": 30},
    )
    models_path = f"{data_dir}/models/test_target/cb_subset_test"

    results = predict_mlframe_models_suite(
        df=test_df,
        models_path=models_path,
        features_and_targets_extractor=fte,
        return_probabilities=True,
        verbose=0,
    )
    probs = results.get("probabilities", {})
    assert probs, "predict_mlframe_models_suite returned no probabilities"
    for arr in probs.values():
        arr = np.asarray(arr)
        assert arr.shape[0] == len(test_df)
        assert np.isfinite(arr).all()
