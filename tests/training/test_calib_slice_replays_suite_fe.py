"""The disjoint calib slice must reach the calib predict with the same features the model was fit on.

The calib slice is carved from the raw frame before the suite-level FE stages, and used to be predicted on raw: a model
fit with suite-level composite FE columns (here cross-sectional neighbours, ``xsnn_*``) then failed on the calib predict
(LightGBM: "train and valid dataset categorical_feature do not match"), the failure was swallowed with a warning, and
finalize lost its conformal residuals.
"""
from __future__ import annotations

import numpy as np
import pytest


def _frame(n: int = 1500, seed: int = 3):
    """A time-grouped frame whose target depends on two numeric features and a category, for a suite fit with cross-sectional-neighbour FE."""
    import polars as pl

    rng = np.random.default_rng(seed)
    f0 = rng.normal(size=n).astype(np.float32)
    f1 = rng.normal(size=n).astype(np.float32)
    cat = rng.choice(["a", "b", "c"], size=n)
    y = 2.0 * f0 - f1 + (cat == "a") + rng.normal(scale=0.3, size=n)
    return pl.DataFrame({"time_id": np.repeat(np.arange(n // 25), 25)[:n], "f0": f0, "f1": f1, "cat": cat, "target": y})


def test_calib_predict_sees_suite_composite_fe_columns(tmp_path):
    """The calib slice's own predictions must equal predict_from_models() on the same raw rows -- both must see the suite's composite FE columns."""
    pytest.importorskip("lightgbm")
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import (
        BaselineDiagnosticsConfig, DummyBaselinesConfig, OutputConfig, PreprocessingBackendConfig, ReportingConfig, TrainingBehaviorConfig,
    )
    from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig, TrainingSplitConfig
    from .shared import SimpleFeaturesAndTargetsExtractor

    frame = _frame()
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)
    models, metadata = train_mlframe_models_suite(
        df=frame,
        target_name="calib_fe",
        model_name="calib_fe",
        features_and_targets_extractor=fte,
        mlframe_models=["lgb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        pipeline_config=PreprocessingBackendConfig(prefer_polarsds=False, scaler_name=None, imputer_strategy=None),
        preprocessing_extensions=PreprocessingExtensionsConfig(
            cross_sectional_neighbors_snapshot_col="time_id", cross_sectional_neighbors_feature_cols=["f0", "f1"], cross_sectional_neighbors_k=3,
        ),
        split_config=TrainingSplitConfig(test_size=0.2, val_size=0.1, calib_size=0.15, random_seed=3),
        behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False),
        hyperparams_config={"iterations": 30},
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(honest_estimator_diagnostics=False),
        enable_target_distribution_analyzer=False,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        verbose=0,
    )
    from mlframe.training.core._predict_main_from_models import predict_from_models

    entries = [e for by_name in models.values() for lst in by_name.values() if isinstance(lst, list) for e in lst]
    assert len(entries) == 1
    entry = entries[0]
    assert any(str(c).startswith("xsnn_") for c in entry.columns), f"fixture no longer fits on the composite FE columns: {entry.columns}"
    calib_preds = getattr(entry, "calib_preds", None)
    assert calib_preds is not None, "calib-slice predict failed: the calib frame lacked the suite-level FE columns"

    # The calib rows are unseen at fit time, so their calib predictions must be exactly what predict() returns for them.
    # Raw calib predicting NaN-filled the missing xsnn_* columns (LGBM reindexes to feature_names_in_), silently diverging.
    y_all = frame["target"].to_numpy()
    row_of = {float(v): i for i, v in enumerate(y_all)}
    calib_rows = np.array([row_of[float(v)] for v in np.asarray(entry.calib_target, dtype=np.float64).ravel()])
    res = predict_from_models(frame[calib_rows], models, metadata, features_and_targets_extractor=fte)
    (expected,) = list(res["predictions"].values())
    np.testing.assert_allclose(np.asarray(calib_preds, dtype=np.float64), np.asarray(expected, dtype=np.float64).ravel(), rtol=1e-6, atol=1e-9)
