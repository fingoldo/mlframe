"""Unit + e2e tests for TTA uncertainty evaluation (Workstream B).

``evaluate_tta_quality`` assesses TTA predictive uncertainty on a held-out (X, y); it is also wired into
the suite (``behavior_config.uncertainty_eval=True``) to stamp test-split TTA quality into
``metadata["uncertainty_eval"]`` after each regression model trains (e2e test below).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training._uncertainty_eval import _narrow_numeric_frame, evaluate_tta_quality


def test_evaluate_tta_quality_metrics():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 3))
    y = X[:, 0] * 2 - X[:, 1]

    def f(Z):
        return Z[:, 0] * 2 - Z[:, 1] + 0.5 * np.sin(11 * Z[:, 2])

    rep = evaluate_tta_quality(f, X, y, n=24, sigma_scale=0.2)
    assert set(rep) == {"rmse_point", "rmse_tta", "tta_rmse_gain", "spread_error_corr", "mean_spread"}
    assert rep["mean_spread"] > 0
    assert rep["rmse_tta"] <= rep["rmse_point"] + 1e-9  # averaging should not hurt here


def test_narrow_numeric_frame_pandas_and_reject_non_numeric():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": ["x", "y"]})
    arr = _narrow_numeric_frame(df, ["a", "b"])
    assert arr.shape == (2, 2)
    assert _narrow_numeric_frame(df, ["a", "c"]) is None  # non-numeric column -> None


def test_narrow_numeric_frame_polars():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    arr = _narrow_numeric_frame(df, ["a", "b"])
    assert arr.shape == (3, 2)


def test_evaluate_tta_quality_is_public():
    import mlframe.training as training_mod

    assert hasattr(training_mod, "evaluate_tta_quality")
    assert "evaluate_tta_quality" in training_mod.__all__


def _reg_frame(seed=17, n=1400):
    import polars as pl

    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n).astype(np.float32)
    x1 = rng.normal(size=n).astype(np.float32)
    x2 = rng.normal(size=n).astype(np.float32)
    y = (2 * x0 - x1 + 0.5 * x2 + 0.3 * rng.normal(size=n)).astype(np.float32)
    return pl.DataFrame({"f0": x0, "f1": x1, "f2": x2, "target": y})


def test_e2e_uncertainty_eval_stamped_in_suite(tmp_path):
    pytest.importorskip("xgboost")
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import (
        PreprocessingBackendConfig,
        OutputConfig,
        TrainingBehaviorConfig,
        BaselineDiagnosticsConfig,
        DummyBaselinesConfig,
        ReportingConfig,
    )
    from mlframe.training._preprocessing_configs import TrainingSplitConfig
    from .shared import SimpleFeaturesAndTargetsExtractor

    _, metadata = train_mlframe_models_suite(
        df=_reg_frame(),
        target_name="ue",
        model_name="ue_run",
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True),
        mlframe_models=["xgb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        pipeline_config=PreprocessingBackendConfig(prefer_polarsds=False, categorical_encoding=None, scaler_name=None, imputer_strategy=None),
        split_config=TrainingSplitConfig(test_size=0.25, val_size=0.1),
        behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False, uncertainty_eval=True),
        hyperparams_config={"iterations": 40, "xgb_kwargs": {"device": "cpu"}},
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(honest_estimator_diagnostics=False),
        enable_target_distribution_analyzer=False,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        verbose=0,
    )
    import orjson as _json

    assert "uncertainty_eval" in metadata, "not stamped; dbg=" + _json.dumps(metadata.get("_ue_dbg", {}), default=str).decode()
    rep = next(iter(metadata["uncertainty_eval"].values()))
    assert "test" in rep and "spread_error_corr" in rep["test"] and "tta_rmse_gain" in rep["test"]
