"""Regression test for iter#55:

predict_from_models on Polars input with categorical columns failed for
LGB with::

    train and valid dataset categorical_feature do not match.

The polars-ds pipeline's output Polars frame was converted to pandas
via ``get_pandas_view_of_polars_df`` (to_arrow + to_pandas). Polars
String columns survived as pandas object/string - NOT Categorical.
LightGBM sklearn auto-detects categorical_feature from input dtype at
predict time and compared the result against the fit-time spec; the
mismatch raised before the booster's scorer ran.

The fix coerces ``metadata['cat_features']`` columns to pandas
``category`` dtype right after the Polars-to-pandas conversion. CB and
XGB accept the cast too (CB happily takes Categorical via Pool; XGB
requires non-object dtypes).
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.core.predict import predict_from_models
from mlframe.training.configs import (
    BaselineDiagnosticsConfig,
    CompositeTargetDiscoveryConfig,
    DummyBaselinesConfig,
    OutputConfig,
    ReportingConfig,
)
from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor


def _build_polars_frame_with_cats(n: int = 3_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = pl.DataFrame({
        "x0": rng.normal(size=n).astype("float32"),
        "x1": rng.normal(size=n).astype("float32"),
        "cat_low": np.array(["A", "B", "C", "D", "E"], dtype=object)[rng.integers(0, 5, n)],
        "cat_mid": np.array([f"M{j:02d}" for j in range(20)], dtype=object)[rng.integers(0, 20, n)],
        "y": rng.integers(0, 3, n).astype("int32"),  # 3-class multiclass
    })
    return df


def _run_suite(df, models_list, classification: bool = True):
    if classification:
        fte = SimpleFeaturesAndTargetsExtractor(classification_targets=["y"])
    else:
        fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    return train_mlframe_models_suite(
        df=df,
        target_name="y",
        model_name="prof",
        features_and_targets_extractor=fte,
        mlframe_models=list(models_list),
        verbose=0,
        output_config=OutputConfig(data_dir="", models_dir="", save_charts=False),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(plot_outputs="matplotlib[png]", plot_inline_display=False),
    )


def test_predict_from_models_lgb_polars_cats():
    """LGB predict on Polars input with cat columns must complete without
    the categorical_feature-mismatch error."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("polars_ds")
    df = _build_polars_frame_with_cats()
    fte = SimpleFeaturesAndTargetsExtractor(classification_targets=["y"])

    models, metadata = _run_suite(df, ["lgb"])
    assert models, "training returned empty models dict"

    results = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=True,
        verbose=0,
    )
    assert results["models_used"], (
        "predict_from_models returned no successful models -- LGB likely "
        "tripped the categorical_feature mismatch (iter#55 bug)"
    )
