"""Regression test for iter#53:

predict_from_models converted Polars input to pandas via
get_pandas_view_of_polars_df BEFORE calling pipeline.transform(). When
the saved pipeline was a polars-ds Pipeline (prefer_polarsds=True at
fit time, the default for Polars-backed training), its .transform()
called ``df.lazy()`` on the now-pandas DataFrame, raising
``AttributeError: 'DataFrame' object has no attribute 'lazy'``.

The fix defers the polars->pandas conversion until AFTER
pipeline.transform, so a polars-ds pipeline sees a Polars frame and a
sklearn pipeline sees a pandas frame -- matching the format each was
fitted on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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


def _build_polars_frame(n: int = 3_000, seed: int = 0):
    """Minimal numeric + low-card int Polars frame for fast suite training."""
    rng = np.random.default_rng(seed)
    df = pl.DataFrame(
        {
            "x0": rng.normal(size=n).astype("float32"),
            "x1": rng.normal(size=n).astype("float32"),
            "c_low": rng.integers(0, 5, n).astype("int32"),
            "y": (1.5 * rng.normal(size=n) + rng.normal(0, 0.3, n)).astype("float32"),
        }
    )
    return df


def _run_suite(df: pl.DataFrame, models_list):
    """Returns ``train_mlframe_models_suite(df=df, target_name='y', model_name='prof', features_and_targ...`` (after 1 setup step)."""
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


def test_predict_from_models_polars_input_polarsds_pipeline():
    """Predict path must run end-to-end on Polars input without
    AttributeError('lazy'). The trained polars-ds pipeline must receive
    a Polars frame, NOT the pre-converted pandas view."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("polars_ds")
    df = _build_polars_frame()
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])

    models, metadata = _run_suite(df, ["lgb"])
    assert models, "training returned empty models dict"

    # Smoke test: predict_from_models must complete without exception on
    # the same Polars frame the suite was trained on.
    results = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=False,
        verbose=0,
    )
    assert results["models_used"], (
        "predict_from_models returned no successful models -- the polars-ds "
        "pipeline likely crashed because we passed it a pandas frame "
        "(.lazy() AttributeError, the iter#53 bug)"
    )
    assert results["ensemble_predictions"] is not None


def test_predict_from_models_pandas_input_sklearn_pipeline():
    """Sanity: pandas input still works (sklearn pipeline path)."""
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    n = 3_000
    df = pd.DataFrame(
        {
            "x0": rng.normal(size=n).astype("float32"),
            "x1": rng.normal(size=n).astype("float32"),
            "c_low": rng.integers(0, 5, n).astype("int32"),
            "y": (1.5 * rng.normal(size=n) + rng.normal(0, 0.3, n)).astype("float32"),
        }
    )
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])

    models, metadata = _run_suite(df, ["lgb"])
    assert models

    results = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=False,
        verbose=0,
    )
    assert results["models_used"]
