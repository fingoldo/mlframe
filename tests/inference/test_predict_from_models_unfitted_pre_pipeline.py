"""Regression test for iter#79:

predict_from_models on a tree-model (lgb / hgb / cb / xgb) suite trained
on a Polars frame with the polars-ds main pipeline crashed with::

    Error predicting with model regression_y_Pipeline: This Pipeline
    instance is not fitted yet. Call 'fit' with appropriate arguments
    before using this estimator.

Tree-model strategies attach an empty sklearn Pipeline as
``model_obj.pre_pipeline`` for sklearn-compat introspection, but never
call ``.fit()`` on it - the main polars-ds pipeline handles all
preprocessing. predict_from_models naively called
``pre_pipeline.transform(df)`` which raised NotFittedError.

The fix probes ``check_is_fitted(pre_pipeline)`` before calling
.transform. When the pipeline is unfitted (the tree-model placeholder
case), the call is skipped and we trust the main pipeline's output.
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


def test_predict_from_models_lgb_hgb_polars_unfitted_pre_pipeline():
    """LGB+HGB suite on Polars input must predict without
    NotFittedError on the placeholder pre_pipeline."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("polars_ds")
    rng = np.random.default_rng(0)
    n = 3_000
    df = pl.DataFrame(
        {
            "x0": rng.normal(size=n).astype("float32"),
            "x1": rng.normal(size=n).astype("float32"),
            "c_low": rng.integers(0, 5, n).astype("int32"),
            "y": (1.5 * rng.normal(size=n) + rng.normal(0, 0.3, n)).astype("float32"),
        }
    )
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])

    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="y",
        model_name="prof",
        features_and_targets_extractor=fte,
        mlframe_models=["lgb", "hgb"],
        verbose=0,
        output_config=OutputConfig(data_dir="", models_dir="", save_charts=False),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(plot_outputs="matplotlib[png]", plot_inline_display=False),
    )
    assert models

    results = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=False,
        verbose=0,
    )
    assert len(results["models_used"]) >= 2, (
        f"Expected lgb+hgb to predict; got {results['models_used']}. If both crashed, the unfitted-pre_pipeline guard is missing."
    )
