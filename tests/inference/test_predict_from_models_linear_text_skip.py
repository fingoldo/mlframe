"""Regression test for iter#64:

predict_from_models on a linear (or ridge / sgd) model failed with::

    Error predicting with model regression_y_Pipeline: The feature
    names should match those that were passed during fit.
    Feature names unseen at fit time: - text_col

The linear-family models wrap their fitted estimator in a sklearn
Pipeline (imputer + scaler) exposed as ``model_obj.pre_pipeline``.
predict_from_models routed input through ``pre_pipeline.transform()``
BEFORE applying the per-model feature_names_in_ subset, so text/
embedding columns reached the pre_pipeline's own input-feature checker
and tripped sklearn's "Feature names unseen at fit time" validation.

The fix reorders the two steps: subset by per-model expected features
FIRST (preferring pre_pipeline.feature_names_in_ when available so we
see the raw input the pre_pipeline expects), then run
pre_pipeline.transform on the cleaned frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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


def _build_frame_with_text(n: int = 3_000, seed: int = 0):
    """Builds seeded synthetic test data; returns ``df``."""
    rng = np.random.default_rng(seed)
    _vocab = np.array("alpha beta gamma delta epsilon zeta".split(), dtype=object)
    _idx = rng.integers(0, len(_vocab), (n, 4))
    df = pd.DataFrame(
        {
            "x0": rng.normal(size=n).astype("float32"),
            "x1": rng.normal(size=n).astype("float32"),
            "text_col": np.array([" ".join(_vocab[r]) for r in _idx], dtype=object),
        }
    )
    df["y"] = (1.5 * df["x0"] - 1.0 * df["x1"] + rng.normal(0, 0.3, n)).astype("float32")
    return df


def _run_suite(df, models_list):
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


def test_predict_from_models_linear_with_text_col():
    """Linear regression predict must not crash on text_col input."""
    df = _build_frame_with_text()
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])

    models, metadata = _run_suite(df, ["linear"])
    assert models, "training returned empty models dict"

    results = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=False,
        verbose=0,
    )
    assert results["models_used"], (
        "predict_from_models returned no successful models -- linear "
        "pre_pipeline.transform likely tripped the 'Feature names unseen "
        "at fit time' error (the iter#64 bug)"
    )
    assert results["ensemble_predictions"] is not None
