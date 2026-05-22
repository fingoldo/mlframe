"""Wave-2 predict-path parity Fix 2: replay ``metadata['extensions_pipeline']`` at predict time.

Pre-fix ``predict_mlframe_models_suite`` / ``predict_from_models`` read only ``metadata['pipeline']`` and ignored
``metadata['extensions_pipeline']`` (PySR, TF-IDF, polynomial, scaler, KBins, RBF, PCA). Models trained with
``preprocessing_extensions`` saw RAW columns at predict, producing predictions inconsistent with the trained model's
expected scale.
"""
from __future__ import annotations

from unittest.mock import patch

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
    PreprocessingExtensionsConfig,
    ReportingConfig,
)
from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor


def _build_frame(n: int = 2_000, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "x0": rng.normal(size=n).astype("float32"),
        "x1": rng.normal(size=n).astype("float32"),
        "y": (0.5 * rng.normal(size=n) + rng.normal(0, 0.3, n)).astype("float32"),
    })


def _run_with_ext(df, ext_cfg):
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    return train_mlframe_models_suite(
        df=df,
        target_name="y",
        model_name="ext_replay",
        features_and_targets_extractor=fte,
        mlframe_models=["lgb"],
        verbose=0,
        output_config=OutputConfig(data_dir="", models_dir="", save_charts=False),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(plot_outputs="matplotlib[png]", plot_inline_display=False),
        preprocessing_extensions=ext_cfg,
    )


def test_predict_invokes_extensions_pipeline_when_present():
    """When ``metadata['extensions_pipeline']`` is populated by training (scaler, polynomial, PCA, etc.), the
    predict path must call the replay helper. Pre-fix the helper did not exist and the metadata key was ignored;
    the assertion would fail because the spy never fires."""
    pytest.importorskip("lightgbm")
    df = _build_frame()
    ext_cfg = PreprocessingExtensionsConfig(scaler="StandardScaler", verbose_logging=False)
    models, metadata = _run_with_ext(df, ext_cfg)
    assert metadata.get("extensions_pipeline") is not None, (
        "training did not persist extensions_pipeline; the rest of the test depends on its presence."
    )

    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    import mlframe.training.core.predict as predict_mod

    invocations = {"n": 0, "input_was_frame": False}
    orig_helper = predict_mod._apply_extensions_pipeline

    def _spy(df_in, ep, verbose=0):
        invocations["n"] += 1
        invocations["input_was_frame"] = hasattr(df_in, "columns")
        return orig_helper(df_in, ep, verbose=verbose)

    with patch.object(predict_mod, "_apply_extensions_pipeline", side_effect=_spy):
        results = predict_from_models(
            df=df,
            models=models,
            metadata=metadata,
            features_and_targets_extractor=fte,
            return_probabilities=False,
            verbose=0,
        )
    assert results["models_used"], "no models predicted"
    assert invocations["n"] >= 1, "extensions_pipeline replay was NEVER called even though metadata carried the key"
    assert invocations["input_was_frame"], "replay helper received a non-DataFrame; should have received the post-main-pipeline frame"


def test_predict_no_extensions_no_replay():
    """Sanity: when metadata has no extensions_pipeline the replay helper is NOT invoked."""
    pytest.importorskip("lightgbm")
    df = _build_frame()
    # Train without preprocessing_extensions to leave metadata['extensions_pipeline'] absent / None.
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="y",
        model_name="no_ext",
        features_and_targets_extractor=fte,
        mlframe_models=["lgb"],
        verbose=0,
        output_config=OutputConfig(data_dir="", models_dir="", save_charts=False),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(plot_outputs="matplotlib[png]", plot_inline_display=False),
    )
    assert not metadata.get("extensions_pipeline"), (
        "metadata unexpectedly carries an extensions_pipeline even though preprocessing_extensions was None."
    )

    import mlframe.training.core.predict as predict_mod
    invocations = {"n": 0}
    orig_helper = predict_mod._apply_extensions_pipeline

    def _spy(df_in, ep, verbose=0):
        invocations["n"] += 1
        return orig_helper(df_in, ep, verbose=verbose)

    with patch.object(predict_mod, "_apply_extensions_pipeline", side_effect=_spy):
        results = predict_from_models(
            df=df,
            models=models,
            metadata=metadata,
            features_and_targets_extractor=fte,
            return_probabilities=False,
            verbose=0,
        )
    assert results["models_used"]
    assert invocations["n"] == 0, (
        f"replay helper was called {invocations['n']} time(s) despite no extensions_pipeline being saved at training; "
        "the predict path should guard the call on the metadata key."
    )
