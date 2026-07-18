"""Regression test: replay row-wise extension columns (row_summary_*/row_extreme_*) at predict time.

Root cause: ``row_wise_summary_stats`` / ``row_wise_top_k_extreme_columns`` are wired into
``PreprocessingExtensionsConfig`` DEFAULT ON (``_phase_helpers_fit_pipeline.py`` auto-constructs a
default config whenever the caller passes none at all -- see
``tests/training/test_biz_val_row_wise_extensions_default_on.py``), so every model's fit-time
feature set includes them unless explicitly disabled. Unlike the sklearn-bridge
``extensions_pipeline`` (scaler/PCA/etc, already replayed at predict -- see
``test_predict_extensions_pipeline_replay.py``), these two steps are STATELESS per-row functions
with no fitted object, so nothing replayed them at predict time at all: every deployed model with
the default config raised "feature names ... unseen at fit time" on its very first real
``predict_from_models``/``predict_mlframe_models_suite`` call on raw data.

Fixed by stamping the enabled/params knobs onto ``metadata["row_wise_extensions_config"]`` at fit
time (``_phase_helpers_fit_pipeline.py``) and recomputing the columns directly from the predict-time
frame's own numeric columns via ``_apply_row_wise_extensions`` (``_predict_pre_pipeline.py``), wired
into both ``predict_mlframe_models_suite`` and ``predict_from_models``.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import polars as pl

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


def _build_frame(n: int = 300, seed: int = 0) -> pl.DataFrame:
    """Small regression frame; default PreprocessingExtensionsConfig requires no explicit opt-in."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n).astype("float32")
    x1 = rng.normal(size=n).astype("float32")
    y = (0.5 * x0 - 0.3 * x1 + rng.normal(0, 0.3, n)).astype("float32")
    return pl.DataFrame({"x0": x0, "x1": x1, "y": y})


def _run_default(df):
    """Train with NO preprocessing_extensions passed -- the suite auto-defaults it ON."""
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    return train_mlframe_models_suite(
        df=df,
        target_name="y",
        model_name="row_wise_replay",
        features_and_targets_extractor=fte,
        mlframe_models=["linear"],
        verbose=0,
        output_config=OutputConfig(data_dir="", models_dir="", save_charts=False),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(plot_outputs="matplotlib[png]", plot_inline_display=False),
    )


def test_row_wise_extensions_config_persisted_by_default():
    """Training with no explicit preprocessing_extensions must still stamp metadata['row_wise_extensions_config']."""
    df = _build_frame()
    _models, metadata = _run_default(df)
    cfg = metadata.get("row_wise_extensions_config")
    assert cfg is not None, "row_wise_extensions_config was not persisted despite row-wise extensions being default ON"
    assert cfg["summary_stats_enabled"] is True
    assert cfg["extreme_columns_enabled"] is True


def test_predict_from_models_survives_default_row_wise_extensions():
    """End-to-end: predict_from_models on RAW data must not crash with "feature names unseen at fit time".

    Pre-fix this raised inside sklearn's ``_check_feature_names`` the moment the fitted linear
    model's pre_pipeline/estimator (whose fit-time X included row_summary_*/row_extreme_* columns)
    saw the raw 2-column predict frame.
    """
    df = _build_frame()
    models, metadata = _run_default(df)
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    results = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=False,
        verbose=0,
    )
    assert results["models_used"], "no models predicted"
    for _preds in results["predictions"].values():
        assert np.all(np.isfinite(np.asarray(_preds))), "predictions must be finite"


def test_predict_invokes_row_wise_replay_when_present():
    """The replay helper must actually be invoked (not just present) when metadata carries the config."""
    df = _build_frame()
    models, metadata = _run_default(df)
    assert metadata.get("row_wise_extensions_config") is not None

    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    import mlframe.training.core.predict as predict_mod

    invocations = {"n": 0, "input_was_frame": False}
    orig_helper = predict_mod._apply_row_wise_extensions

    def _spy(df_in, cfg, verbose=0):
        """Spy wrapper recording invocation + input type before delegating to the real helper."""
        invocations["n"] += 1
        invocations["input_was_frame"] = hasattr(df_in, "columns")
        return orig_helper(df_in, cfg, verbose=verbose)

    with patch.object(predict_mod, "_apply_row_wise_extensions", side_effect=_spy):
        results = predict_from_models(
            df=df,
            models=models,
            metadata=metadata,
            features_and_targets_extractor=fte,
            return_probabilities=False,
            verbose=0,
        )
    assert results["models_used"]
    assert invocations["n"] >= 1, "row-wise extensions replay was NEVER called even though metadata carried the config"
    assert invocations["input_was_frame"]


def test_predict_no_row_wise_config_no_replay():
    """Sanity: when row-wise extensions were explicitly disabled, the replay helper must not fire."""
    from mlframe.training.configs import PreprocessingExtensionsConfig

    df = _build_frame()
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="y",
        model_name="row_wise_off",
        features_and_targets_extractor=fte,
        mlframe_models=["linear"],
        verbose=0,
        output_config=OutputConfig(data_dir="", models_dir="", save_charts=False),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(plot_outputs="matplotlib[png]", plot_inline_display=False),
        preprocessing_extensions=PreprocessingExtensionsConfig(
            row_wise_summary_stats_enabled=False,
            row_wise_extreme_columns_enabled=False,
        ),
    )
    assert not metadata.get("row_wise_extensions_config"), "row_wise_extensions_config unexpectedly persisted despite both flags disabled"

    import mlframe.training.core.predict as predict_mod

    invocations = {"n": 0}
    orig_helper = predict_mod._apply_row_wise_extensions

    def _spy(df_in, cfg, verbose=0):
        """Spy wrapper recording invocation count before delegating to the real helper."""
        invocations["n"] += 1
        return orig_helper(df_in, cfg, verbose=verbose)

    with patch.object(predict_mod, "_apply_row_wise_extensions", side_effect=_spy):
        results = predict_from_models(
            df=df,
            models=models,
            metadata=metadata,
            features_and_targets_extractor=fte,
            return_probabilities=False,
            verbose=0,
        )
    assert results["models_used"]
    assert invocations["n"] == 0, f"replay helper was called {invocations['n']} time(s) despite row-wise extensions being disabled at training"
