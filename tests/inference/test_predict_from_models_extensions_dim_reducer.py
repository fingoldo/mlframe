"""iter#189 regression: predict_from_models must not drop raw input cols when training used
a column-changing preprocessing extension (dim_reducer / one-hot / polynomial / KBins / etc.).

Pre-fix: ``metadata["columns"]`` was overwritten to the post-extensions output column list
(e.g. ``[truncatedsvd0..9]``). The predict-time ``_validate_input_columns_against_metadata``
filtered the raw user input against that post-pipeline list, dropping every raw user column
as "extra" and producing a (N, 0) frame. The downstream extensions transform then crashed
with ``Found array with 0 sample(s) (shape=(0, K)) while a minimum of 1 is required by SimpleImputer``.

Post-fix: the raw-input columns are captured in ``metadata["input_columns"]`` at the START of
``_phase_fit_pipeline``, and ``_validate_input_columns_against_metadata`` consults that key
first. The pipeline + extensions then reshape the validated raw frame into the model-input
shape.

Surfaced by the diverse-harness fuzz profile (binary x lgb,linear,ridge x cat_enc=onehot x
dim_reducer=TruncatedSVD x 1M rows, seed 2026051901).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest


def _make_small_frame(n_rows: int = 600, seed: int = 123) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "x0": rng.normal(size=n_rows).astype(np.float32),
            "x1": rng.normal(size=n_rows).astype(np.float32),
            "cat_low": rng.choice(["A", "B", "C"], size=n_rows),
            "y": rng.integers(0, 2, size=n_rows).astype(np.int64),
        }
    )


def test_predict_from_models_with_dim_reducer_extension_does_not_drop_raw_input_cols():
    """End-to-end: train with PreprocessingExtensionsConfig(dim_reducer=TruncatedSVD), then
    call predict_from_models on the original raw frame. Pre-fix this crashed at the
    extensions-transform step because metadata-driven input validation dropped every raw
    user column as "extra"."""
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.core.predict import predict_from_models
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor
    from mlframe.training.configs import (
        BaselineDiagnosticsConfig,
        CompositeTargetDiscoveryConfig,
        DummyBaselinesConfig,
        FeatureSelectionConfig,
        OutlierDetectionConfig,
        OutputConfig,
        PreprocessingBackendConfig,
        PreprocessingExtensionsConfig,
        ReportingConfig,
    )

    df = _make_small_frame()

    fte_kwargs = dict(
        classification_targets=["y"],
        classification_exact_values={"y": 1},
    )
    fte_train = SimpleFeaturesAndTargetsExtractor(**fte_kwargs)

    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="y",
        model_name="test_iter189",
        features_and_targets_extractor=fte_train,
        mlframe_models=["lgb"],  # one fast model is enough to repro the metadata bug
        feature_selection_config=FeatureSelectionConfig(),
        outlier_detection_config=OutlierDetectionConfig(),
        pipeline_config=PreprocessingBackendConfig(
            categorical_encoding="onehot",
            scaler_name="standard",
        ),
        preprocessing_extensions=PreprocessingExtensionsConfig(
            dim_reducer="TruncatedSVD",
            dim_n_components=3,
        ),
        verbose=0,
        output_config=OutputConfig(data_dir="", models_dir=""),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(),
    )

    # Sanity-check the metadata captures the RAW input schema (the contract
    # downstream consumers like _validate_input_columns_against_metadata rely on).
    assert "input_columns" in metadata, (
        "metadata must carry input_columns (raw pre-pipeline schema) so predict-time validation can compare against the user-supplied frame"
    )
    raw_cols = set(metadata["input_columns"])
    # cat_low / x0..x1 are the user-supplied columns; y is the target which the
    # FTE strips before pipeline runs. The exact set after FTE may exclude y.
    expected_raw_cols = {"x0", "x1", "cat_low"}
    assert expected_raw_cols.issubset(raw_cols), f"input_columns missing raw user columns: {expected_raw_cols - raw_cols} (have: {raw_cols})"

    # The actual regression: predict must not crash with "Found array with 0 sample(s)".
    fte_predict = SimpleFeaturesAndTargetsExtractor(**fte_kwargs)
    results = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte_predict,
        return_probabilities=True,
        verbose=0,
    )

    # Every trained model should have produced predictions on the input frame.
    assert results["predictions"], "predict_from_models returned empty predictions dict"
    for model_name, preds in results["predictions"].items():
        assert preds is not None, f"model {model_name} returned None predictions"
        assert len(preds) == len(df), f"model {model_name} returned {len(preds)} predictions for {len(df)} input rows"
