"""Regression: suite-owned datetime decomposition must (a) be replayed at predict time so derived cols are byte-identical to training, and (b) skip any datetime sources the FTE already decomposed (otherwise duplicate / overwriting cols land in the frame).

Pre-fix the suite called ``create_date_features`` on every detected datetime col without persisting the resolved methods list; the predict path had no equivalent step, so a model trained on a ``feature_types_config``-supplied datetime col crashed (or silently got wrong derived cols) at inference. Separately, when the FTE used ``delete_original_cols=False`` to keep its ``ts_field`` for downstream consumers, the suite would re-decompose it and emit duplicate / overwriting ``year_ts`` / ``month_ts`` cols on top of FTE's.
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
    FeatureTypesConfig,
    OutputConfig,
    ReportingConfig,
)
from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor


def _make_frame_with_two_dt_cols(n: int = 3_000, seed: int = 0):
    """Build a frame with two datetime columns: ``ts`` (FTE-known) and ``extra_dt`` (FTE-unknown; only the suite should decompose it)."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2022-01-01")
    ts = pd.to_datetime(base + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n), unit="m"))
    extra = pd.to_datetime(base + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n), unit="m"))
    df = pl.DataFrame(
        {
            "x0": rng.normal(size=n).astype("float32"),
            "x1": rng.normal(size=n).astype("float32"),
            "ts": pl.Series(ts.values),
            "extra_dt": pl.Series(extra.values),
            "y": (1.5 * rng.normal(size=n) + rng.normal(0, 0.3, n)).astype("float32"),
        }
    )
    return df


def _run_suite(df, fte, feature_types_config=None):
    """Runs the mlframe training suite on df with the given feature-type-extraction toggle."""
    return train_mlframe_models_suite(
        df=df,
        target_name="y",
        model_name="dt_replay",
        features_and_targets_extractor=fte,
        mlframe_models=["lgb"],
        verbose=0,
        output_config=OutputConfig(data_dir="", models_dir="", save_charts=False),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        feature_types_config=feature_types_config,
        reporting_config=ReportingConfig(plot_outputs="matplotlib[png]", plot_inline_display=False),
    )


def test_suite_skips_redecomposition_of_fte_emitted_datetime_columns():
    """When the FTE expands ``ts`` via ``create_date_features`` (its standard datetime handling), the suite MUST NOT re-decompose ``ts`` -- the FTE already produced derived cols and the suite's second pass would emit duplicates / overwrite them."""
    pytest.importorskip("lightgbm")
    df = _make_frame_with_two_dt_cols()
    fte = SimpleFeaturesAndTargetsExtractor(
        regression_targets=["y"],
        ts_field="ts",
        datetime_features={"year": np.int32, "month": np.int8},
        columns_to_drop={"extra_dt"},  # don't let extra_dt break the test
    )
    models, metadata = _run_suite(df, fte)
    assert models, "training returned empty models dict"

    # FTE-emitted columns recorded.
    fte_emitted = metadata.get("ftextractor_emitted_columns") or {}
    assert "ts" in fte_emitted, f"FTE did not record ts decomposition: {fte_emitted}"
    # ``datetime_features={"year": ..., "month": ...}`` is the USER-CONFIGURED
    # explicit set; the FTE additionally emits cyclical-encoding companions
    # (hour_sin / hour_cos / day_sin / day_cos) by default. The test's real
    # concern is that the user-configured methods MUST be present and the
    # FTE owns the ``ts`` decomposition (so the suite-side check below
    # confirms it didn't double-decompose). Use subset, not strict equality.
    _emitted_for_ts = set(fte_emitted["ts"])
    assert {"ts_year", "ts_month"}.issubset(_emitted_for_ts), f"FTE missing user-configured year/month methods on ts; emitted={_emitted_for_ts}"

    # Suite-side ``datetime_methods`` must NOT contain ``ts`` (FTE owned it; suite was supposed to skip).
    suite_dt = metadata.get("datetime_methods") or {}
    assert "ts" not in suite_dt, f"Suite re-decomposed ts despite FTE already emitting derived cols; metadata.datetime_methods={suite_dt}"


def test_suite_decomposes_and_persists_methods_for_fte_unknown_datetime_column():
    """When ``feature_types_config`` adds a datetime column the FTE does NOT know about, the suite owns the decomposition and MUST persist the resolved method list so predict can replay deterministically."""
    pytest.importorskip("lightgbm")
    df = _make_frame_with_two_dt_cols()
    fte = SimpleFeaturesAndTargetsExtractor(
        regression_targets=["y"],
        ts_field="ts",
        datetime_features={"year": np.int32, "month": np.int8},
    )
    ftc = FeatureTypesConfig(datetime_methods=("year", "month", "weekday"))
    models, metadata = _run_suite(df, fte, feature_types_config=ftc)
    assert models, "training returned empty models dict"

    suite_dt = metadata.get("datetime_methods") or {}
    assert "extra_dt" in suite_dt, f"Suite did not persist methods for FTE-unknown extra_dt; metadata.datetime_methods={suite_dt}"
    persisted = suite_dt["extra_dt"]
    assert set(persisted.keys()) == {"year", "month", "weekday"}, persisted
    # Year uses int32 dtype (per ``_wide_int_methods``), others use int8.
    assert persisted["year"] == "int32"
    assert persisted["month"] == "int8"
    assert persisted["weekday"] == "int8"

    # FTE-handled ``ts`` is still NOT in the suite map.
    assert "ts" not in suite_dt, suite_dt


def test_predict_replays_suite_owned_datetime_decomposition_byte_identical():
    """End-to-end: train with an FTE-unknown datetime col, predict on the same raw frame -- predictions must be valid (model received correctly-decomposed derived cols at inference)."""
    pytest.importorskip("lightgbm")
    df = _make_frame_with_two_dt_cols()
    fte = SimpleFeaturesAndTargetsExtractor(
        regression_targets=["y"],
        ts_field="ts",
        datetime_features={"year": np.int32, "month": np.int8},
    )
    ftc = FeatureTypesConfig(datetime_methods=("year", "month", "day"))

    models, metadata = _run_suite(df, fte, feature_types_config=ftc)
    assert models, "training returned empty models dict"
    assert "extra_dt" in (metadata.get("datetime_methods") or {})

    results = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=False,
        verbose=0,
    )
    assert results["models_used"], (
        "predict_from_models produced no successful models -- the datetime replay likely failed and the pipeline saw a raw datetime col it could not handle."
    )
    assert results["ensemble_predictions"] is not None
    preds = results["ensemble_predictions"]
    assert len(preds) == len(df)
    assert np.isfinite(preds).all(), "predictions contain non-finite values"
