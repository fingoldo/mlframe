"""iter#190 regression: a bare LocalOutlierFactor (no imputer wrapper) fed a NaN-bearing train
frame must NOT take down the entire training suite — the framework should log + skip outlier
detection and continue training on the unfiltered frames.

Pre-fix: ``_setup_helpers.apply_outlier_detection_once`` called ``outlier_detector.fit(...)``
directly without wrapping in try/except. ``LocalOutlierFactor.fit`` raises
``ValueError: Input X contains NaN. LocalOutlierFactor does not accept missing values...``
propagating up through ``train_mlframe_models_suite``, leaving the entire run in an unusable
state regardless of which downstream models would have tolerated the NaN.

Post-fix: the fit/predict pair is wrapped so an OD raise turns into a logged error +
graceful skip; the rest of the suite proceeds on the unfiltered frame.

Surfaced by fuzz iter#190 (regression x lgb x outlier=lof x NaN-bearing 1M frame).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _make_nan_frame(n_rows: int = 600, seed: int = 191) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n_rows).astype(np.float32)
    x1 = rng.normal(size=n_rows).astype(np.float32)
    # Inject NaN into ~10% of x0 and x1 so a bare LOF (no imputer wrapper) crashes at fit.
    nan_mask_0 = rng.random(size=n_rows) < 0.10
    nan_mask_1 = rng.random(size=n_rows) < 0.10
    x0[nan_mask_0] = np.nan
    x1[nan_mask_1] = np.nan
    return pl.DataFrame({
        "x0": x0,
        "x1": x1,
        "y": rng.normal(size=n_rows).astype(np.float32),
    })


def test_bare_lof_with_nan_input_does_not_crash_suite():
    """When the caller passes a NaN-tolerant-but-not-imputer-wrapped detector (bare LOF) and the
    input frame has NaN, the suite should log the OD failure and continue training. Pre-fix this
    raised ``ValueError: Input X contains NaN`` out of ``train_mlframe_models_suite``."""
    from sklearn.neighbors import LocalOutlierFactor

    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor
    from mlframe.training.configs import (
        BaselineDiagnosticsConfig, CompositeTargetDiscoveryConfig,
        DummyBaselinesConfig, FeatureSelectionConfig, OutlierDetectionConfig,
        OutputConfig, PreprocessingBackendConfig, ReportingConfig,
    )

    df = _make_nan_frame()

    bare_lof = LocalOutlierFactor(novelty=True, n_neighbors=20)

    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])

    # Training must complete (returning models + metadata) rather than re-raising the
    # ``ValueError: Input X contains NaN`` out of the OD step.
    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="y",
        model_name="test_iter190",
        features_and_targets_extractor=fte,
        mlframe_models=["lgb"],
        feature_selection_config=FeatureSelectionConfig(),
        outlier_detection_config=OutlierDetectionConfig(detector=bare_lof),
        pipeline_config=PreprocessingBackendConfig(),
        preprocessing_extensions=None,
        verbose=0,
        output_config=OutputConfig(data_dir="", models_dir=""),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(),
    )

    # Suite proceeded; at least one model was trained on the (unfiltered) frame.
    assert models, "train_mlframe_models_suite returned empty models dict; suite did not recover from OD failure"
    assert metadata is not None
