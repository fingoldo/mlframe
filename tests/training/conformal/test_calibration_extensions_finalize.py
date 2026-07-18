"""Unit + biz_value + e2e tests wiring opt-in mlframe.calibration extensions into finalize_suite.

Covers three previously-isolated, direct-import-only utilities now reachable via config:
    - ``threshold_optimizer.optimize_decision_threshold`` -> ``TrainingBehaviorConfig.auto_optimize_threshold``
    - ``confidence_shrinkage.compute_oof_confidence``/``apply_confidence_shrinkage`` ->
      ``RegressionCalibrationConfig.apply_confidence_shrinkage``
    - ``isotonic_risk.isotonic_overfit_risk`` -> ``TrainingBehaviorConfig.check_isotonic_overfit_risk``

Each new flag defaulted to False/None through 2026-07-11; the corresponding metadata key was absent (and,
for confidence shrinkage, predictions untouched) when omitted. As of 2026-07-12 all three flags default to
True/sensible values (see ``DEFAULTS_CHANGELOG.md``) -- the default-behavior regression tests below assert
the NEW default-on metadata keys are present, and a separate explicit-opt-out test per flag still covers the
old bit-identical no-op path for callers who explicitly disable it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.training.core._phase_finalize import (
    _apply_confidence_shrinkage_to_regression,
    _isotonic_overfit_risk_check,
    _optimize_decision_threshold_on_calib_slice,
)


def _binary_ctx(cp, ct, behavior_config=None):
    """Binary ctx."""
    e = SimpleNamespace(
        model=SimpleNamespace(),
        calib_probs=cp,
        calib_target=ct,
        model_name="clf",
    )
    return SimpleNamespace(
        models={"BINARY_CLASSIFICATION": {"y": [e]}},
        metadata={},
        verbose=0,
        configs=None,
        behavior_config=behavior_config,
    ), e


# ---------------------------------------------------------------------------
# isotonic_overfit_risk
# ---------------------------------------------------------------------------


def test_isotonic_overfit_risk_check_flags_noisy_fit():
    """A calib slice with one label flip per point (pure noise) trips the overfit-risk flag."""
    # segment_ratio_threshold=1e-9 guarantees any real isotonic fit (>=1 breakpoint) trips ``flagged`` --
    # deterministic way to exercise the flagged/remediation-eligible branch, since pure-noise labels can
    # legitimately PAVA-pool into very few segments (segment_ratio is not reliably "high" just because the
    # labels are random).
    rng = np.random.default_rng(0)
    n = 400
    p = np.sort(rng.uniform(0, 1, size=n))
    y = rng.integers(0, 2, size=n).astype(np.float64)  # uncorrelated with p -> isotonic tracks noise
    cp = np.column_stack([1.0 - p, p])
    ctx, _ = _binary_ctx(cp, y, behavior_config=SimpleNamespace(check_isotonic_overfit_risk=True, isotonic_risk_kwargs={"segment_ratio_threshold": 1e-9}))

    _isotonic_overfit_risk_check(ctx)

    assert "isotonic_risk_report" in ctx.metadata
    rep = ctx.metadata["isotonic_risk_report"]["BINARY_CLASSIFICATION/y/clf"]
    assert rep["flagged"] is True
    assert rep["n_samples"] == n
    assert "isotonic_fit" not in rep  # fitted objects dropped before stamping


def test_isotonic_overfit_risk_check_default_off_noop():
    """Isotonic overfit risk check default off noop."""
    rng = np.random.default_rng(1)
    n = 200
    p = np.sort(rng.uniform(0, 1, size=n))
    y = (rng.uniform(size=n) < p).astype(np.float64)
    cp = np.column_stack([1.0 - p, p])
    ctx, _ = _binary_ctx(cp, y, behavior_config=SimpleNamespace(check_isotonic_overfit_risk=False, isotonic_risk_kwargs=None))

    _isotonic_overfit_risk_check(ctx)

    assert "isotonic_risk_report" not in ctx.metadata


def test_isotonic_overfit_risk_check_no_config_noop():
    """No behavior_config anywhere on ctx -> clean no-op (never raises)."""
    rng = np.random.default_rng(2)
    n = 50
    p = np.sort(rng.uniform(0, 1, size=n))
    y = (rng.uniform(size=n) < p).astype(np.float64)
    cp = np.column_stack([1.0 - p, p])
    ctx, _ = _binary_ctx(cp, y, behavior_config=None)

    _isotonic_overfit_risk_check(ctx)

    assert "isotonic_risk_report" not in ctx.metadata


# ---------------------------------------------------------------------------
# threshold_optimizer
# ---------------------------------------------------------------------------


def _separable_probs(rng, n=800):
    """Separable probs."""
    y = rng.integers(0, 2, size=n)
    p = np.clip(y * 0.6 + rng.normal(0, 0.15, size=n) + 0.2, 0.0, 1.0)
    return p, y.astype(np.float64)


def test_optimize_decision_threshold_stamps_metadata_sensible_value():
    """Optimize decision threshold stamps metadata sensible value."""
    rng = np.random.default_rng(3)
    p, y = _separable_probs(rng)
    cp = np.column_stack([1.0 - p, p])
    ctx, _ = _binary_ctx(cp, y, behavior_config=SimpleNamespace(auto_optimize_threshold=True, threshold_optimizer_kwargs=None))

    _optimize_decision_threshold_on_calib_slice(ctx)

    assert "decision_threshold" in ctx.metadata
    rep = ctx.metadata["decision_threshold"]["BINARY_CLASSIFICATION/y/clf"]
    assert 0.0 <= rep["best_threshold"] <= 1.0
    assert 0.0 <= rep["best_score"] <= 1.0
    assert "thresholds" not in rep and "scores" not in rep  # full sweep dropped, compact summary kept


def test_optimize_decision_threshold_groups_and_cv_kwargs_thread_through():
    """Optimize decision threshold groups and cv kwargs thread through."""
    rng = np.random.default_rng(4)
    p, y = _separable_probs(rng, n=1200)
    groups = rng.integers(0, 3, size=1200)
    cp = np.column_stack([1.0 - p, p])
    ctx, _ = _binary_ctx(
        cp,
        y,
        behavior_config=SimpleNamespace(
            auto_optimize_threshold=True,
            threshold_optimizer_kwargs={"groups": groups, "min_group_size": 10, "cv": 3},
        ),
    )

    _optimize_decision_threshold_on_calib_slice(ctx)

    rep = ctx.metadata["decision_threshold"]["BINARY_CLASSIFICATION/y/clf"]
    assert "group_thresholds" in rep
    assert set(rep["group_thresholds"].keys()) == {0, 1, 2}
    assert "cv_report" in rep


def test_optimize_decision_threshold_default_off_noop():
    """Optimize decision threshold default off noop."""
    rng = np.random.default_rng(5)
    p, y = _separable_probs(rng)
    cp = np.column_stack([1.0 - p, p])
    ctx, _ = _binary_ctx(cp, y, behavior_config=SimpleNamespace(auto_optimize_threshold=False, threshold_optimizer_kwargs=None))

    _optimize_decision_threshold_on_calib_slice(ctx)

    assert "decision_threshold" not in ctx.metadata


# ---------------------------------------------------------------------------
# confidence_shrinkage
# ---------------------------------------------------------------------------


def _regression_entry(rng, n=400, discriminative=True):
    """Build a regression-style entry whose OOF/target look like a binary-flag output (Santander-style)."""
    y = rng.integers(0, 2, size=n).astype(np.float64)
    if discriminative:
        oof = np.clip(y * 0.7 + rng.normal(0, 0.1, size=n) + 0.15, 0.0, 1.0)
    else:
        oof = rng.uniform(0.0, 1.0, size=n)  # uncorrelated with y -> low confidence
    test_preds = np.clip(rng.uniform(0.3, 0.7, size=50), 0.0, 1.0)
    return SimpleNamespace(
        model=SimpleNamespace(),
        oof_preds=oof,
        train_target=y,
        test_preds=test_preds,
        model_name="reg",
    ), test_preds


def test_confidence_shrinkage_shrinks_weak_target_toward_neutral():
    """Confidence shrinkage shrinks weak target toward neutral."""
    rng = np.random.default_rng(6)
    weak_entry, weak_test_preds = _regression_entry(rng, discriminative=False)
    strong_entry, _strong_test_preds = _regression_entry(rng, discriminative=True)
    ctx = SimpleNamespace(
        models={"REGRESSION": {"weak": [weak_entry], "strong": [strong_entry]}},
        metadata={},
        verbose=0,
        configs=None,
        regression_calibration_config=SimpleNamespace(apply_confidence_shrinkage=True, confidence_shrinkage_kwargs={"neutral_value": 0.5}),
    )

    _apply_confidence_shrinkage_to_regression(ctx)

    assert "confidence_shrinkage" in ctx.metadata
    # The weak (uncorrelated OOF) target's confidence must be materially lower than the strong target's.
    weak_conf = ctx.metadata["confidence_shrinkage"]["REGRESSION/weak/reg"]["confidence"]
    strong_conf = ctx.metadata["confidence_shrinkage"]["REGRESSION/strong/reg"]["confidence"]
    assert weak_conf < strong_conf
    # The weak entry's shipped test_preds moved toward neutral (0.5) relative to its raw predictions.
    assert not np.array_equal(weak_entry.test_preds, weak_test_preds)
    assert np.mean(np.abs(weak_entry.test_preds - 0.5)) < np.mean(np.abs(weak_test_preds - 0.5))


def test_confidence_shrinkage_default_off_noop_bit_identical():
    """Confidence shrinkage default off noop bit identical."""
    rng = np.random.default_rng(7)
    entry, test_preds = _regression_entry(rng, discriminative=False)
    ctx = SimpleNamespace(
        models={"REGRESSION": {"t": [entry]}},
        metadata={},
        verbose=0,
        configs=None,
        regression_calibration_config=SimpleNamespace(apply_confidence_shrinkage=False, confidence_shrinkage_kwargs=None),
    )

    _apply_confidence_shrinkage_to_regression(ctx)

    assert "confidence_shrinkage" not in ctx.metadata
    assert np.array_equal(entry.test_preds, test_preds)


# ---------------------------------------------------------------------------
# e2e: reachable via train_mlframe_models_suite, and default run is bit-identical
# ---------------------------------------------------------------------------


def _classification_frame(seed=11, n=1500):
    """Classification frame."""
    import polars as pl

    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n).astype(np.float32)
    x1 = rng.normal(size=n).astype(np.float32)
    logit = 1.5 * x0 - 0.8 * x1
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(np.int64)
    return pl.DataFrame({"f0": x0, "f1": x1, "target": y})


def _run_classification_suite(tmp_path, behavior_config, seed=11):
    """Run classification suite."""
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import (
        PreprocessingBackendConfig,
        OutputConfig,
        BaselineDiagnosticsConfig,
        DummyBaselinesConfig,
        ReportingConfig,
    )
    from mlframe.training._preprocessing_configs import TrainingSplitConfig
    from tests.training.shared import SimpleFeaturesAndTargetsExtractor

    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    return train_mlframe_models_suite(
        df=_classification_frame(seed=seed),
        target_name="calib_ext_e2e",
        model_name="calib_ext_e2e",
        features_and_targets_extractor=fte,
        mlframe_models=["xgb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        pipeline_config=PreprocessingBackendConfig(
            prefer_polarsds=False,
            categorical_encoding=None,
            scaler_name=None,
            imputer_strategy=None,
        ),
        split_config=TrainingSplitConfig(test_size=0.25, val_size=0.1, calib_size=0.2, random_seed=seed),
        behavior_config=behavior_config,
        hyperparams_config={"iterations": 40, "xgb_kwargs": {"device": "cpu"}},
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(honest_estimator_diagnostics=False),
        conformal_config=None,
        enable_target_distribution_analyzer=False,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        verbose=0,
    )


def test_e2e_auto_optimize_threshold_reachable_via_config(tmp_path):
    """E2e auto optimize threshold reachable via config."""
    pytest.importorskip("xgboost")
    from mlframe.training.configs import TrainingBehaviorConfig

    _, metadata = _run_classification_suite(
        tmp_path,
        behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False, auto_optimize_threshold=True),
    )
    assert "decision_threshold" in metadata, "auto_optimize_threshold=True did not stamp metadata['decision_threshold']"
    rep = next(iter(metadata["decision_threshold"].values()))
    assert 0.0 <= rep["best_threshold"] <= 1.0
    assert 0.0 <= rep["best_score"] <= 1.0


def test_e2e_default_config_populates_new_calibration_extension_keys(tmp_path):
    """Default suite run (all new flags left at their post-2026-07-12 True default) now populates BOTH new
    metadata keys without the caller opting in explicitly."""
    pytest.importorskip("xgboost")
    from mlframe.training.configs import TrainingBehaviorConfig

    _, metadata = _run_classification_suite(tmp_path, behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False))
    assert "decision_threshold" in metadata, "auto_optimize_threshold default True did not stamp metadata['decision_threshold']"
    assert "isotonic_risk_report" in metadata, "check_isotonic_overfit_risk default True did not stamp metadata['isotonic_risk_report']"


def test_e2e_explicit_opt_out_omits_new_calibration_extension_keys(tmp_path):
    """Explicitly disabling both flags still reproduces the pre-2026-07-12 bit-identical no-op path."""
    pytest.importorskip("xgboost")
    from mlframe.training.configs import TrainingBehaviorConfig

    _, metadata = _run_classification_suite(
        tmp_path,
        behavior_config=TrainingBehaviorConfig(
            prefer_gpu_configs=False,
            auto_optimize_threshold=False,
            check_isotonic_overfit_risk=False,
        ),
    )
    assert "decision_threshold" not in metadata
    assert "isotonic_risk_report" not in metadata
    assert "confidence_shrinkage" not in metadata
