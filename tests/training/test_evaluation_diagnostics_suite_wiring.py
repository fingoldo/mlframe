"""``train_mlframe_models_suite`` wiring of standalone ``mlframe.evaluation`` diagnostics.

``output_config.run_diagnostics`` (default: all 6 registered diagnostics, since 2026-07-12) reaches the
previously-isolated evaluation functions through the public suite entry point via
``mlframe.training.core._diagnostics_registry``. This is the wiring test, not a re-test of the underlying
evaluation functions' math.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import OutputConfig, TargetTypes
from mlframe.training.core import train_mlframe_models_suite

from .shared import SimpleFeaturesAndTargetsExtractor, get_cpu_config, skip_if_dependency_missing


def _make_frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(0, 1, n)
    f1 = rng.uniform(0, 1, n)
    df = pd.DataFrame({"f0": f0, "f1": f1})
    logit = 3 * f0 - 1.5
    df["target"] = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def test_run_diagnostics_reaches_evaluation_functions_through_suite(tmp_path):
    """Requesting cv_informativeness + compare_cv_schemes lands non-error reports under metadata["diagnostics"]."""
    skip_if_dependency_missing("hgb")
    df = _make_frame(500)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="target",
        model_name="diag_wire",
        features_and_targets_extractor=fte,
        mlframe_models=["hgb"],
        hyperparams_config=get_cpu_config("hgb", 20),
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(
            data_dir=str(tmp_path),
            models_dir="models",
            save_charts=False,
            run_diagnostics=["cv_informativeness", "compare_cv_schemes"],
        ),
        verbose=0,
    )
    assert TargetTypes.BINARY_CLASSIFICATION in models
    assert "diagnostics" in metadata, "metadata['diagnostics'] not stamped despite run_diagnostics being set"
    diag = metadata["diagnostics"]
    for name in ("cv_informativeness", "compare_cv_schemes"):
        assert name in diag, f"{name!r} missing from metadata['diagnostics']; got keys={list(diag)}"
        report = diag[name]
        assert isinstance(report, dict), f"{name}: expected a dict report, got {type(report)}"
        assert "error" not in report, f"{name}: adapter reported an error: {report}"


def test_unknown_diagnostic_name_reports_error_without_crashing(tmp_path):
    skip_if_dependency_missing("hgb")
    df = _make_frame(300)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="target",
        model_name="diag_unknown",
        features_and_targets_extractor=fte,
        mlframe_models=["hgb"],
        hyperparams_config=get_cpu_config("hgb", 20),
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(
            data_dir=str(tmp_path),
            models_dir="models",
            save_charts=False,
            run_diagnostics=["not_a_real_diagnostic"],
        ),
        verbose=0,
    )
    assert "error" in metadata["diagnostics"]["not_a_real_diagnostic"]


def test_run_diagnostics_default_on_populates_all_six(tmp_path):
    """Default ``OutputConfig()`` (``run_diagnostics`` omitted) now runs all 6 registered diagnostics --
    default-on wiring landed 2026-07-12. ``metadata["diagnostics"]`` must carry every registry key with a
    non-error report."""
    skip_if_dependency_missing("hgb")
    df = _make_frame(300, seed=1)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

    _, metadata = train_mlframe_models_suite(
        df=df.copy(),
        target_name="target",
        model_name="diag_default",
        features_and_targets_extractor=fte,
        mlframe_models=["hgb"],
        hyperparams_config=get_cpu_config("hgb", 20),
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=str(tmp_path / "a"), models_dir="models", save_charts=False),
        verbose=0,
    )
    assert "diagnostics" in metadata, "metadata['diagnostics'] missing despite run_diagnostics defaulting on"
    diag = metadata["diagnostics"]
    for name in (
        "cv_informativeness",
        "compare_cv_schemes",
        "group_leakage",
        "constant_group_leak",
        "adversarial_fold_selection",
        "subpopulation_drift",
    ):
        assert name in diag, f"{name!r} missing from metadata['diagnostics']; got keys={list(diag)}"


def test_run_diagnostics_explicit_none_opts_out(tmp_path):
    """Explicitly passing ``run_diagnostics=None`` opts back out to the pre-2026-07-12 no-op behavior."""
    skip_if_dependency_missing("hgb")
    df = _make_frame(300, seed=1)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

    _, metadata = train_mlframe_models_suite(
        df=df.copy(),
        target_name="target",
        model_name="diag_optout",
        features_and_targets_extractor=fte,
        mlframe_models=["hgb"],
        hyperparams_config=get_cpu_config("hgb", 20),
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=str(tmp_path / "b"), models_dir="models", save_charts=False, run_diagnostics=None),
        verbose=0,
    )
    assert "diagnostics" not in metadata
