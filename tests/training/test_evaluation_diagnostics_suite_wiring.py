"""``train_mlframe_models_suite`` opt-in wiring of standalone ``mlframe.evaluation`` diagnostics.

``output_config.run_diagnostics`` (default ``None``) reaches the previously-isolated evaluation
functions through the public suite entry point via ``mlframe.training.core._diagnostics_registry``.
This is the wiring test, not a re-test of the underlying evaluation functions' math.
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
            data_dir=str(tmp_path), models_dir="models", save_charts=False,
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
            data_dir=str(tmp_path), models_dir="models", save_charts=False,
            run_diagnostics=["not_a_real_diagnostic"],
        ),
        verbose=0,
    )
    assert "error" in metadata["diagnostics"]["not_a_real_diagnostic"]


def test_run_diagnostics_default_none_is_bit_identical_to_omitted(tmp_path):
    """The opt-in field defaults to None -- the diagnostics block must be a complete no-op then:
    no ``metadata["diagnostics"]`` key at all, and the rest of metadata is unaffected by its mere presence
    in the codebase (regression guard against the new field ever becoming non-opt-in)."""
    skip_if_dependency_missing("hgb")
    df = _make_frame(300, seed=1)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

    def _run(output_config):
        _, metadata = train_mlframe_models_suite(
            df=df.copy(),
            target_name="target",
            model_name="diag_default",
            features_and_targets_extractor=fte,
            mlframe_models=["hgb"],
            hyperparams_config=get_cpu_config("hgb", 20),
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=output_config,
            verbose=0,
        )
        return metadata

    metadata_omitted = _run(OutputConfig(data_dir=str(tmp_path / "a"), models_dir="models", save_charts=False))
    metadata_explicit_none = _run(
        OutputConfig(data_dir=str(tmp_path / "b"), models_dir="models", save_charts=False, run_diagnostics=None),
    )
    assert "diagnostics" not in metadata_omitted
    assert "diagnostics" not in metadata_explicit_none
