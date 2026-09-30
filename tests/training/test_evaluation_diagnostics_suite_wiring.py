"""``train_mlframe_models_suite`` wiring of standalone ``mlframe.evaluation`` diagnostics.

``output_config.run_diagnostics`` (default: the 5 cheap registered diagnostics; ``adversarial_fold_selection`` is opt-in) reaches the
previously-isolated evaluation functions through the public suite entry point via
``mlframe.training.core._diagnostics_registry``. This is the wiring test, not a re-test of the underlying
evaluation functions' math.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.configs import OutputConfig, TargetTypes
from mlframe.training.core import train_mlframe_models_suite

from .shared import SimpleFeaturesAndTargetsExtractor, get_cpu_config, skip_if_dependency_missing


def _make_frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Builds a binary-classification frame with a logistic target driven by feature f0."""
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
    """An unrecognized run_diagnostics name reports an error entry under metadata['diagnostics'] instead of raising."""
    skip_if_dependency_missing("hgb")
    df = _make_frame(300)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    _models, metadata = train_mlframe_models_suite(
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


def test_run_diagnostics_default_on_populates_five_cheap(tmp_path):
    """Default ``OutputConfig()`` (``run_diagnostics`` omitted) runs the 5 cheap registered diagnostics;
    ``adversarial_fold_selection`` is opt-in. ``metadata["diagnostics"]`` must carry each default key."""
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


def test_default_run_diagnostics_excludes_adversarial_fold_selection():
    """The default list is the five cheap diagnostics; the registry still knows the opt-in sixth."""
    from mlframe.training.core._diagnostics_registry import DIAGNOSTICS_REGISTRY

    default = OutputConfig().run_diagnostics
    assert "adversarial_fold_selection" not in default
    assert len(default) == 5
    assert "adversarial_fold_selection" in DIAGNOSTICS_REGISTRY


def test_default_suite_never_builds_adversarial_fold(tmp_path, monkeypatch):
    """A default-config suite run must not fit the adversarial classifier nor store its key."""
    skip_if_dependency_missing("hgb")
    import mlframe.evaluation.adversarial_fold_selection as afs

    calls = []
    orig = afs.build_test_like_validation_fold
    monkeypatch.setattr(afs, "build_test_like_validation_fold", lambda *a, **k: calls.append(1) or orig(*a, **k))
    df = _make_frame(300, seed=1)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    _, metadata = train_mlframe_models_suite(
        df=df.copy(), target_name="target", model_name="diag_adv_off", features_and_targets_extractor=fte,
        mlframe_models=["hgb"], hyperparams_config=get_cpu_config("hgb", 20), use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=str(tmp_path / "c"), models_dir="models", save_charts=False), verbose=0,
    )
    assert "adversarial_fold_selection" not in metadata["diagnostics"]
    assert calls == []


def test_adversarial_fold_selection_opt_in_stores_compact_int32(tmp_path):
    """Explicitly requesting it still works, and ``val_idx`` is a compact int32 array, not a Python list."""
    skip_if_dependency_missing("hgb")
    df = _make_frame(300, seed=1)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    _, metadata = train_mlframe_models_suite(
        df=df.copy(), target_name="target", model_name="diag_adv_on", features_and_targets_extractor=fte,
        mlframe_models=["hgb"], hyperparams_config=get_cpu_config("hgb", 20), use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(
            data_dir=str(tmp_path / "d"), models_dir="models", save_charts=False, run_diagnostics=["adversarial_fold_selection"]
        ),
        verbose=0,
    )
    res = metadata["diagnostics"]["adversarial_fold_selection"]
    assert "error" not in res, res
    assert isinstance(res["val_idx"], np.ndarray) and res["val_idx"].dtype == np.int32
    assert len(res["val_idx"]) == res["n_selected"]
