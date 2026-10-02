"""``train_mlframe_models_suite`` wiring of standalone ``mlframe.evaluation`` diagnostics.

``output_config.run_diagnostics`` (default: the 5 cheap registered diagnostics; ``adversarial_fold_selection`` is opt-in) reaches the
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
    """Builds a binary-classification frame with a logistic target driven by feature f0."""
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(0, 1, n)
    f1 = rng.uniform(0, 1, n)
    df = pd.DataFrame({"f0": f0, "f1": f1})
    logit = 3 * f0 - 1.5
    df["target"] = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def _run(tmp_path, name, run_diagnostics="default", n=300, **suite_kw):
    """One hgb suite run on the synthetic binary frame; ``run_diagnostics="default"`` leaves the ``OutputConfig`` default in force."""
    skip_if_dependency_missing("hgb")
    out_kw = {} if run_diagnostics == "default" else {"run_diagnostics": run_diagnostics}
    return train_mlframe_models_suite(
        df=_make_frame(n, seed=1),
        target_name="target",
        model_name=name,
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False),
        mlframe_models=["hgb"],
        hyperparams_config=get_cpu_config("hgb", 20),
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models", save_charts=False, **out_kw),
        verbose=0,
        **suite_kw,
    )


@pytest.fixture(scope="module")
def explicit_run(tmp_path_factory):
    """One suite run requesting two real diagnostics, the opt-in adversarial fold and a name that does not exist."""
    return _run(
        tmp_path_factory.mktemp("explicit"),
        "diag_explicit",
        run_diagnostics=["cv_informativeness", "compare_cv_schemes", "not_a_real_diagnostic", "adversarial_fold_selection"],
    )


@pytest.fixture(scope="module")
def default_run(tmp_path_factory):
    """One default-config suite run, with the adversarial fold builder spied on; returns ``(models, metadata, builder_calls)``."""
    import mlframe.evaluation.adversarial_fold_selection as afs

    calls = []
    orig = afs.build_test_like_validation_fold
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(afs, "build_test_like_validation_fold", lambda *a, **k: calls.append(1) or orig(*a, **k))
        models, metadata = _run(tmp_path_factory.mktemp("default"), "diag_default")
    return models, metadata, calls


def test_run_diagnostics_reaches_evaluation_functions_through_suite(explicit_run):
    """Requesting cv_informativeness + compare_cv_schemes lands non-error reports under metadata["diagnostics"]."""
    models, metadata = explicit_run
    assert TargetTypes.BINARY_CLASSIFICATION in models
    assert "diagnostics" in metadata, "metadata['diagnostics'] not stamped despite run_diagnostics being set"
    diag = metadata["diagnostics"]
    for name in ("cv_informativeness", "compare_cv_schemes"):
        assert name in diag, f"{name!r} missing from metadata['diagnostics']; got keys={list(diag)}"
        report = diag[name]
        assert isinstance(report, dict), f"{name}: expected a dict report, got {type(report)}"
        assert "error" not in report, f"{name}: adapter reported an error: {report}"


def test_unknown_diagnostic_name_reports_error_without_crashing(explicit_run):
    """An unrecognized run_diagnostics name reports an error entry under metadata['diagnostics'] instead of raising."""
    _models, metadata = explicit_run
    assert "error" in metadata["diagnostics"]["not_a_real_diagnostic"]


def test_adversarial_fold_selection_opt_in_stores_compact_int32(explicit_run):
    """Explicitly requesting it still works, and ``val_idx`` is a compact int32 array, not a Python list."""
    _, metadata = explicit_run
    res = metadata["diagnostics"]["adversarial_fold_selection"]
    assert "error" not in res, res
    assert isinstance(res["val_idx"], np.ndarray) and res["val_idx"].dtype == np.int32
    assert len(res["val_idx"]) == res["n_selected"]


def test_run_diagnostics_default_on_populates_five_cheap(default_run):
    """Default ``OutputConfig()`` (``run_diagnostics`` omitted) runs the 5 cheap registered diagnostics;
    ``adversarial_fold_selection`` is opt-in. ``metadata["diagnostics"]`` must carry each default key."""
    _, metadata, _calls = default_run
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


def test_default_suite_never_builds_adversarial_fold(default_run):
    """A default-config suite run must not fit the adversarial classifier nor store its key."""
    _, metadata, calls = default_run
    assert "adversarial_fold_selection" not in metadata["diagnostics"]
    assert calls == []


def test_run_diagnostics_explicit_none_opts_out(tmp_path):
    """Explicitly passing ``run_diagnostics=None`` opts back out to the pre-2026-07-12 no-op behavior."""
    _, metadata = _run(tmp_path, "diag_optout", run_diagnostics=None)
    assert "diagnostics" not in metadata


def test_default_run_diagnostics_excludes_adversarial_fold_selection():
    """The default list is the five cheap diagnostics; the registry still knows the opt-in sixth."""
    from mlframe.training.core._diagnostics_registry import DIAGNOSTICS_REGISTRY

    default = OutputConfig().run_diagnostics
    assert "adversarial_fold_selection" not in default
    assert len(default) == 5
    assert "adversarial_fold_selection" in DIAGNOSTICS_REGISTRY
