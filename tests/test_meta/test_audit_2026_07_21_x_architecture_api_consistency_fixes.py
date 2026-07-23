"""Regression tests for audits/full_audit_2026-07-21/x_architecture_api_consistency.md findings F1-F5, F7-F9.

F6 (3 files exceeding 1000 LOC: training/core/_phase_composite_post_xt_ensemble/__init__.py,
training/composite/transforms/nonlinear.py, training/_trainer_train_and_evaluate.py) is DELIBERATELY
DEFERRED, not fixed: a correct split needs the full AST-audit-every-sibling-for-unresolved-names
procedure CLAUDE.md's own "Monolith split" section mandates, applied to 3 large files in the core
training pipeline (one of them a subpackage facade __init__.py) -- disproportionate regression risk
for a P2 architecture finding relative to the other 8 (all P2, lower blast radius) findings in this
report. Concrete next action: split each in its own dedicated session with the full AST-audit gate,
starting with whichever of the 3 next crosses 900 LOC again after any unrelated edit touches it.

PR1-PR4 are proposals (a longer 900-999 LOC watchlist, a test-coverage mirror-check, an import-linter
tooling suggestion, and a docs note) with no reported bug -- assessed, no fix required.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

# ---------------------------------------------------------------------------
# F1: calibration/post.py no longer eagerly imports mlframe.training.evaluation /
# mlframe.models.ensembling at module scope (TargetTypes remains eager -- used as a function default)
# ---------------------------------------------------------------------------


def test_f1_calibration_post_defers_training_evaluation_import():
    """F1 calibration post defers training evaluation import."""
    import mlframe.calibration.post as post

    src = inspect.getsource(post)
    assert "from mlframe.training.evaluation import report_model_perf" not in src.split("def ")[0]
    assert "from mlframe.training.evaluation import report_model_perf" in src  # still present, just deferred


def test_f1_calibration_post_defers_ensembling_import():
    """F1 calibration post defers ensembling import."""
    import mlframe.calibration.post as post

    src = inspect.getsource(post)
    module_header = src.split("class _CalibTestOverlapError")[0]
    assert "from mlframe.models.ensembling import ensemble_probabilistic_predictions" not in module_header
    assert "from mlframe.models.ensembling import ensemble_probabilistic_predictions" in src


def test_f1_calibration_post_still_importable_standalone():
    """F1 calibration post still importable standalone."""
    importlib.reload(importlib.import_module("mlframe.calibration.post"))


# ---------------------------------------------------------------------------
# F2/F3: feature_engineering modules no longer eagerly import mlframe.training.feature_handling
# ---------------------------------------------------------------------------


def test_f2_categorical_powerset_concat_defers_training_import():
    """F2 categorical powerset concat defers training import."""
    import mlframe.feature_engineering.categorical_powerset_concat as mod

    src = inspect.getsource(mod)
    module_header = src.split("def categorical_powerset_concat")[0]
    assert "mlframe.training" not in module_header


def test_f3_two_step_target_encode_defers_training_import():
    """F3 two step target encode defers training import."""
    import mlframe.feature_engineering.two_step_target_encode as mod

    src = inspect.getsource(mod)
    module_header = src.split("def two_step_recency_weighted_target_encode")[0]
    assert "mlframe.training" not in module_header


def test_f2_f3_feature_engineering_modules_still_functionally_correct():
    """F2 f3 feature engineering modules still functionally correct."""
    import numpy as np
    import pandas as pd

    from mlframe.feature_engineering.categorical_powerset_concat import categorical_powerset_concat

    df = pd.DataFrame({"a": ["x", "y", "x", "y"], "b": ["p", "q", "p", "q"]})
    y = np.array([0, 1, 0, 1])
    out = categorical_powerset_concat(df, ["a", "b"], prune_against_target=(y, -1.0))
    assert out.shape[0] == 4


# ---------------------------------------------------------------------------
# F4: preprocessing/outlier_detector_zoo.py no longer eagerly imports mlframe.models.ensembling
# ---------------------------------------------------------------------------


def test_f4_outlier_detector_zoo_defers_models_import():
    """F4 outlier detector zoo defers models import."""
    import mlframe.preprocessing.outlier_detector_zoo as mod

    src = inspect.getsource(mod)
    module_header = src.split("def make_outlier_detector")[0]
    assert "mlframe.models" not in module_header
    assert "mlframe.models.ensembling.selection import rank_average_blend" in src  # still present, deferred


# ---------------------------------------------------------------------------
# F5: 8 previously-uncurated __init__.py facades now compute an explicit __all__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    [
        "mlframe.core",
        "mlframe.data",
        "mlframe.estimators",
        "mlframe.inference",
        "mlframe.models",
        "mlframe.preprocessing",
        "mlframe.utils",
        "mlframe.training.composite.transforms",
    ],
)
def test_f5_package_has_curated_all(module_name):
    """F5 package has curated all."""
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "__all__"), f"F5 REGRESSION: {module_name} must define an explicit __all__"
    assert len(mod.__all__) > 0
    assert len(mod.__all__) == len(set(mod.__all__)), f"F5 REGRESSION: {module_name}.__all__ has duplicates"


# ---------------------------------------------------------------------------
# F7: 3 duplicate seed-derivation helpers consolidated onto mlframe.core.helpers.derive_seed
# ---------------------------------------------------------------------------


def test_f7_honest_diagnostics_derive_seed_matches_canonical():
    """F7 honest diagnostics derive seed matches canonical."""
    from mlframe.core.helpers import derive_seed
    from mlframe.training.honest_diagnostics import _derive_seed

    assert _derive_seed(42, "foo") == derive_seed(42, "foo")


def test_f7_composite_ensemble_derive_seeds_matches_canonical():
    """F7 composite ensemble derive seeds matches canonical."""
    from mlframe.core.helpers import derive_seed
    from mlframe.training.composite.ensemble import derive_seeds

    out = derive_seeds(42, ["a", "b", "c"])
    assert out == {c: derive_seed(42, c) for c in ["a", "b", "c"]}


def test_f7_per_target_seed_documented_as_intentionally_different():
    """The 3rd helper (_dummy_baseline_compute._per_target_seed) is deliberately NOT consolidated --
    it has a load-bearing backward-compat contract with already-persisted BaselineReports. Confirm the
    documenting comment landed instead of a silent, undocumented 3rd construction."""
    from mlframe.training.baselines import _dummy_baseline_compute as mod

    src = inspect.getsource(mod._per_target_seed)
    assert "Deliberately NOT the shared" in src


def test_f7_derive_seed_is_deterministic_and_in_range():
    """F7 derive seed is deterministic and in range."""
    from mlframe.core.helpers import derive_seed

    v1 = derive_seed(7, "key")
    v2 = derive_seed(7, "key")
    assert v1 == v2
    assert 0 <= v1 < 2**31 - 1
    assert derive_seed(7, "key_a") != derive_seed(7, "key_b")


# ---------------------------------------------------------------------------
# F8: compute_numaggs_parallel's df param now correctly typed Optional[pd.DataFrame]
# ---------------------------------------------------------------------------


def test_f8_compute_numaggs_parallel_df_param_is_optional():
    """F8 compute numaggs parallel df param is optional."""
    from mlframe.feature_engineering.numerical import compute_numaggs_parallel

    sig = inspect.signature(compute_numaggs_parallel)
    df_param = sig.parameters["df"]
    assert df_param.default is None
    # typing.Optional[X] renders as typing.Union[X, None] / "X | None" depending on Python version;
    # check the annotation string contains "Optional" or "None" alongside DataFrame, not a bare DataFrame.
    ann = str(df_param.annotation)
    assert "None" in ann or "Optional" in ann


# ---------------------------------------------------------------------------
# F9: metrics/__init__.py documents that direct mlframe.metrics.core imports are an intentionally
# supported, bit-identical alternative to the facade -- not a violation needing ~20 call-site rewrites
# ---------------------------------------------------------------------------


def test_f9_metrics_facade_documents_core_direct_import_convention():
    """F9 metrics facade documents core direct import convention."""
    import mlframe.metrics as metrics

    doc = metrics.__doc__ or ""
    assert "Import convention" in doc
    assert "bit-identical" in doc


def test_f9_metrics_core_and_facade_export_the_same_object():
    """F9 metrics core and facade export the same object."""
    import mlframe.metrics as metrics
    import mlframe.metrics.core as core

    assert metrics.fast_roc_auc is core.fast_roc_auc
