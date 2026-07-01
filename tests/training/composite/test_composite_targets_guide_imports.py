"""Import-only sensor for ``docs/composite_targets_guide.md``.

Pins every public symbol the guide references so the guide cannot silently
rot when the composite-targets public surface is renamed / moved. Import-only
by design: it asserts the documented names resolve and that the dotted modules
the guide imports from smoke-import cleanly.
"""
from __future__ import annotations

import importlib

import pytest


GUIDE_SYMBOLS = [
    # estimators / discovery / ensemble
    "CompositeTargetEstimator", "CompositeClassificationEstimator",
    "CompositeGLMEstimator", "CompositeQuantileEstimator",
    "CompositeMultiOutputEstimator", "make_per_column_specs",
    "make_composite_regressor", "CompositeTargetTransformer",
    "CompositeTargetDiscovery", "CompositeTargetDiscoveryConfig",
    "CompositeCrossTargetEnsemble", "CompositeSpec", "CompositeProvenance",
    "DiscoveryCache",
    # transforms + registry
    "Transform", "get_transform", "list_transforms",
    # uncertainty / ensemble helpers
    "conformal_quantile", "predict_quantile_ensemble",
    # discovery helpers referenced in the guide
    "detect_time_column_candidates", "sort_df_by_time_column",
    # reporting
    "report_to_markdown",
    # errors
    "DomainViolationError", "UnknownTransformError",
]

# Transforms the guide names explicitly in the catalogue section / snippets.
GUIDE_TRANSFORMS = [
    "diff", "linear_residual", "linear_residual_multi", "linear_residual_grouped",
    "ratio", "logratio", "cbrt_y", "log_y", "signed_power_y",
    "target_encoding_residual", "theilsen_residual",
    "ewma_residual", "frac_diff", "rolling_quantile_ratio",
]

# Dotted modules the guide imports from directly.
GUIDE_MODULES = [
    "mlframe.training.composite",
    "mlframe.training.composite.autoconfig",
    "mlframe.training.composite.diagnostics",
    "mlframe.training.composite.spec",
]

# Methods / classmethods the guide calls on CompositeTargetEstimator.
GUIDE_ESTIMATOR_ATTRS = [
    "fit", "predict", "predict_quantile", "from_fitted_inner",
    "calibrate_conformal", "predict_interval",
    "calibrate_conformal_cqr", "predict_interval_cqr",
    "calibrate_conformal_mondrian", "predict_interval_mondrian",
]

# autoconfig entry point referenced by the discovery section.
GUIDE_AUTOCONFIG_FUNCS = ["suggest_discovery_config"]

# Diagnostics plot helpers referenced in the guide.
GUIDE_DIAGNOSTICS_FUNCS = [
    "plot_target_distribution", "plot_qq", "plot_linear_fit",
    "plot_predictions_vs_actual", "plot_reliability_diagram",
    "plot_interval_coverage", "plot_interval_width_vs_x",
    "plot_mi_gain_with_ci", "plot_alpha_stability",
]


@pytest.mark.parametrize("name", GUIDE_SYMBOLS)
def test_guide_symbol_importable_from_composite(name):
    mod = importlib.import_module("mlframe.training.composite")
    assert hasattr(mod, name), f"composite guide references missing symbol {name!r}"


@pytest.mark.parametrize("dotted", GUIDE_MODULES)
def test_guide_dotted_module_smoke_imports(dotted):
    assert importlib.import_module(dotted) is not None


@pytest.mark.parametrize("tname", GUIDE_TRANSFORMS)
def test_guide_transform_registered(tname):
    from mlframe.training.composite import get_transform, list_transforms

    assert tname in list_transforms(), f"guide names unregistered transform {tname!r}"
    assert get_transform(tname) is not None


@pytest.mark.parametrize("attr", GUIDE_ESTIMATOR_ATTRS)
def test_guide_estimator_method_present(attr):
    from mlframe.training.composite import CompositeTargetEstimator

    assert hasattr(CompositeTargetEstimator, attr), (
        f"guide calls CompositeTargetEstimator.{attr} but it is absent"
    )


def test_common_usage_block_in_class_docstring():
    from mlframe.training.composite import CompositeTargetEstimator

    doc = CompositeTargetEstimator.__doc__ or ""
    assert "Common Usage" in doc, "Common Usage examples block missing from docstring"


@pytest.mark.parametrize("fn", GUIDE_AUTOCONFIG_FUNCS)
def test_guide_autoconfig_func_present(fn):
    mod = importlib.import_module("mlframe.training.composite.autoconfig")
    assert hasattr(mod, fn)


@pytest.mark.parametrize("fn", GUIDE_DIAGNOSTICS_FUNCS)
def test_guide_diagnostics_func_present(fn):
    mod = importlib.import_module("mlframe.training.composite.diagnostics")
    assert hasattr(mod, fn)
