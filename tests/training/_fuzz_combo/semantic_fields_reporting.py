"""Semantic generators for ``OutputConfig`` / ``ReportingConfig`` (see :mod:`semantic_fields` for the mechanism)."""

from __future__ import annotations

import os
from typing import Any, Dict, Sequence

import numpy as np

from .semantic_callables import binned_ece_integral_error, brier_integral_error
from .semantic_fields import _choice, _register, _subset

# ---------------------------------------------------------------------------
# OutputConfig / ReportingConfig
# ---------------------------------------------------------------------------


@_register("OutputConfig", "diagnostics_kwargs")
def _gen_diagnostics_kwargs(rng, ctx, model) -> Dict[str, Any]:
    """Per-diagnostic kwargs drawn from each registered adapter's callee signature; ``subgroup_col`` names a real categorical column."""
    if rng.random() < 0.3:
        return {}
    pool: Dict[str, Dict[str, Any]] = {
        "constant_group_leak": {"min_group_size": int(_choice(rng, (5, 20))), "family_alpha": float(_choice(rng, (0.01, 0.05)))},
        "cv_informativeness": {"check_trend": bool(rng.random() < 0.5), "leaked_dummy_stat": _choice(rng, ("mean", "median"))},
        "compare_cv_schemes": {"significance_alpha": _choice(rng, (None, 0.05))},
        "adversarial_fold_selection": {"max_union_rows": 50_000, "n_splits": 3, "val_fraction": 0.2},
    }
    cats = ctx.cat_columns()
    if cats:
        pool["subpopulation_drift"] = {"subgroup_col": _choice(rng, cats), "ratio_threshold": float(_choice(rng, (1.5, 2.0, 4.0)))}
    return {"diagnostics_kwargs": {k: pool[k] for k in _subset(rng, list(pool))}}


@_register("OutputConfig", "plot_file")
def _gen_plot_file(rng, ctx, model) -> Dict[str, Any]:
    """A chart prefix under the run's own ``data_dir`` (read by the LTR ranker suite); empty when the run persists nothing."""
    data_dir = getattr(model, "data_dir", None)
    if not data_dir or rng.random() < 0.4:
        return {}
    plot_dir = os.path.join(str(data_dir), "suite_plots")
    os.makedirs(plot_dir, exist_ok=True)
    return {"plot_file": os.path.join(plot_dir, "ranker")}


_REGRESSION_TITLE_TOKENS = (
    "CV_RMSE", "ConcordanceIndex", "ExplainedVariance", "Huber", "Kendall", "MAE", "MAPE_mean", "MBE", "MaxError", "MdAPE", "NSE", "Pearson", "R2",
    "RMSE", "RMSLE", "SMAPE", "Spearman", "wMAPE",
)  # the token if-chain in training/reporting/_reporting_regression (_render_regression_token); unknown tokens render as "" silently
#: Colormap names both renderers resolve (matplotlib by name, plotly via sample_colorscale with a matplotlib-compatible name).
_SHARED_COLORMAPS = ("viridis", "RdBu", "Spectral", "PuOr", "plasma")
_DIAGNOSTIC_NAMES = (
    "pdp_ice", "pdp_2d", "interaction_strength", "engineered_separability", "class_structure", "category_discriminability", "slice_finder",
    "decision_curve", "shap", "shap_interactions", "shap_per_instance", "decile_table", "risk_coverage", "model_card",
)  # the DiagnosticsBudget.run(name, ...) names in training/reporting/_reporting_diagnostics.py


def _panel_template(rng: np.random.Generator, allowed: Sequence[str]) -> str:
    """A space-separated, duplicate-free panel template drawn from a chart module's allowed token set."""
    return " ".join(_subset(rng, sorted(allowed)))


def _plot_outputs_dsl(rng: np.random.Generator) -> str:
    """A ``plot_outputs`` DSL string built from ``BACKEND_FORMATS``.

    Plotly static formats (png/svg/pdf) are left out: each goes through a kaleido Chromium export at 12-15s per figure (see the field's own
    comment), which on a chart-rendering combo exceeds the per-combo wall timeout -- a budget limit, not a validity one.
    """
    from mlframe.reporting.output import BACKEND_FORMATS

    formats = {"matplotlib": sorted(BACKEND_FORMATS["matplotlib"]), "plotly": sorted(BACKEND_FORMATS["plotly"] & {"html", "json"})}
    clauses = [f"{b}[{','.join(_subset(rng, formats[b], max_size=2))}]" for b in _subset(rng, ["matplotlib", "plotly"])]
    return " + ".join(clauses)


@_register("ReportingConfig", "custom_ice_metric", "custom_rice_metric")
def _gen_ice_metrics(rng, ctx, model) -> Dict[str, Any]:
    """Custom ICE / RICE callables with the reporter's ``(y_true=, y_score=[, return_per_class=])`` signature; RICE differs from ICE so its path runs."""
    funcs = (brier_integral_error, binned_ece_integral_error)
    out: Dict[str, Any] = {}
    if rng.random() < 0.6:
        out["custom_ice_metric"] = _choice(rng, funcs)
    if rng.random() < 0.5:
        out["custom_rice_metric"] = _choice(rng, [f for f in funcs if f is not out.get("custom_ice_metric")])
    return out


@_register("ReportingConfig", "decision_costs")
def _gen_decision_costs(rng, ctx, model) -> Dict[str, Any]:
    """``{"fp": .., "fn": ..}`` positive costs (only their ratio matters)."""
    return {} if rng.random() < 0.3 else {"decision_costs": _choice(rng, ({"fp": 1.0, "fn": 12.0}, {"fp": 2.0, "fn": 1.0}, {"fp": 1.0, "fn": 1.0}))}


@_register("ReportingConfig", "diagnostic_splits")
def _gen_diagnostic_splits(rng, ctx, model) -> Dict[str, Any]:
    """Restrict a few real post-fit diagnostics to a subset of the report splits."""
    if rng.random() < 0.3:
        return {}
    return {"diagnostic_splits": {name: tuple(_subset(rng, ["val", "test", "train"], max_size=2)) for name in _subset(rng, _DIAGNOSTIC_NAMES, max_size=4)}}


@_register("ReportingConfig", "feature_importance_config")
def _gen_fi_config(rng, ctx, model) -> Dict[str, Any]:
    """A ``FeatureImportanceConfig`` within the ranges the FI plot can render (at least one factor)."""
    from mlframe.training._training_runtime_configs import FeatureImportanceConfig

    if rng.random() < 0.3:
        return {}
    return {
        "feature_importance_config": FeatureImportanceConfig(
            num_factors=int(rng.integers(1, 25)),
            figsize=_choice(rng, ((8.0, 6.0), (6.0, 4.0))),
            positive_fi_only=bool(rng.random() < 0.5),
            show_plots=bool(rng.random() < 0.5),
            max_zero_fi_to_plot=int(rng.integers(0, 6)),
        )
    }


@_register("ReportingConfig", "learning_curve")
def _gen_learning_curve(rng, ctx, model) -> Dict[str, Any]:
    """An enabled ``LearningCurveConfig`` with a short size sweep and a wall budget (it is K full refits per reported model and split)."""
    from mlframe.training.diagnostics.learning_curve import LearningCurveConfig

    if rng.random() < 0.5:
        return {}
    return {
        "learning_curve": LearningCurveConfig(
            enabled=True,
            sizes=_choice(rng, ((0.5, 1.0), (0.25, 0.5, 1.0))),
            holdout=float(_choice(rng, (0.2, 0.3))),
            n_jobs=int(_choice(rng, (1, 2))),
            parallel_backend=_choice(rng, ("threads", "processes")),
            warm_start=bool(rng.random() < 0.5),
            time_budget_s=float(_choice(rng, (10.0, 30.0))),
            random_state=int(rng.integers(0, 100)),
            score_repeats=int(_choice(rng, (1, 2))),
        )
    }


@_register("ReportingConfig", "binary_panels")
def _gen_binary_panels(rng, ctx, model) -> Dict[str, Any]:
    """A binary-report panel template from ``ALLOWED_BINARY_PANEL_TOKENS``."""
    from mlframe.reporting.charts import ALLOWED_BINARY_PANEL_TOKENS

    return {} if rng.random() < 0.3 else {"binary_panels": _panel_template(rng, ALLOWED_BINARY_PANEL_TOKENS)}


@_register("ReportingConfig", "multilabel_panels")
def _gen_multilabel_panels(rng, ctx, model) -> Dict[str, Any]:
    """A multilabel-report panel template from ``ALLOWED_MULTILABEL_PANEL_TOKENS``."""
    from mlframe.reporting.charts import ALLOWED_MULTILABEL_PANEL_TOKENS

    return {} if rng.random() < 0.3 else {"multilabel_panels": _panel_template(rng, ALLOWED_MULTILABEL_PANEL_TOKENS)}


@_register("ReportingConfig", "regression_panels")
def _gen_regression_panels(rng, ctx, model) -> Dict[str, Any]:
    """An explicit single-figure regression panel template (None keeps the default three-figure split)."""
    from mlframe.reporting.charts import ALLOWED_REGRESSION_PANEL_TOKENS

    return {} if rng.random() < 0.4 else {"regression_panels": _panel_template(rng, ALLOWED_REGRESSION_PANEL_TOKENS)}


@_register("ReportingConfig", "plot_outputs")
def _gen_plot_outputs(rng, ctx, model) -> Dict[str, Any]:
    """A backend x format DSL string (see :func:`_plot_outputs_dsl`)."""
    return {} if rng.random() < 0.3 else {"plot_outputs": _plot_outputs_dsl(rng)}


@_register("ReportingConfig", "regression_title_metrics_tokens")
def _gen_regression_title_tokens(rng, ctx, model) -> Dict[str, Any]:
    """A subset of the regression chart-title tokens the renderer formats."""
    return {} if rng.random() < 0.3 else {"regression_title_metrics_tokens": tuple(_subset(rng, _REGRESSION_TITLE_TOKENS, max_size=8))}


@_register("ReportingConfig", "calibration_colormap")
def _gen_calibration_colormap(rng, ctx, model) -> Dict[str, Any]:
    """A colormap name both renderers resolve."""
    return {} if rng.random() < 0.3 else {"calibration_colormap": _choice(rng, _SHARED_COLORMAPS)}
