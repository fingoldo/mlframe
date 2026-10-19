"""Stages of ``_render_post_fit_diagnostics``."""

from __future__ import annotations

# --- imports (managed) ---
from mlframe.training.reporting._reporting_diagnostics import _binary_positive_score
# --- end imports ---


def _render_standalone_charts(cfg, y_arr, _collapsed, _budget, model, df, names, importances, plot_outputs, plot_file, metrics, _multilabel, tt, y_pred, task):
    """Stage of ``_render_post_fit_diagnostics``."""
    from mlframe.reporting.diagnostics_dispatch import (
        render_category_discriminability_diagnostic,
        render_class_structure_diagnostic,
        render_engineered_separability_diagnostic,
        render_interaction_strength_diagnostic,
        render_pdp_2d_diagnostic,
        render_pdp_ice_diagnostic,
        render_slice_finder_diagnostic,
    )

    if getattr(cfg, "pdp_ice", False) and y_arr is not None and not _collapsed:
        _budget.run(
            "pdp_ice",
            lambda: render_pdp_ice_diagnostic(
                model=model,
                df=df,
                feature_names=names,
                feature_importances=importances,
                plot_outputs=plot_outputs,
                base_path=plot_file,
                metrics_dict=metrics,
                top_features=getattr(cfg, "pdp_top_features", 4),
                sample=getattr(cfg, "pdp_sample", 2000),
                grid=getattr(cfg, "pdp_grid", 20),
            ),
        )

    if getattr(cfg, "pdp_2d_charts", False) and y_arr is not None and not _collapsed:
        _budget.run(
            "pdp_2d",
            lambda: render_pdp_2d_diagnostic(
                model=model,
                df=df,
                feature_names=names,
                feature_importances=importances,
                plot_outputs=plot_outputs,
                base_path=plot_file,
                metrics_dict=metrics,
                sample=getattr(cfg, "pdp_sample", 2000),
                grid=getattr(cfg, "pdp_grid", 20),
            ),
        )

    if getattr(cfg, "interaction_strength_charts", False) and y_arr is not None and not _collapsed:
        _budget.run(
            "interaction_strength",
            lambda: render_interaction_strength_diagnostic(
                model=model,
                df=df,
                feature_names=names,
                feature_importances=importances,
                plot_outputs=plot_outputs,
                base_path=plot_file,
                metrics_dict=metrics,
                max_features=getattr(cfg, "interaction_strength_max_features", 8),
                sample=getattr(cfg, "pdp_sample", 2000),
                grid=getattr(cfg, "pdp_grid", 20),
                max_seconds=getattr(cfg, "interaction_strength_max_seconds", 20.0),
            ),
        )

    if getattr(cfg, "engineered_separability_charts", True) and df is not None and y_arr is not None and not _collapsed and not _multilabel:
        _budget.run(
            "engineered_separability",
            lambda: render_engineered_separability_diagnostic(
                df=df,
                y_true=y_arr,
                feature_names=names,
                feature_importances=importances,
                plot_outputs=plot_outputs,
                base_path=plot_file,
                metrics_dict=metrics,
                sample=getattr(cfg, "pdp_sample", 2000) * 2,
            ),
        )

    if getattr(cfg, "class_structure_charts", True) and df is not None and y_arr is not None and not _multilabel:
        _budget.run(
            "class_structure",
            lambda: render_class_structure_diagnostic(
                df=df,
                y_true=y_arr,
                feature_names=names,
                plot_outputs=plot_outputs,
                base_path=plot_file,
                metrics_dict=metrics,
                max_groups=getattr(cfg, "class_structure_max_groups", 30),
                n_time_bins=getattr(cfg, "class_structure_time_bins", 20),
            ),
        )

    if getattr(cfg, "category_discriminability_charts", True) and df is not None and tt == "binary_classification" and y_arr is not None and not _multilabel:
        _budget.run(
            "category_discriminability",
            lambda: render_category_discriminability_diagnostic(
                df=df,
                y_true=y_arr,
                feature_names=names,
                feature_importances=importances,
                plot_outputs=plot_outputs,
                base_path=plot_file,
                metrics_dict=metrics,
                top_k=getattr(cfg, "category_discriminability_top_k", 15),
                max_columns=getattr(cfg, "category_discriminability_max_columns", 40),
            ),
        )

    if (
        getattr(cfg, "slice_finder", False)
        and df is not None
        and y_arr is not None
        and y_pred is not None
        and not _multilabel
        and len(y_pred) == len(y_arr)
        and not _collapsed
    ):
        _budget.run(
            "slice_finder",
            lambda: render_slice_finder_diagnostic(
                df=df,
                y_true=y_arr,
                y_pred=y_pred,
                task=task,
                feature_names=names,
                plot_outputs=plot_outputs,
                base_path=plot_file,
                metrics_dict=metrics,
            ),
        )


def _render_shap_panels(cfg, model, df, _collapsed, _budget, names, plot_outputs, plot_file, metrics, plot_dpi, y_arr, tt, probs, task, y_pred):
    """Stage of ``_render_post_fit_diagnostics``."""
    from mlframe.reporting.diagnostics_dispatch import (
        render_shap_diagnostic,
        render_shap_interactions_diagnostic,
        render_shap_per_instance_diagnostic,
    )

    if getattr(cfg, "shap_panels", False) and model is not None and df is not None and not _collapsed:
        _budget.run(
            "shap",
            lambda: render_shap_diagnostic(
                model=model,
                df=df,
                feature_names=names,
                plot_outputs=plot_outputs,
                base_path=plot_file,
                metrics_dict=metrics,
                max_rows=getattr(cfg, "shap_max_rows", 20000),
                plot_dpi=plot_dpi,
                top_k=getattr(cfg, "shap_top_k", 6),
                allow_kernel=getattr(cfg, "shap_allow_kernel", False),
            ),
        )

    if getattr(cfg, "shap_interactions", False) and model is not None and df is not None and not _collapsed:
        _budget.run(
            "shap_interactions",
            lambda: render_shap_interactions_diagnostic(
                model=model,
                df=df,
                feature_names=names,
                plot_outputs=plot_outputs,
                base_path=plot_file,
                metrics_dict=metrics,
                max_rows=getattr(cfg, "shap_interaction_max_rows", 2000),
            ),
        )

    if getattr(cfg, "shap_per_instance", False) and model is not None and df is not None and y_arr is not None:
        # Per-instance needs a 1-D score: binary positive-class prob, else the regression prediction.
        _yscore = _binary_positive_score(probs) if tt == "binary_classification" else (y_pred if task == "regression" else None)
        if _yscore is not None and len(_yscore) == len(y_arr):
            _budget.run(
                "shap_per_instance",
                lambda: render_shap_per_instance_diagnostic(
                    model=model,
                    df=df,
                    y_true=y_arr,
                    y_score=_yscore,
                    feature_names=names,
                    plot_outputs=plot_outputs,
                    base_path=plot_file,
                    metrics_dict=metrics,
                ),
            )


def _render_decision_curve_panel(cfg, tt, y_arr, probs, _budget, plot_outputs, plot_file, metrics):
    """Stage of ``_render_post_fit_diagnostics``."""
    from mlframe.reporting.diagnostics_dispatch import (
        render_decision_curve_diagnostic,
    )

    if getattr(cfg, "decision_curve", True) and tt == "binary_classification" and y_arr is not None:
        _bs = _binary_positive_score(probs)
        if _bs is not None and len(_bs) == len(y_arr):
            # Bound to a non-Optional local: the lambda defers execution, so the enclosing narrowing
            # of ``_bs`` does not reach inside it.
            _score = _bs
            _budget.run(
                "decision_curve",
                lambda: render_decision_curve_diagnostic(
                    y_true=y_arr,
                    y_score=_score,
                    plot_outputs=plot_outputs,
                    base_path=plot_file,
                    metrics_dict=metrics,
                ),
            )
