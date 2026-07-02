"""Domain-specific chart builders that produce ``FigureSpec`` instances.

Each chart's ``build_*_spec(...)`` function is pure data prep -- the
returned spec renders identically on either backend via
``mlframe.reporting.renderers.render_and_save(spec, output, base_path)``.

Existing chart functions in ``mlframe.metrics.core`` /
``mlframe.training.targets.regression_residual_audit`` /
``mlframe.training.targets.target_temporal_audit`` are kept as back-compat
wrappers that internally delegate to these builders + render via the
``ReportingConfig.plot_outputs`` DSL.
"""

from __future__ import annotations


from mlframe.reporting.charts.binary import (
    ALLOWED_BINARY_PANEL_TOKENS, DEFAULT_BINARY_PANELS,
    binary_decile_table, compose_binary_figure,
)
from mlframe.reporting.charts.calibration import (
    build_calibration_spec, build_reliability_overlay_spec, wilson_ci,
)
from mlframe.reporting.charts.calibration_drift import (
    CalibrationDriftResult, build_calibration_drift_spec, calibration_drift,
)
from mlframe.reporting.charts.decision_curve import (
    DecisionCurveResult, build_decision_curve_spec, compute_net_benefit,
)
from mlframe.reporting.charts.drift import (
    adversarial_validation, metric_over_time, psi_heatmap, residual_vs_time,
)
from mlframe.reporting.charts.fairness_calibration import (
    compose_fairness_calibration_figure, compute_subgroup_ece_disparity,
)
from mlframe.reporting.charts.calibration_by_feature import (
    compose_calibration_by_feature_figure, compute_calibration_by_feature_heterogeneity,
)
from mlframe.reporting.charts.calibration_heatmap_2d import (
    compose_calibration_heatmap_2d_figure, compute_calibration_heatmap_2d,
)
from mlframe.reporting.charts.model_comparison import compose_model_comparison_figure
from mlframe.reporting.charts.model_card import (
    ModelCardVerdict, compose_model_card_figure, model_card_verdict,
)
from mlframe.reporting.charts.split_comparison import (
    OverfitVerdict, compose_split_comparison_figure, overfit_verdict,
)
from mlframe.reporting.charts.pdp_ice import (
    compose_pdp_figure, compute_pdp, compute_pdp_2d, pdp_2d_panel, pdp_panel,
)
from mlframe.reporting.charts.shap_panels import (
    ShapPanelsResult, is_tree_model, shap_summary_and_dependence,
)
from mlframe.reporting.charts.shap_per_instance import (
    ShapPerInstanceResult, shap_worst_errors_explanation,
)
from mlframe.reporting.charts.slice_finder import (
    SliceFinderResult, find_weak_slices,
)
from mlframe.reporting.charts.error_analysis import (
    error_bias_per_feature, segments_bar, target_dist_overlay,
    weak_segment_heatmap, worst_k_table,
)
from mlframe.reporting.charts.ltr import (
    ALLOWED_LTR_PANEL_TOKENS, compose_ltr_figure,
)
from mlframe.reporting.charts.multiclass import (
    ALLOWED_MULTICLASS_PANEL_TOKENS, compose_multiclass_figure,
)
from mlframe.reporting.charts.multilabel import (
    ALLOWED_MULTILABEL_PANEL_TOKENS, compose_multilabel_figure,
)
from mlframe.reporting.charts.quantile import (
    ALLOWED_QUANTILE_PANEL_TOKENS, compose_quantile_figure,
)
from mlframe.reporting.charts.regression import (
    ALLOWED_REGRESSION_PANEL_TOKENS, DEFAULT_REGRESSION_PANELS,
    build_regression_panel_spec, compose_regression_figure,
)
from mlframe.reporting.charts.temporal import (
    ALLOWED_TEMPORAL_PANEL_TOKENS, DEFAULT_TEMPORAL_TARGET_PANELS,
    build_temporal_audit_spec, compose_target_acf_figure,
)
from mlframe.reporting.charts.training_curve import compose_training_curve_figure
from mlframe.reporting.charts.confusion_matrix_plot import (
    plot_confusion_matrix, confusion_matrix_counts,
)
# PZAD-derived diagnostics (case_visual / case_sdsj / fuzzy / SGT). Exposed on the package surface so callers reach them
# from the gallery facade rather than the implementation modules.
from mlframe.reporting.charts.class_structure_heatmap import (
    class_structure_matrix, class_structure_panel, compose_class_structure_figure,
)
from mlframe.reporting.charts.engineered_separability import (
    compose_separability_figure, separability_panel, separability_score,
)
from mlframe.reporting.charts.category_discriminability import (
    category_discriminability_panel, category_discriminability_table,
    compose_category_discriminability_figure, level_woe,
)
from mlframe.reporting.charts.fuzzy_membership import (
    compose_fuzzy_membership_figure, fuzzy_membership_curves, fuzzy_membership_panel,
)
from mlframe.reporting.charts.spectral_embedding import (
    compose_spectral_embedding_figure, spectral_embedding_panel, spectral_layout,
)
# Public re-export of the chart-sampling helpers so cross-package consumers (renderers, diagnostics_dispatch, training-side reporting) import them from the package surface instead of the ``_sampling`` implementation module.
from mlframe.reporting.charts._sampling import subsample_preserving_extremes, prebin_histogram

__all__ = [
    "subsample_preserving_extremes",
    "prebin_histogram",
    "class_structure_matrix",
    "class_structure_panel",
    "compose_class_structure_figure",
    "separability_score",
    "separability_panel",
    "compose_separability_figure",
    "level_woe",
    "category_discriminability_table",
    "category_discriminability_panel",
    "compose_category_discriminability_figure",
    "fuzzy_membership_curves",
    "fuzzy_membership_panel",
    "compose_fuzzy_membership_figure",
    "spectral_layout",
    "spectral_embedding_panel",
    "compose_spectral_embedding_figure",
    "build_calibration_spec",
    "build_reliability_overlay_spec",
    "wilson_ci",
    "build_regression_panel_spec",
    "compose_regression_figure",
    "build_temporal_audit_spec",
    "compose_target_acf_figure",
    "ALLOWED_TEMPORAL_PANEL_TOKENS",
    "DEFAULT_TEMPORAL_TARGET_PANELS",
    "compose_binary_figure",
    "binary_decile_table",
    "compose_multiclass_figure",
    "plot_confusion_matrix",
    "confusion_matrix_counts",
    "compose_multilabel_figure",
    "compose_ltr_figure",
    "compose_quantile_figure",
    "compose_training_curve_figure",
    "compose_pdp_figure",
    "compute_pdp",
    "compute_pdp_2d",
    "pdp_panel",
    "pdp_2d_panel",
    "compose_model_comparison_figure",
    "compose_fairness_calibration_figure",
    "compose_calibration_by_feature_figure",
    "compute_calibration_by_feature_heterogeneity",
    "compose_calibration_heatmap_2d_figure",
    "compute_calibration_heatmap_2d",
    "compute_subgroup_ece_disparity",
    "compose_model_card_figure",
    "model_card_verdict",
    "compose_split_comparison_figure",
    "overfit_verdict",
    "OverfitVerdict",
    "ModelCardVerdict",
    "build_decision_curve_spec",
    "compute_net_benefit",
    "DecisionCurveResult",
    "find_weak_slices",
    "SliceFinderResult",
    "shap_summary_and_dependence",
    "ShapPerInstanceResult",
    "shap_worst_errors_explanation",
    "is_tree_model",
    "ShapPanelsResult",
    "calibration_drift",
    "build_calibration_drift_spec",
    "CalibrationDriftResult",
    "weak_segment_heatmap",
    "error_bias_per_feature",
    "segments_bar",
    "worst_k_table",
    "target_dist_overlay",
    "psi_heatmap",
    "residual_vs_time",
    "metric_over_time",
    "adversarial_validation",
    "ALLOWED_BINARY_PANEL_TOKENS",
    "DEFAULT_BINARY_PANELS",
    "ALLOWED_MULTICLASS_PANEL_TOKENS",
    "ALLOWED_MULTILABEL_PANEL_TOKENS",
    "ALLOWED_LTR_PANEL_TOKENS",
    "ALLOWED_QUANTILE_PANEL_TOKENS",
    "ALLOWED_REGRESSION_PANEL_TOKENS",
    "DEFAULT_REGRESSION_PANELS",
]
