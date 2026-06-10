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
from mlframe.reporting.charts.drift import (
    adversarial_validation, metric_over_time, psi_heatmap, residual_vs_time,
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
from mlframe.reporting.charts.temporal import build_temporal_audit_spec
from mlframe.reporting.charts.training_curve import compose_training_curve_figure

__all__ = [
    "build_calibration_spec",
    "build_reliability_overlay_spec",
    "wilson_ci",
    "build_regression_panel_spec",
    "compose_regression_figure",
    "build_temporal_audit_spec",
    "compose_binary_figure",
    "binary_decile_table",
    "compose_multiclass_figure",
    "compose_multilabel_figure",
    "compose_ltr_figure",
    "compose_quantile_figure",
    "compose_training_curve_figure",
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
