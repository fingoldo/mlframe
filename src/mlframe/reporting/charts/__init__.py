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


from mlframe.reporting.charts.calibration import build_calibration_spec
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
from mlframe.reporting.charts.regression import build_regression_panel_spec
from mlframe.reporting.charts.temporal import build_temporal_audit_spec

__all__ = [
    "build_calibration_spec",
    "build_regression_panel_spec",
    "build_temporal_audit_spec",
    "compose_multiclass_figure",
    "compose_multilabel_figure",
    "compose_ltr_figure",
    "compose_quantile_figure",
    "ALLOWED_MULTICLASS_PANEL_TOKENS",
    "ALLOWED_MULTILABEL_PANEL_TOKENS",
    "ALLOWED_LTR_PANEL_TOKENS",
    "ALLOWED_QUANTILE_PANEL_TOKENS",
]
