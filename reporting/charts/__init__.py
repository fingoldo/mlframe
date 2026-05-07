"""Domain-specific chart builders that produce ``FigureSpec`` instances.

Each chart's ``build_*_spec(...)`` function is pure data prep -- the
returned spec renders identically on either backend via
``mlframe.reporting.renderers.render_and_save(spec, output, base_path)``.

Existing chart functions in ``mlframe.metrics`` /
``mlframe.training.regression_residual_audit`` /
``mlframe.training.target_temporal_audit`` are kept as back-compat
wrappers that internally delegate to these builders + render via the
``ReportingConfig.plot_outputs`` DSL.
"""

from mlframe.reporting.charts.calibration import build_calibration_spec
from mlframe.reporting.charts.regression import build_regression_panel_spec
from mlframe.reporting.charts.temporal import build_temporal_audit_spec

__all__ = [
    "build_calibration_spec",
    "build_regression_panel_spec",
    "build_temporal_audit_spec",
]
