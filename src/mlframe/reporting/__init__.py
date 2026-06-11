"""Reporting backend abstraction (matplotlib + plotly) for mlframe charts.

User-facing entrypoints:
- ``parse_plot_output_dsl(s)`` — parse DSL like "plotly[html,png] + matplotlib[pdf]"
- ``render_and_save(spec, output, base_path)`` — render once per backend, save in all formats
- ``Renderer`` Protocol + ``MatplotlibRenderer`` / ``PlotlyRenderer`` impls

Spec dataclasses (``ScatterPanelSpec`` / ``HistogramPanelSpec`` / etc.)
describe one chart in pure-data form so the same spec renders identically
on either backend.

See ``docs/REPORTING.md`` for the design and per-target_type viz catalogue.
"""

from __future__ import annotations


from mlframe.reporting.output import (
    PlotOutputSpec,
    parse_plot_output_dsl,
    BACKEND_FORMATS,
)
from mlframe.reporting.auto_dispatch import render_multi_target_panels
from mlframe.reporting.catalog import available_panels, describe_available_panels
from mlframe.reporting.report_html import ChartEntry, build_combined_report
from mlframe.reporting.charts import (
    build_calibration_drift_spec,
    build_decision_curve_spec,
    calibration_drift,
    compose_model_comparison_figure,
    compose_pdp_figure,
    compose_target_acf_figure,
    find_weak_slices,
    is_tree_model,
    shap_summary_and_dependence,
)

__all__ = [
    "PlotOutputSpec",
    "parse_plot_output_dsl",
    "BACKEND_FORMATS",
    "render_multi_target_panels",
    "available_panels",
    "describe_available_panels",
    "ChartEntry",
    "build_combined_report",
    "compose_pdp_figure",
    "compose_model_comparison_figure",
    "build_decision_curve_spec",
    "find_weak_slices",
    "shap_summary_and_dependence",
    "is_tree_model",
    "calibration_drift",
    "build_calibration_drift_spec",
    "compose_target_acf_figure",
]
