"""Renderer dispatch.

Public surface:
- ``Renderer`` Protocol — one render() + one save() + one show() method.
- ``get_renderer(backend)`` factory.
- ``render_and_save(spec, output, base_path)`` — high-level orchestrator
  that loops over the parsed PlotOutputSpec and dispatches to the right
  renderer + saves to all requested formats.
"""

from __future__ import annotations


from mlframe.reporting.renderers._render_timings import (
    chart_timings_snapshot, format_chart_timings, record_chart_render, reset_chart_timings,
)
from mlframe.reporting.renderers.base import Renderer, get_renderer
from mlframe.reporting.renderers.save import (
    get_render_failure_stats, render_and_save, reset_render_failure_stats,
)

__all__ = [
    "Renderer",
    "get_renderer",
    "render_and_save",
    "get_render_failure_stats",
    "reset_render_failure_stats",
    "chart_timings_snapshot",
    "format_chart_timings",
    "record_chart_render",
    "reset_chart_timings",
]
