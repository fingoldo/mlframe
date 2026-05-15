"""Renderer dispatch.

Public surface:
- ``Renderer`` Protocol — one render() + one save() + one show() method.
- ``get_renderer(backend)`` factory.
- ``render_and_save(spec, output, base_path)`` — high-level orchestrator
  that loops over the parsed PlotOutputSpec and dispatches to the right
  renderer + saves to all requested formats.
"""

from __future__ import annotations


from mlframe.reporting.renderers.base import Renderer, get_renderer
from mlframe.reporting.renderers.save import render_and_save

__all__ = ["Renderer", "get_renderer", "render_and_save"]
