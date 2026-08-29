"""Inline-display and render-skipping policy for ``render_and_save``.

Two behaviours a notebook user depends on:

* The default ``plot_outputs`` ("plotly[html] + matplotlib[png]") asks for BOTH backends, and inline display
  used to fire for each -- so every chart appeared TWICE in the cell, an interactive plotly figure plus a
  static matplotlib duplicate of the same data. matplotlib is a save-only backend: its file is written
  exactly as before, but it never renders into the cell.
* With saving switched off, a save-only backend has no consumer left at all, so it must not even be
  rendered -- on a multi-model suite that is seconds of matplotlib work whose output nothing ever sees.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers.save import render_and_save
from mlframe.reporting.spec import FigureSpec, LinePanelSpec

_BOTH = "plotly[html] + matplotlib[png]"


def _spec():
    """One-panel line spec, cheap to render on either backend."""
    x = np.arange(10)
    return FigureSpec(panels=((LinePanelSpec(x=x, y=x.astype(float), title="t", series_labels=("s",)),),), figsize=(6, 4))


@pytest.fixture
def shown(monkeypatch):
    """Record which backends had ``renderer.show`` called on them."""
    seen: list = []
    import mlframe.reporting.renderers.save as save_mod

    real_get = save_mod.get_renderer

    def _spy_get(backend):
        """Record which backends the dispatcher asked for."""
        r = real_get(backend)

        class _Wrapped:
            """Renderer stand-in that records the calls made against it."""
            backend = r.backend

            def render(self, *a, **kw):
                """Delegate rendering to the real renderer."""
                return r.render(*a, **kw)

            def save(self, *a, **kw):
                """Delegate saving to the real renderer."""
                return r.save(*a, **kw)

            def show(self, fig):
                """Record the show call instead of opening a window."""
                seen.append(backend)

        return _Wrapped()

    monkeypatch.setattr(save_mod, "get_renderer", _spy_get)
    return seen


def test_matplotlib_is_not_shown_inline_but_is_still_saved(shown, tmp_path):
    """In a notebook only the interactive backend reaches the cell; the PNG is written regardless."""
    base = str(tmp_path / "chart")
    render_and_save(_spec(), parse_plot_output_dsl(_BOTH), base, interactive=True)

    assert "plotly" in shown, shown
    assert "matplotlib" not in shown, shown
    # The save side is untouched by the display policy. A direct ``render_and_save`` caller keeps the FLAT layout:
    # per-format subfolders are opt-in for the library and are turned on by the training suite's config, so a library
    # upgrade cannot move a direct caller's files (test_save_format_subfolders.py covers the on case).
    assert (tmp_path / "chart.matplotlib.png").exists()
    assert (tmp_path / "chart.plotly.html").exists()


def test_matplotlib_is_not_rendered_at_all_when_nothing_is_saved(monkeypatch, tmp_path):
    """With saving off, a save-only backend has no consumer -- skip the render entirely."""
    rendered: list = []
    import mlframe.reporting.renderers.save as save_mod

    real_get = save_mod.get_renderer

    def _spy_get(backend):
        """Record which backends the dispatcher asked for."""
        r = real_get(backend)

        class _Wrapped:
            """Renderer stand-in that records the calls made against it."""
            backend = r.backend

            def render(self, *a, **kw):
                """Record that this backend actually built a figure."""
                rendered.append(backend)
                return r.render(*a, **kw)

            def save(self, *a, **kw):
                """Delegate saving to the real renderer."""
                return r.save(*a, **kw)

            def show(self, fig):
                """No-op display."""

        return _Wrapped()

    monkeypatch.setattr(save_mod, "get_renderer", _spy_get)
    render_and_save(_spec(), parse_plot_output_dsl(_BOTH), "", interactive=True)

    assert rendered == ["plotly"], rendered


def test_nothing_is_rendered_when_neither_saved_nor_displayed(monkeypatch):
    """No save path and no inline display means no backend has a consumer."""
    rendered: list = []
    import mlframe.reporting.renderers.save as save_mod

    real_get = save_mod.get_renderer

    def _spy_get(backend):
        """Record which backends the dispatcher asked for."""
        r = real_get(backend)

        class _Wrapped:
            """Renderer stand-in that records the calls made against it."""
            backend = r.backend

            def render(self, *a, **kw):
                """Record that this backend actually built a figure."""
                rendered.append(backend)
                return r.render(*a, **kw)

            def save(self, *a, **kw):
                """Delegate saving to the real renderer."""
                return r.save(*a, **kw)

            def show(self, fig):
                """No-op display."""

        return _Wrapped()

    monkeypatch.setattr(save_mod, "get_renderer", _spy_get)
    render_and_save(_spec(), parse_plot_output_dsl(_BOTH), "", interactive=False)

    assert rendered == [], rendered


def test_keep_handles_still_renders_every_requested_backend(tmp_path):
    """``keep_handles`` is an explicit request for the figure objects, so nothing may be skipped."""
    handles = render_and_save(_spec(), parse_plot_output_dsl(_BOTH), "", keep_handles=True, interactive=False)

    assert set(handles or {}) == {"plotly", "matplotlib"}, handles
