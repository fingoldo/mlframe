"""Tests for ``render_and_save`` dispatch + file naming."""

from __future__ import annotations

import os

import numpy as np
import pytest

from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import render_and_save
from mlframe.reporting.spec import FigureSpec, ScatterPanelSpec


@pytest.fixture
def trivial_spec():
    return FigureSpec(
        suptitle="t",
        panels=((ScatterPanelSpec(
            x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]),
            title="s", xlabel="x", ylabel="y",
        ),),),
        figsize=(4, 3),
    )


class TestNamingPolicy:
    def test_single_backend_single_format_uses_short_path(self, trivial_spec, tmp_path):
        """``base_path.fmt`` (no backend in filename) when only one
        backend × one format requested."""
        out = parse_plot_output_dsl("matplotlib[png]")
        base = str(tmp_path / "plot")
        render_and_save(trivial_spec, out, base)
        assert os.path.exists(base + ".png")
        assert not os.path.exists(base + ".matplotlib.png")

    def test_multi_backend_uses_backend_in_filename(self, trivial_spec, tmp_path):
        out = parse_plot_output_dsl("plotly[html] + matplotlib[png]")
        base = str(tmp_path / "plot")
        render_and_save(trivial_spec, out, base)
        assert os.path.exists(base + ".plotly.html")
        assert os.path.exists(base + ".matplotlib.png")
        assert not os.path.exists(base + ".html")
        assert not os.path.exists(base + ".png")

    def test_single_backend_multi_format_uses_backend_in_filename(self, trivial_spec, tmp_path):
        out = parse_plot_output_dsl("plotly[html,json]")
        base = str(tmp_path / "plot")
        render_and_save(trivial_spec, out, base)
        assert os.path.exists(base + ".plotly.html")
        assert os.path.exists(base + ".plotly.json")


class TestKeepHandles:
    def test_default_releases_handles(self, trivial_spec, tmp_path):
        out = parse_plot_output_dsl("matplotlib[png]")
        result = render_and_save(trivial_spec, out, str(tmp_path / "p"))
        assert result is None

    def test_keep_handles_returns_dict(self, trivial_spec, tmp_path):
        out = parse_plot_output_dsl("plotly[html] + matplotlib[png]")
        result = render_and_save(trivial_spec, out, str(tmp_path / "p"), keep_handles=True)
        assert isinstance(result, dict) and set(result.keys()) == {"plotly", "matplotlib"}, (
            f"expected exactly {{plotly, matplotlib}} keys; got {set(result.keys()) if isinstance(result, dict) else type(result).__name__}"
        )
        # Native handles -- each must be the real backend object so callers can chain
        # .to_html() / .savefig().
        assert hasattr(result["plotly"], "to_html")
        assert hasattr(result["matplotlib"], "savefig")


class TestInteractiveDisplay:
    """``interactive=True`` calls ``renderer.show(fig)`` per backend so
    figures render inline in the notebook cell IN ADDITION to the on-
    disk save. Auto-detected via ``__IPYTHON__``/``sys.ps1`` when
    ``interactive=None``; explicit override skips detection."""

    def test_interactive_false_save_only(self, trivial_spec, tmp_path, monkeypatch):
        """interactive=False: file saved, renderer.show NOT called.
        Pre-2026-05-10 default behaviour for non-jupyter."""
        out = parse_plot_output_dsl("matplotlib[png]")
        base = str(tmp_path / "p")
        # Patch the class method so any instance get_renderer() returns
        # honors the spy (get_renderer returns a fresh instance per call).
        from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
        show_calls = []
        monkeypatch.setattr(
            MatplotlibRenderer, "show",
            lambda self, fig: show_calls.append(fig),
        )
        render_and_save(trivial_spec, out, base, interactive=False)
        # File saved, show NOT called.
        assert os.path.exists(base + ".png")
        assert show_calls == []

    def test_interactive_true_calls_show_per_backend(self, trivial_spec, tmp_path, monkeypatch):
        """interactive=True: file saved AND renderer.show called once per backend."""
        out = parse_plot_output_dsl("matplotlib[png]")
        base = str(tmp_path / "p")
        from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
        show_calls = []
        monkeypatch.setattr(
            MatplotlibRenderer, "show",
            lambda self, fig: show_calls.append(fig),
        )
        render_and_save(trivial_spec, out, base, interactive=True)
        assert os.path.exists(base + ".png")
        assert len(show_calls) == 1

    def test_interactive_none_auto_detects_non_ipython(self, trivial_spec, tmp_path, monkeypatch):
        """interactive=None in a non-IPython context (no __IPYTHON__, no
        sys.ps1) auto-detects to False → save-only."""
        out = parse_plot_output_dsl("matplotlib[png]")
        base = str(tmp_path / "p")
        import builtins
        import sys
        if hasattr(builtins, "__IPYTHON__"):
            monkeypatch.delattr(builtins, "__IPYTHON__")
        if hasattr(sys, "ps1"):
            monkeypatch.delattr(sys, "ps1")

        from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
        show_calls = []
        monkeypatch.setattr(
            MatplotlibRenderer, "show",
            lambda self, fig: show_calls.append(fig),
        )
        render_and_save(trivial_spec, out, base, interactive=None)
        # Auto-detected non-interactive → show NOT called.
        assert show_calls == []
        assert os.path.exists(base + ".png")

    def test_interactive_show_failure_does_not_break_save(self, trivial_spec, tmp_path, monkeypatch):
        """If renderer.show raises, the on-disk save still completes
        and render_and_save doesn't propagate the exception."""
        out = parse_plot_output_dsl("matplotlib[png]")
        base = str(tmp_path / "p")
        from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer

        def _explode(self, fig):
            raise RuntimeError("simulated jupyter display backend failure")
        monkeypatch.setattr(MatplotlibRenderer, "show", _explode)
        # Must not raise — show failures are non-fatal.
        render_and_save(trivial_spec, out, base, interactive=True)
        assert os.path.exists(base + ".png")


class TestInlineDisplayOptOut:
    """Process-wide opt-out via ``MLFRAME_PLOT_INLINE_DISPLAY`` env var
    (set directly or via ``set_inline_display_mode``). Lets batch
    jupyter runs (papermill / nbconvert / scheduled notebooks) skip the
    inline render even when ``__IPYTHON__`` is set."""

    def test_env_var_force_false_overrides_ipython(self, trivial_spec, tmp_path, monkeypatch):
        """Even with __IPYTHON__ set, env var=0 → save-only."""
        import builtins
        # Simulate jupyter kernel
        monkeypatch.setattr(builtins, "__IPYTHON__", True, raising=False)
        # But operator wants save-only via env
        monkeypatch.setenv("MLFRAME_PLOT_INLINE_DISPLAY", "0")
        from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
        show_calls = []
        monkeypatch.setattr(
            MatplotlibRenderer, "show",
            lambda self, fig: show_calls.append(fig),
        )
        out = parse_plot_output_dsl("matplotlib[png]")
        base = str(tmp_path / "p")
        render_and_save(trivial_spec, out, base, interactive=None)
        # Env var won → no inline display.
        assert show_calls == []
        assert os.path.exists(base + ".png")

    def test_env_var_force_true_overrides_non_ipython(self, trivial_spec, tmp_path, monkeypatch):
        """Even outside a kernel, env var=1 → inline display fires."""
        import builtins
        import sys
        if hasattr(builtins, "__IPYTHON__"):
            monkeypatch.delattr(builtins, "__IPYTHON__")
        if hasattr(sys, "ps1"):
            monkeypatch.delattr(sys, "ps1")
        monkeypatch.setenv("MLFRAME_PLOT_INLINE_DISPLAY", "1")
        from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
        show_calls = []
        monkeypatch.setattr(
            MatplotlibRenderer, "show",
            lambda self, fig: show_calls.append(fig),
        )
        out = parse_plot_output_dsl("matplotlib[png]")
        base = str(tmp_path / "p")
        render_and_save(trivial_spec, out, base, interactive=None)
        assert len(show_calls) == 1
        assert os.path.exists(base + ".png")

    def test_setter_helper_writes_env_var(self, monkeypatch):
        """``set_inline_display_mode`` writes/clears the env var without
        requiring callers to manipulate ``os.environ`` directly."""
        from mlframe.reporting.renderers.save import (
            set_inline_display_mode, _detect_interactive_session,
        )
        # Clean slate
        monkeypatch.delenv("MLFRAME_PLOT_INLINE_DISPLAY", raising=False)
        # Hide IPython markers so auto-detect is False
        import builtins
        import sys
        if hasattr(builtins, "__IPYTHON__"):
            monkeypatch.delattr(builtins, "__IPYTHON__")
        if hasattr(sys, "ps1"):
            monkeypatch.delattr(sys, "ps1")

        # auto-detect → False (no kernel + no env var)
        assert _detect_interactive_session() is False

        # Force True
        set_inline_display_mode(True)
        assert _detect_interactive_session() is True
        assert os.environ.get("MLFRAME_PLOT_INLINE_DISPLAY") == "1"

        # Force False
        set_inline_display_mode(False)
        assert _detect_interactive_session() is False
        assert os.environ.get("MLFRAME_PLOT_INLINE_DISPLAY") == "0"

        # Clear → falls back to auto-detect (False with no kernel)
        set_inline_display_mode(None)
        assert "MLFRAME_PLOT_INLINE_DISPLAY" not in os.environ
        assert _detect_interactive_session() is False

        # Reject garbage
        with pytest.raises(ValueError):
            set_inline_display_mode("yes")  # type: ignore[arg-type]

    def test_unrecognized_env_value_falls_through_to_auto_detect(self, monkeypatch):
        """A typo in the env var (``MLFRAME_PLOT_INLINE_DISPLAY=maybe``)
        must not silently flip behavior — falls through to auto-detect."""
        from mlframe.reporting.renderers.save import _detect_interactive_session
        import builtins
        import sys
        if hasattr(builtins, "__IPYTHON__"):
            monkeypatch.delattr(builtins, "__IPYTHON__")
        if hasattr(sys, "ps1"):
            monkeypatch.delattr(sys, "ps1")
        monkeypatch.setenv("MLFRAME_PLOT_INLINE_DISPLAY", "maybe")
        # Falls through to auto-detect (False without IPython markers).
        assert _detect_interactive_session() is False
