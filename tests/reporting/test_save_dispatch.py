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
        assert isinstance(result, dict)
        assert "plotly" in result
        assert "matplotlib" in result
        # Native handles
        assert hasattr(result["plotly"], "to_html")
        assert hasattr(result["matplotlib"], "savefig")
