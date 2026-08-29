"""Per-format output subfolders: ``png/plot.png`` + ``html/plot.html`` instead of one mixed directory.

The SUITE enables this (``ReportingConfig.plot_format_subfolders``); the library default stays flat so a
direct ``render_and_save`` caller keeps its on-disk contract. Tests that assert the nested layout therefore
request it, exactly as the suite does.

With the default ``plot_outputs`` every figure is written twice, so a multi-model suite used to leave hundreds of
files of two kinds interleaved in one listing.
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import render_and_save
from mlframe.reporting.renderers.save import (
    get_format_subfolders, resolve_output_path, set_format_subfolders,
)
from mlframe.reporting.spec import FigureSpec, ScatterPanelSpec


@pytest.fixture
def spec():
    """A one-panel figure."""
    return FigureSpec(
        panels=((ScatterPanelSpec(x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]), title="s"),),),
        figsize=(4.0, 3.0),
    )


@pytest.fixture(autouse=True)
def _clean_override():
    """Leave no thread override behind: it would flip the layout for every later test in this process."""
    prior = get_format_subfolders()
    yield
    set_format_subfolders(prior)


class TestLayout:
    """Where each file lands."""

    def test_both_backends_land_in_their_own_format_folder(self, spec, tmp_path):
        """The whole point: the interactive copy and the static copy stop sharing a directory."""
        base = str(tmp_path / "plot")
        render_and_save(spec, parse_plot_output_dsl("plotly[html] + matplotlib[png]"), base, interactive=False, format_subfolders=True)
        assert (tmp_path / "html" / "plot.plotly.html").exists()
        assert (tmp_path / "png" / "plot.matplotlib.png").exists()
        assert not (tmp_path / "plot.plotly.html").exists()

    def test_the_filename_is_unchanged_by_the_layout(self, spec, tmp_path):
        """A caller that knows the flat name finds the file by prepending the format directory, nothing else."""
        base = str(tmp_path / "plot")
        render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"), base, interactive=False, format_subfolders=True)
        flat = resolve_output_path(base, "matplotlib", "png", multi_output=False, subfolders=False)
        nested = resolve_output_path(base, "matplotlib", "png", multi_output=False, subfolders=True)
        assert os.path.basename(flat) == os.path.basename(nested)
        assert os.path.exists(nested) and not os.path.exists(flat)

    def test_the_flat_layout_is_still_available(self, spec, tmp_path):
        """An explicit False must win over the default, for callers with a fixed on-disk contract."""
        base = str(tmp_path / "plot")
        render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"), base, interactive=False, format_subfolders=False)
        assert (tmp_path / "plot.png").exists()
        assert not (tmp_path / "png").exists()

    def test_multi_format_on_one_backend_splits_by_format(self, spec, tmp_path):
        """Two formats from one backend are two different KINDS of artifact, so they split too."""
        base = str(tmp_path / "plot")
        render_and_save(spec, parse_plot_output_dsl("plotly[html,json]"), base, interactive=False, format_subfolders=True)
        assert (tmp_path / "html" / "plot.plotly.html").exists()
        assert (tmp_path / "json" / "plot.plotly.json").exists()


class TestOverrideResolution:
    """Thread override, then env var, then the module default."""

    def test_thread_override_beats_the_default(self, spec, tmp_path):
        """This is how the suite config reaches render_and_save without threading a parameter through every call."""
        set_format_subfolders(True)  # the opposite of the library default, so the override is what is under test
        base = str(tmp_path / "plot")
        render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"), base, interactive=False)
        assert (tmp_path / "png" / "plot.png").exists()

    def test_env_var_is_read_when_no_thread_override_is_set(self, spec, tmp_path, monkeypatch):
        """For runs that cannot reach the config object (subprocess workers, notebooks driven externally)."""
        set_format_subfolders(None)
        monkeypatch.setenv("MLFRAME_PLOT_FORMAT_SUBFOLDERS", "1")
        base = str(tmp_path / "plot")
        render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"), base, interactive=False)
        assert (tmp_path / "png" / "plot.png").exists()

    def test_an_explicit_argument_beats_both(self, spec, tmp_path, monkeypatch):
        """The parameter is the most local statement of intent, so it wins."""
        set_format_subfolders(False)
        monkeypatch.setenv("MLFRAME_PLOT_FORMAT_SUBFOLDERS", "0")
        base = str(tmp_path / "plot")
        render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"), base, interactive=False, format_subfolders=True)
        assert (tmp_path / "png" / "plot.png").exists()


class TestTheReportBuilderStillFindsTheImages:
    """The combined HTML report LOOKS UP files it did not write, so it has to know both layouts."""

    @pytest.mark.parametrize("subfolders", [True, False])
    def test_a_chart_is_embedded_under_either_layout(self, spec, tmp_path, subfolders):
        """Reconstructing only one layout drops the image from the report instead of failing."""
        from mlframe.reporting._diagnostics_dispatch_extra import build_combined_html_report

        base = str(tmp_path / "chart")
        render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"), base, interactive=False, format_subfolders=subfolders)
        out = build_combined_html_report(
            base_path=str(tmp_path / "run"), chart_paths=[base], plot_outputs="matplotlib[png]",
        )
        assert out is not None and os.path.exists(out)
        with open(out, encoding="utf-8") as fh:
            html = fh.read()
        # Assert the IMAGE, not the label: when the lookup misses, the report is still written and still names the
        # chart -- it just has no picture in it. That is the failure mode this test exists to catch (measured: a
        # 2.8 KB report against 12 KB with the image embedded).
        assert "<img" in html and "base64" in html
