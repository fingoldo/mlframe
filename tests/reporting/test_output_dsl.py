"""Tests for the plot-output DSL parser (``plot_outputs`` config field)."""

from __future__ import annotations

import pytest

from mlframe.reporting.output import (
    BACKEND_FORMATS, PlotOutputSpec, parse_plot_output_dsl,
)


class TestParseHappyPath:
    def test_single_backend_single_format(self):
        out = parse_plot_output_dsl("plotly[html]")
        assert isinstance(out, PlotOutputSpec)
        assert out.backends == (("plotly", frozenset({"html"})),)

    def test_single_backend_multi_format(self):
        out = parse_plot_output_dsl("plotly[html,png]")
        assert out.backends == (("plotly", frozenset({"html", "png"})),)

    def test_multi_backend(self):
        out = parse_plot_output_dsl("plotly[html] + matplotlib[pdf]")
        assert out.backends == (
            ("plotly", frozenset({"html"})),
            ("matplotlib", frozenset({"pdf"})),
        )

    def test_matplotlib_only_back_compat(self):
        out = parse_plot_output_dsl("matplotlib[png]")
        assert out.backends == (("matplotlib", frozenset({"png"})),)

    def test_whitespace_tolerance(self):
        # Extra spaces inside / around clauses + format list.
        out = parse_plot_output_dsl("  plotly [ html , png ]  +  matplotlib[ pdf ] ")
        assert out.backends == (
            ("plotly", frozenset({"html", "png"})),
            ("matplotlib", frozenset({"pdf"})),
        )

    def test_case_insensitive_backend(self):
        out = parse_plot_output_dsl("PLOTLY[html]")
        assert out.backends[0][0] == "plotly"

    def test_case_insensitive_format(self):
        out = parse_plot_output_dsl("plotly[HTML]")
        assert out.backends[0][1] == frozenset({"html"})


class TestParseValidationErrors:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_plot_output_dsl("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_plot_output_dsl("   ")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            parse_plot_output_dsl("bokeh[png]")

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="does not support"):
            parse_plot_output_dsl("plotly[gif]")

    def test_matplotlib_html_incompat_raises(self):
        # matplotlib can't write interactive html.
        with pytest.raises(ValueError, match="does not support"):
            parse_plot_output_dsl("matplotlib[html]")

    def test_plotly_jpg_incompat_raises(self):
        # plotly's write_image doesn't support jpg per documented allowlist.
        with pytest.raises(ValueError, match="does not support"):
            parse_plot_output_dsl("plotly[jpg]")

    def test_duplicate_backend_raises(self):
        with pytest.raises(ValueError, match="more than once"):
            parse_plot_output_dsl("plotly[html] + plotly[png]")

    def test_duplicate_format_raises(self):
        with pytest.raises(ValueError, match="duplicate format"):
            parse_plot_output_dsl("plotly[html, html]")

    def test_malformed_clause_raises(self):
        with pytest.raises(ValueError, match="malformed"):
            parse_plot_output_dsl("garbage")

    def test_empty_format_list_raises(self):
        with pytest.raises(ValueError, match="malformed|no formats"):
            parse_plot_output_dsl("plotly[]")


class TestBackendFormatsAllowlist:
    def test_matplotlib_allowed_formats(self):
        assert BACKEND_FORMATS["matplotlib"] == frozenset(
            {"png", "pdf", "svg", "jpg", "jpeg"}
        )

    def test_plotly_allowed_formats(self):
        assert BACKEND_FORMATS["plotly"] == frozenset(
            {"html", "png", "svg", "pdf", "json"}
        )
