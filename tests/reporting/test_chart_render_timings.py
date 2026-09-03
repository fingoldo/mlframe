"""Per-chart-type render cost, so a slow report can be attributed instead of just observed.

A production run spent longer on post-fit diagnostics than on the model fit, with the only visible number being
the enclosing phase's total -- enough to know the report was slow, not enough to know which chart to cap.
"""

from __future__ import annotations

import threading

import pytest

from mlframe.reporting.renderers._render_timings import (
    chart_timings_snapshot,
    chart_type_of,
    format_chart_timings,
    record_chart_render,
    reset_chart_timings,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is process-wide, so every test starts and leaves it empty."""
    reset_chart_timings()
    yield
    reset_chart_timings()


class TestTheKey:
    """What counts as one chart type."""

    @pytest.mark.parametrize(
        "path, expected",
        [
            ("reports/pdp_ice_price", "pdp_ice_price"),
            ("reports/shap_beeswarm.matplotlib", "shap_beeswarm"),
            ("reports/decile_table_3", "decile_table"),
            ("reports/roc_curve_a1b2c3d4", "roc_curve"),
            ("", "unnamed"),
        ],
    )
    def test_stems_collapse_to_a_type(self, path, expected):
        """The backend tag and any numeric/hex disambiguator are not part of the chart type."""
        assert chart_type_of(path) == expected

    def test_two_targets_stay_two_rows(self):
        """Collapsing them would need a guess at which part of the stem is the target; a wrong guess merges charts."""
        assert chart_type_of("calibration_y_dummy") != chart_type_of("calibration_y_price")


class TestAccumulation:
    """The numbers a reader acts on."""

    def test_repeat_renders_sum_and_count(self):
        """Total, count, average and slowest are what make a row actionable."""
        for s in (1.0, 3.0):
            record_chart_render("pdp_ice", s, backend="matplotlib")
        row = chart_timings_snapshot()[0]
        assert row["count"] == 2
        assert row["seconds"] == pytest.approx(4.0)
        assert row["avg_seconds"] == pytest.approx(2.0)
        assert row["max_seconds"] == pytest.approx(3.0)

    def test_backends_are_separate_rows(self):
        """They render concurrently, so charging both to one row would double-count the wall time."""
        record_chart_render("roc", 1.0, backend="matplotlib")
        record_chart_render("roc", 2.0, backend="plotly")
        assert len(chart_timings_snapshot()) == 2

    def test_rows_are_sorted_by_total_cost(self):
        """Total time is what a reader can act on, so it decides the order."""
        record_chart_render("cheap", 0.1)
        record_chart_render("dear", 9.0)
        assert [r["chart"] for r in chart_timings_snapshot()] == ["dear", "cheap"]

    def test_reset_clears_everything(self):
        """A stale row would attribute the previous suite's cost to this one."""
        record_chart_render("x", 1.0)
        reset_chart_timings()
        assert chart_timings_snapshot() == []

    def test_concurrent_records_are_not_lost(self):
        """Renders run on a thread pool, so the accumulator has to be safe under real concurrency."""
        def _worker():
            """One thread's share of the concurrent writes."""
            for _ in range(200):
                record_chart_render("hot", 0.001)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert chart_timings_snapshot()[0]["count"] == 800


class TestTheTable:
    """It has to be readable in a log without opening metadata."""

    def test_empty_says_so_rather_than_printing_a_header(self):
        """An empty table with a header reads as "everything was free"."""
        assert "no figures" in format_chart_timings(chart_timings_snapshot())

    def test_it_names_the_charts_and_their_share(self):
        """Share of total is what identifies the chart worth capping."""
        record_chart_render("pdp_ice", 8.0)
        record_chart_render("roc", 2.0)
        text = format_chart_timings(chart_timings_snapshot())
        assert "pdp_ice" in text and "80.0%" in text
        assert "10.00s total" in text

    def test_long_tails_are_truncated_but_accounted_for(self):
        """The truncated tail still has to report its own cost, or the shares stop adding up."""
        for i in range(30):
            record_chart_render(f"chart_{i:02d}", float(30 - i))
        text = format_chart_timings(chart_timings_snapshot(), top=5)
        assert "and 25 more chart type(s)" in text


class TestTheRenderHook:
    """The registry is only useful if the real save path feeds it."""

    def test_a_real_render_records_its_own_cost(self, tmp_path):
        """Behavioural end of the wiring: render one figure and find it in the table, named after its file."""
        from mlframe.reporting.output import parse_plot_output_dsl
        from mlframe.reporting.renderers.save import render_and_save
        from mlframe.reporting.spec import AnnotationPanelSpec, FigureSpec

        spec = FigureSpec(suptitle="", panels=((AnnotationPanelSpec(text="hello", title="t"),),), figsize=(4.0, 2.0))
        render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"), str(tmp_path / "roc_curve_y"), interactive=False)
        rows = chart_timings_snapshot()
        assert [r["chart"] for r in rows] == ["roc_curve_y [matplotlib]"]
        assert rows[0]["seconds"] > 0.0
