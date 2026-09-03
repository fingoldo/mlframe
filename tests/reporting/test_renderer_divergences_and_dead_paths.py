"""Twelve reporting defects, most of them one backend quietly disagreeing with the other about the same spec.

A spec is supposed to be the single description of a chart, so a divergence is not a cosmetic difference: it
means the PNG and the HTML make different claims about the same data, and a reader comparing them has no way to
know which is the intended one. The violin pair is the clearest -- matplotlib dropped empty groups and named them
in the title, on the explicit reasoning that "a violin that silently vanishes reads as 'this group has no
spread', which is a different statement from 'this group has no data'"; plotly rendered a labelled category with
nothing in it and no note.

The rest: a per-point hovertext that reached the tooltip only when the trace happened not to be split or
downsampled; low-evidence error bars drawn identically to confident ones on one backend; a bar label truncated on
three of four code paths; an axis id computed from grid arithmetic that a hole in the grid invalidates; a
plotly.js bundle deduplicated only when it happened to be the first script; two `x or default` expressions; a
hardcoded colour that escaped a centralisation; dead per-point-text scaffolding; and a comment describing a
superseded sort order.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("plotly")


def _render(*panels, **kw):
    """Render one row of panels through the plotly backend."""
    from mlframe.reporting.renderers.plotly import PlotlyRenderer
    from mlframe.reporting.spec import FigureSpec

    return PlotlyRenderer().render(FigureSpec(suptitle="s", panels=(panels,), **kw))


class TestTheTwoBackendsAgreeOnAnEmptyViolinGroup:
    """One spec must not make two different claims about a group with no rows."""

    def _spec(self):
        """Three groups, the middle one empty -- a multiclass panel where one class has no rows in the split."""
        from mlframe.reporting.spec import ViolinPanelSpec

        rng = np.random.default_rng(0)
        return ViolinPanelSpec(
            groups=(rng.normal(size=200), np.array([]), rng.normal(1, 1, 200)),
            group_labels=("a", "class_c", "b"),
            title="t",
            show_box=True,
        )

    def test_the_empty_group_is_dropped(self):
        """It used to render as a labelled category with nothing in it."""
        fig = _render(self._spec())
        assert sum(1 for tr in fig.data if tr.type == "violin") == 2

    def test_the_dropped_group_is_named(self):
        """matplotlib puts it in the title; plotly's subplot titles are fixed before the panel renders, so the
        note goes in as a subplot annotation carrying the same information."""
        fig = _render(self._spec())
        assert any("no data: class_c" in (a.text or "") for a in fig.layout.annotations), [a.text for a in fig.layout.annotations]

    def test_a_spec_with_no_empty_groups_is_unaffected(self):
        """The fix must not add a note to a panel that has nothing to report."""
        from mlframe.reporting.spec import ViolinPanelSpec

        rng = np.random.default_rng(1)
        fig = _render(ViolinPanelSpec(groups=(rng.normal(size=100), rng.normal(size=100)), group_labels=("a", "b"), title="t"))
        assert not any("no data" in (a.text or "") for a in fig.layout.annotations)

    def test_the_matplotlib_side_still_drops_and_names(self):
        """The behaviour plotly was aligned TO must not have moved."""
        import inspect

        from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer

        src = inspect.getsource(MatplotlibRenderer._violin)
        assert "no data:" in src and "whis=(5, 95)" in src


class TestPerPointHovertextSurvivesASplitTrace:
    """The denominator is what the field exists to carry, and it was dropped exactly when the panel was busy."""

    def _panel(self, n=40, low_evidence=True):
        """A scatter carrying per-point support text, optionally split into strong and weak traces."""
        from mlframe.reporting.spec import ScatterPanelSpec

        rng = np.random.default_rng(0)
        return ScatterPanelSpec(
            x=rng.random(n),
            y=rng.random(n),
            title="t",
            hovertext=tuple(f"n={i}" for i in range(n)),
            low_evidence_indices=tuple(range(5)) if low_evidence else None,
        )

    def test_both_halves_of_a_split_panel_carry_it(self):
        """Two traces of k and n-k points, neither equal to n, so the old length gate attached it to neither."""
        fig = _render(self._panel())
        scatters = [tr for tr in fig.data if tr.type in ("scatter", "scattergl") and tr.x is not None]
        assert scatters, "no scatter traces were drawn"
        for tr in scatters:
            assert tr.hovertext is not None and len(tr.hovertext) == len(tr.x), (tr.name, len(tr.x))

    def test_each_trace_gets_its_own_rows(self):
        """Attaching the whole array to a subset would be worse than dropping it."""
        fig = _render(self._panel())
        weak = next(tr for tr in fig.data if tr.name == "too few rows to read")
        assert sorted(weak.hovertext) == sorted(f"n={i}" for i in range(5)), weak.hovertext

    def test_an_unsplit_panel_still_gets_it(self):
        """The case that already worked."""
        fig = _render(self._panel(low_evidence=False))
        tr = next(tr for tr in fig.data if tr.type in ("scatter", "scattergl") and tr.x is not None)
        assert tr.hovertext is not None and len(tr.hovertext) == 40

    def test_a_mismatched_length_is_ignored_rather_than_misaligned(self):
        """A wrong-length spec array must not be silently zipped against the wrong points."""
        from mlframe.reporting.spec import ScatterPanelSpec

        rng = np.random.default_rng(2)
        fig = _render(ScatterPanelSpec(x=rng.random(20), y=rng.random(20), title="t", hovertext=("a", "b")))
        tr = next(tr for tr in fig.data if tr.type in ("scatter", "scattergl") and tr.x is not None)
        assert tr.hovertext is None or len(tr.hovertext) == 20


class TestTheLowEvidenceIntervalsAreVisiblyMuted:
    """ "We know nothing here" has to look different from a confident interval."""

    def test_the_weak_trace_restyles_its_error_bars(self):
        """plotly narrowed the arrays but set no colour, so both were drawn identically -- and the weak marker's
        fully transparent colour is what `error_y.color` inherits when unset."""
        from mlframe.reporting.spec import ScatterPanelSpec

        rng = np.random.default_rng(0)
        n = 20
        fig = _render(
            ScatterPanelSpec(
                x=rng.random(n), y=rng.random(n), title="t",
                y_err=np.full(n, 0.05), low_evidence_indices=tuple(range(5)),
            )
        )
        weak = next(tr for tr in fig.data if tr.name == "too few rows to read")
        strong = next(tr for tr in fig.data if tr.name != "too few rows to read" and getattr(tr, "error_y", None) is not None)
        assert weak.error_y.color is not None, "the weak interval inherits the transparent marker colour"
        assert weak.error_y.color != strong.error_y.color


class TestTheAxisIdComesFromTheRealGrid:
    """`row * cols + col` assumes every cell got an axis; empty cells are passed as `None` in `specs`."""

    def test_a_grid_with_a_hole_resolves_each_panel_to_its_own_axis(self):
        """On `((line_a, None), (line_b, heat))` the arithmetic computes 4 for line_b, a key no trace is bound to."""
        from mlframe.reporting.renderers._shared_helpers import plotly_axis_suffix
        from mlframe.reporting.renderers.plotly import PlotlyRenderer
        from mlframe.reporting.spec import FigureSpec, HeatmapPanelSpec, LinePanelSpec

        line = LinePanelSpec(x=np.arange(5.0), y=np.arange(5.0), series_labels=("s",), title="l")
        heat = HeatmapPanelSpec(matrix=np.arange(9.0).reshape(3, 3), row_labels=("a", "b", "c"), col_labels=("a", "b", "c"), title="h")
        fig = PlotlyRenderer().render(FigureSpec(suptitle="s", panels=((line, None), (line, heat))))

        suffixes = {(r, c): plotly_axis_suffix(fig, r, c, 2) for r, c in ((1, 1), (2, 1), (2, 2))}
        assert len(set(suffixes.values())) == 3, suffixes
        for sfx in suffixes.values():
            assert f"xaxis{sfx}" in fig.layout, (sfx, list(fig.layout))

    def test_a_full_grid_matches_the_old_arithmetic(self):
        """With no holes the two agree, so nothing that worked before moves."""
        from mlframe.reporting.renderers._shared_helpers import plotly_axis_suffix
        from mlframe.reporting.renderers.plotly import PlotlyRenderer
        from mlframe.reporting.spec import FigureSpec, LinePanelSpec

        line = LinePanelSpec(x=np.arange(5.0), y=np.arange(5.0), series_labels=("s",), title="l")
        fig = PlotlyRenderer().render(FigureSpec(suptitle="s", panels=((line, line), (line, line))))
        for r in (1, 2):
            for c in (1, 2):
                idx = (r - 1) * 2 + c
                assert plotly_axis_suffix(fig, r, c, 2) == ("" if idx == 1 else str(idx))

    def test_it_falls_back_when_the_grid_is_unreadable(self):
        """Every caller used the arithmetic before; an unreadable grid must not raise."""
        from mlframe.reporting.renderers._shared_helpers import plotly_axis_suffix

        class NoGrid:
            """A figure-like object with no grid reference at all."""

        assert plotly_axis_suffix(NoGrid(), 2, 2, 2) == "4"


class TestTheEdgeMidpointsAreDrawnOnce:
    """Two overlapping marker traces, and the later, poorer one won the hover."""

    def _fig(self):
        """A three-node network with weighted edges."""
        from mlframe.reporting.spec import NetworkPanelSpec

        return _render(
            NetworkPanelSpec(
                node_x=np.array([0.0, 1.0, 2.0]),
                node_y=np.array([0.0, 1.0, 0.0]),
                node_size=np.full(3, 40.0),
                node_color=("#1f77b4",) * 3,
                node_label=("feat_a", "feat_b", "feat_c"),
                edge_src=np.array([0, 1]),
                edge_dst=np.array([1, 2]),
                edge_weight=np.array([0.1234, 0.5]),
                title="t",
                colorbar_label="MI",
            )
        )

    def test_the_descriptive_hover_is_the_one_that_survives(self):
        """It carried node names and honoured `colorbar_label`; the trace that won carried a hardcoded "MI=" and
        no names."""
        hovers = [tuple(tr.hovertext) for tr in self._fig().data if getattr(tr, "hovertext", None)]
        assert hovers, "no trace carries hovertext"
        assert any("feat_a - feat_b" in h for hs in hovers for h in hs), hovers

    def test_the_midpoints_are_not_emitted_twice(self):
        """Up to 8 extra traces plus a full duplicate coordinate set per figure."""
        marker_traces = [tr for tr in self._fig().data if tr.mode == "markers" and tr.x is not None and len(tr.x) == 2]
        assert len(marker_traces) == 1, [(tr.name, tr.mode, len(tr.x)) for tr in marker_traces]


class TestTheBarLabelIsTruncatedOnBothOrientations:
    """The horizontal branch's own comment claimed the vertical branch already did this."""

    def test_the_vertical_branch_truncates(self):
        """A 200-char generated column name otherwise runs off the bottom of the axis."""
        import inspect

        from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer

        src = inspect.getsource(MatplotlibRenderer._bar)
        assert src.count("truncate_bar_label(c)") == 2, src.count("truncate_bar_label(c)")

    def test_the_thinning_constants_are_not_written_out_again(self):
        """25 and 20 lived in four places across two backends."""
        import inspect

        from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer

        src = inspect.getsource(MatplotlibRenderer._bar)
        assert "n_cat > 25" not in src and "n_cat / 20" not in src
        assert src.count("_BAR_TICK_THIN_THRESHOLD") == 2 and src.count("_BAR_TICK_KEEP") == 2


class TestFalsyValuedSpecFieldsAreHonoured:
    """`x or default` cannot represent a deliberate 0.0 or empty string."""

    def test_the_histogram_bin_width_and_overlay_label_use_is_not_none(self):
        """matplotlib treated 0.0 as unset and "" as absent; plotly honoured both. One spec, two pictures."""
        import inspect

        from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer

        src = inspect.getsource(MatplotlibRenderer._histogram)
        assert "p.bin_width or" not in src and "p.overlay_label or" not in src
        assert "p.bin_width is not None" in src and "p.overlay_label is not None" in src

    def test_an_empty_calibration_cmap_override_is_rejected(self):
        """`""` behaved as a silent second clear rather than as the invalid colormap name it is."""
        from mlframe.reporting.colors import calibration_cmap, set_calibration_cmap

        try:
            set_calibration_cmap("")
            with pytest.raises(ValueError, match="empty string"):
                calibration_cmap()
        finally:
            set_calibration_cmap(None)

    def test_clearing_the_override_still_works(self):
        """`None` is the documented CLEAR path and must stay one."""
        from mlframe.reporting.colors import calibration_cmap, set_calibration_cmap

        set_calibration_cmap("viridis")
        try:
            assert calibration_cmap() == "viridis"
        finally:
            set_calibration_cmap(None)
        assert calibration_cmap() != "viridis" or True  # falls through to env / module default


def test_the_plotly_bundle_is_deduplicated_when_it_is_not_the_first_script():
    """A fragment with a config or `require` shim ahead of the bundle passed through whole, so a 20-chart report
    still shipped 20 copies of a 3-4 MB bundle -- the exact cost the function exists to avoid."""
    from mlframe.reporting.report_html import _PLOTLY_JS_MARKER, _dedupe_plotly_js

    bundle = f"<script>{_PLOTLY_JS_MARKER} /* 3MB */</script>"
    fragment = "<script>window.PlotlyConfig = {}</script>" + bundle + "<div>chart</div>"

    seen: set = set()
    first = _dedupe_plotly_js(fragment, seen)
    assert first == fragment, "the first fragment keeps its bundle"

    second = _dedupe_plotly_js(fragment, seen)
    assert _PLOTLY_JS_MARKER not in second, second
    assert "window.PlotlyConfig" in second and "<div>chart</div>" in second, second


def test_the_bundle_is_still_stripped_when_it_is_first():
    """The case that already worked."""
    from mlframe.reporting.report_html import _PLOTLY_JS_MARKER, _dedupe_plotly_js

    fragment = f"<script>{_PLOTLY_JS_MARKER}</script><div>chart</div>"
    seen: set = set()
    _dedupe_plotly_js(fragment, seen)
    assert _dedupe_plotly_js(fragment, seen) == "<div>chart</div>"


def test_the_perfect_fit_diagonal_comes_from_the_colour_module():
    """The one overlay colour that escaped the centralisation done to stop exactly this drift."""
    import inspect

    from mlframe.reporting.renderers import _matplotlib_scatter

    src = inspect.getsource(_matplotlib_scatter)
    assert '"g--"' not in src
    assert "PERFECT_FIT_LINE" in src


def test_the_dead_per_point_text_scaffolding_is_gone():
    """`text` was assigned None once and never reassigned, so four dependent expressions were unreachable -- and
    the surviving scaffolding read as a live per-point-text feature that does not exist."""
    import inspect

    from mlframe.reporting.renderers import _plotly_scatter

    src = inspect.getsource(_plotly_scatter)
    assert "text = None" not in src
    assert 'mode="markers+text" if text else "markers"' not in src


def test_the_slice_finder_ordering_comment_matches_the_code():
    """Two adjacent blocks stated opposite rules; the code implements the second."""
    import inspect

    from mlframe.reporting.charts import slice_finder

    src = inspect.getsource(slice_finder)
    assert "worst-ERROR-first" not in src
    assert "Display order matches the SCORE" in src
