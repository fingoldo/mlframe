"""Every text layer breaks at the edge of the canvas, not at a character count.

The four places a chart puts prose -- figure title, panel title, bottom caption, free-text annotation -- each
wrapped by a fixed chars-per-line budget calibrated once, at one width, for one font size, and then applied at
every width and size. A wide figure therefore folded its headline into a narrow ragged column with a third of the
page unused, and the same budget simultaneously overflowed a panel of CamelCase identifiers, whose glyphs are far
wider than the average the constant assumed.

The contract these tests pin is a single sentence: give the same text more room and it must use it. That is
checked per layer and on both backends, because a spec that reads well as a PNG and badly as an HTML page is the
same defect wearing one of two faces.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mlframe.reporting.renderers._shared_helpers import wrap_annotation_text, wrap_text_to_width
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, LinePanelSpec, ScatterPanelSpec

# A real diagnostic headline: model identity, date span, row count, then a metric block. Deliberately the kind of
# text the character budget was worst at -- digits, percent signs, equals signs and CamelCase, none of which are
# the "average" character a constant assumes.
HEADLINE = (
    "TEST 2025-02-13/2025-09-01 XGBClassifier simple-12-5m-bars newhope2 MAX_LONG_PROFIT-12-5m_above0.5% BT=8% "
    "trained on 51.6M rows 2020-01-06/2024-08-13 @iter=278 [1_464F/9.3M rows] "
    "ICE=-0.45, BR=7.17%, CMAEW=2.74%+-0.95%, LL=0.24, ROC AUC=0.86, PR AUC=0.39, @0.50: [PR=57.87%,RE=11.10%,F1=0.19]"
)


def _scatter(title=""):
    """A minimal non-squared scatter: squaring would make the panel width height-limited, which is a layout property rather than a wrapping one."""
    return ScatterPanelSpec(x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]), title=title, equal_aspect=False)


def _mpl_lines(fig, getter):
    """Line count of a rendered text object, after a draw so the layout engine has run."""
    fig.canvas.draw()
    return len(getter(fig).splitlines())


class TestTheFigureTitleUsesTheFigureWidth:
    """The headline is figure-level, so its budget is the figure."""

    def test_a_wider_figure_needs_fewer_lines(self):
        """The single sentence this file exists to pin."""
        narrow = _mpl_lines(MatplotlibRenderer().render(FigureSpec(suptitle=HEADLINE, figsize=(8.0, 5.0), panels=((_scatter(),),))), lambda f: f._suptitle.get_text())
        wide = _mpl_lines(MatplotlibRenderer().render(FigureSpec(suptitle=HEADLINE, figsize=(20.0, 5.0), panels=((_scatter(),),))), lambda f: f._suptitle.get_text())
        assert wide < narrow, f"same title took {wide} lines at 20in and {narrow} at 8in"

    def test_it_actually_fills_the_width(self):
        """Fewer lines is necessary but not sufficient: the longest line has to reach most of the way across."""
        fig = MatplotlibRenderer().render(FigureSpec(suptitle=HEADLINE, figsize=(12.0, 5.0), panels=((_scatter(),),)))
        fig.canvas.draw()
        used = fig._suptitle.get_window_extent(fig.canvas.get_renderer()).width
        assert used / (12.0 * fig.dpi) > 0.80, f"headline used only {100 * used / (12.0 * fig.dpi):.0f}% of the figure width"

    def test_it_does_not_overflow_the_figure(self):
        """Filling the width must not mean running off it."""
        fig = MatplotlibRenderer().render(FigureSpec(suptitle=HEADLINE, figsize=(12.0, 5.0), panels=((_scatter(),),)))
        fig.canvas.draw()
        bb = fig._suptitle.get_window_extent(fig.canvas.get_renderer())
        assert bb.x0 >= -1.0 and bb.x1 <= 12.0 * fig.dpi + 1.0

    def test_it_does_not_land_on_the_axes(self):
        """A multi-line headline used to start BELOW the top of the axes and print across its own chart."""
        fig = MatplotlibRenderer().render(FigureSpec(suptitle=HEADLINE, figsize=(12.0, 6.0), panels=((_scatter(),),)))
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        assert fig._suptitle.get_window_extent(r).y0 >= max(a.get_window_extent(r).y1 for a in fig.axes)

    def test_the_plotly_twin_responds_too(self):
        """One spec, two backends, one behaviour."""
        narrow = PlotlyRenderer().render(FigureSpec(suptitle=HEADLINE, figsize=(8.0, 5.0), panels=((_scatter(),),))).layout.title.text
        wide = PlotlyRenderer().render(FigureSpec(suptitle=HEADLINE, figsize=(20.0, 5.0), panels=((_scatter(),),))).layout.title.text
        assert wide.count("<br>") < narrow.count("<br>")


class TestThePanelTitleUsesThePanelWidth:
    """Most charts put their identity in the panel title, not the figure title, so this is the common path."""

    @pytest.mark.parametrize("panel_cls", ["scatter", "line"])
    def test_a_wider_panel_needs_fewer_lines(self, panel_cls):
        """Checked on two panel types because the title is set by one shared helper and must stay shared."""

        def _spec(width):
            """One panel of the requested kind carrying the headline as its title."""
            panel = _scatter(HEADLINE) if panel_cls == "scatter" else LinePanelSpec(x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]), title=HEADLINE)
            return FigureSpec(figsize=(width, 4.0), panels=((panel,),))

        narrow = _mpl_lines(MatplotlibRenderer().render(_spec(6.0)), lambda f: f.axes[0].get_title())
        wide = _mpl_lines(MatplotlibRenderer().render(_spec(20.0)), lambda f: f.axes[0].get_title())
        assert wide < narrow, f"{panel_cls} panel title took {wide} lines at 20in and {narrow} at 6in"

    def test_the_plotly_twin_responds_too(self):
        """plotly stamps subplot titles as annotations; they were wrapped by the same constant."""
        narrow = PlotlyRenderer().render(FigureSpec(figsize=(6.0, 4.0), panels=((_scatter(HEADLINE),),))).layout.annotations[0].text
        wide = PlotlyRenderer().render(FigureSpec(figsize=(20.0, 4.0), panels=((_scatter(HEADLINE),),))).layout.annotations[0].text
        assert wide.count("<br>") < narrow.count("<br>")

    def test_the_same_character_count_wraps_differently_by_glyph_width(self):
        """The assertion that separates measuring from counting.

        Panel titles were already wrapped to a budget that SCALED with panel width, so "a wider panel needs
        fewer lines" passes on the old code too. What a character budget cannot do is tell ``WWWW`` from
        ``iiii``: same count, several times the ink. Equal-length titles must not wrap identically.
        """
        wide_glyphs = " ".join(["WWWWWWWW"] * 14)
        narrow_glyphs = " ".join(["iiiiiiii"] * 14)
        assert len(wide_glyphs) == len(narrow_glyphs)
        n_wide = _mpl_lines(MatplotlibRenderer().render(FigureSpec(figsize=(9.0, 4.0), panels=((_scatter(wide_glyphs),),))), lambda f: f.axes[0].get_title())
        n_narrow = _mpl_lines(MatplotlibRenderer().render(FigureSpec(figsize=(9.0, 4.0), panels=((_scatter(narrow_glyphs),),))), lambda f: f.axes[0].get_title())
        assert n_wide > n_narrow, f"equal-length titles wrapped the same ({n_wide} vs {n_narrow}); the budget is still counting characters"

    def test_an_explicit_line_break_survives(self):
        """Callers build titles with deliberate structure; re-flowing it away is the defect this replaced."""
        fig = MatplotlibRenderer().render(FigureSpec(figsize=(20.0, 4.0), panels=((_scatter("first line\nsecond line"),),)))
        assert _mpl_lines(fig, lambda f: f.axes[0].get_title()) == 2


class TestTheCaptionUsesTheFigureWidth:
    """The how-to-read footnote sits in its own band and had its own separate constant."""

    CAPTION = (
        "Below the diagonal is OVER-confidence, above it under-confidence. Bubble area is the bin's row count: a bin "
        "holding a handful of rows swings on noise, and most of the visual span of a calibration curve is usually "
        "carried by very few rows. Hollow markers with a dotted interval hold too few rows to read."
    )

    def _caption_text(self, fig):
        """The caption is the one figure-level text that is not the suptitle."""
        return next(t for t in fig.texts if t is not fig._suptitle).get_text()

    def test_a_wider_figure_needs_fewer_lines(self):
        """Same contract as the headline, on the layer beneath it."""
        narrow = _mpl_lines(MatplotlibRenderer().render(FigureSpec(caption=self.CAPTION, figsize=(6.0, 5.0), panels=((_scatter(),),))), self._caption_text)
        wide = _mpl_lines(MatplotlibRenderer().render(FigureSpec(caption=self.CAPTION, figsize=(20.0, 5.0), panels=((_scatter(),),))), self._caption_text)
        assert wide < narrow

    def test_it_does_not_overflow_the_figure(self):
        """A caption that runs off the page is worse than one that wraps early."""
        fig = MatplotlibRenderer().render(FigureSpec(caption=self.CAPTION, figsize=(9.0, 5.0), panels=((_scatter(),),)))
        fig.canvas.draw()
        bb = next(t for t in fig.texts if t is not fig._suptitle).get_window_extent(fig.canvas.get_renderer())
        assert bb.x0 >= -1.0 and bb.x1 <= 9.0 * fig.dpi + 1.0


class TestFreeTextAnnotationsAreMeasuredToo:
    """The annotation panel had its own budget built on an assumed 0.6 em average advance."""

    LONG_TOKEN = "DummyClassifier(strategy=prior)_baseline_reference_run_2025_06_30_full_matrix_no_spaces_anywhere"

    def test_a_wider_panel_needs_fewer_lines(self):
        """Same contract again."""
        narrow = wrap_annotation_text("metric unavailable for this split " * 6, 3.0, 10)
        wide = wrap_annotation_text("metric unavailable for this split " * 6, 12.0, 10)
        assert len(wide.splitlines()) < len(narrow.splitlines())

    def test_an_unbreakable_token_is_still_broken(self):
        """A generated name or a file path carries no spaces and used to run straight out of the panel."""
        out = wrap_annotation_text(self.LONG_TOKEN, 3.0, 10)
        assert len(out.splitlines()) > 1
        assert "".join(out.split("\n")) == self.LONG_TOKEN, "breaking a token must not lose or add characters"

    def test_the_same_character_count_wraps_differently_by_glyph_width(self):
        """The annotation budget assumed every glyph is ~0.6 em; these two are nowhere near each other."""
        wide_glyphs = " ".join(["WWWWWWWW"] * 10)
        narrow_glyphs = " ".join(["iiiiiiii"] * 10)
        assert len(wide_glyphs) == len(narrow_glyphs)
        n_wide = len(wrap_annotation_text(wide_glyphs, 5.0, 10).splitlines())
        n_narrow = len(wrap_annotation_text(narrow_glyphs, 5.0, 10).splitlines())
        assert n_wide > n_narrow, f"equal-length text wrapped the same ({n_wide} vs {n_narrow}); the budget is still counting characters"

    def test_a_larger_font_needs_more_lines(self):
        """Font size is the half of the assumption a width-scaled character budget still got wrong."""
        small = wrap_annotation_text("metric unavailable for this split " * 4, 6.0, 6)
        large = wrap_annotation_text("metric unavailable for this split " * 4, 6.0, 16)
        assert len(large.splitlines()) > len(small.splitlines())


class TestTheMeasurementItself:
    """The primitive under all four layers."""

    def test_wider_glyphs_wrap_sooner_than_narrow_ones(self):
        """The whole point of measuring: 'W' and 'i' are not the same character."""
        wide_glyphs = wrap_text_to_width(" ".join(["WWWW"] * 30), fontsize=10, width_in=6.0)
        narrow_glyphs = wrap_text_to_width(" ".join(["iiii"] * 30), fontsize=10, width_in=6.0)
        assert len(wide_glyphs) > len(narrow_glyphs)

    def test_a_measurement_failure_still_returns_the_text(self):
        """Text layout must never raise; a bad font or a stripped matplotlib falls back to the character budget."""
        assert wrap_text_to_width(HEADLINE, fontsize=float("nan"), width_in=float("nan"), fallback_chars=60)

    def test_the_font_is_touched_once_per_size_not_once_per_line(self):
        """The regression that hung the renderer, pinned by MECHANISM rather than by a stopwatch.

        The first implementation measured each candidate LINE with ``TextPath``, which rasterises glyph
        outlines at roughly 8 ms a call. Greedily wrapping one 40-word headline therefore cost 4.2 SECONDS, and
        a suite rendering hundreds of charts hit the renderer's 60 s per-figure timeout -- the failure surfaced
        as a hung chart, not as a slow one. Counting font builds is stable under load in a way a timing
        assertion is not (see ``test_no_single_shot_timing_assertion``).
        """
        from mlframe.reporting.renderers import _shared_helpers as sh

        real_builder = sh._char_advances
        builds = {"n": 0}

        def _counting(fontsize):
            """Count the calls that MISS the cache, i.e. the ones that actually touch the font."""
            if fontsize not in sh._CHAR_ADVANCE_CACHE:
                builds["n"] += 1
            return real_builder(fontsize)

        sh._CHAR_ADVANCE_CACHE.clear()
        sh._char_advances = _counting
        try:
            for i in range(40):
                wrap_text_to_width(HEADLINE.replace("newhope2", f"run{i}"), fontsize=11, width_in=12.0)
        finally:
            sh._char_advances = real_builder
            sh._CHAR_ADVANCE_CACHE.clear()
        assert builds["n"] == 1, f"the font was touched {builds['n']} times to wrap 40 titles at one size"

    def test_no_words_are_lost_or_invented(self):
        """Wrapping is a layout operation; the words must come out exactly as they went in."""
        assert " ".join(wrap_text_to_width(HEADLINE, fontsize=10, width_in=9.0)) == HEADLINE
