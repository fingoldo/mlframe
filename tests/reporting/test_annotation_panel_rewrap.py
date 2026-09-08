"""The free-text panel wrapped against a rectangle it had not been given yet.

Constrained layout does not settle a cell's geometry until draw time, so the wrap computed while the panel
is being built is measured against the pre-layout rectangle. That is conservative rather than broken -- the
panel ends up WIDER, not narrower -- but on a figure whose siblings hand width back it left about a fifth
of the cell empty and the text folded onto more lines than it needed.

Rewrapping on the first draw recovers it, and must do so exactly once: the rewrap mutates the artist, which
schedules another draw, which would rewrap again.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.spec import AnnotationPanelSpec, FigureSpec, HeatmapPanelSpec

LONG_TEXT = " ".join(f"word{i}" for i in range(70))


def _figure_with_a_width_giving_sibling():
    """A heatmap-plus-colorbar beside the text panel: the layout hands the text cell width back at draw."""
    annotation = AnnotationPanelSpec(text=LONG_TEXT, title="notes", fontsize=9)
    heatmap = HeatmapPanelSpec(
        matrix=np.random.default_rng(0).random((6, 6)),
        row_labels=tuple("abcdef"),
        col_labels=tuple("abcdef"),
        title="h",
        colorbar_label="a long colorbar label",
    )
    # The suptitle and caption force constrained layout on, which is what defers the geometry.
    return FigureSpec(panels=((heatmap, annotation),), figsize=(10.0, 4.0), suptitle="S", caption="c" * 160)


def _text_axes(fig):
    """The free-text panel's axes: it has text, no image and no patches."""
    return next(a for a in fig.axes if a.texts and not a.images and not a.patches)


def test_the_text_is_rewrapped_wider_once_the_layout_has_run():
    """The defect itself: the build-time wrap folds against a rectangle narrower than the final cell."""
    fig = MatplotlibRenderer().render(_figure_with_a_width_giving_sibling())
    try:
        axes = _text_axes(fig)
        before = axes.texts[0].get_text().count("\n") + 1
        fig.canvas.draw()
        after = axes.texts[0].get_text().count("\n") + 1
        assert after < before, f"the text still folds onto {after} lines; the post-layout width was not used"
    finally:
        plt.close(fig)


def test_the_rewrapped_text_still_fits_inside_its_panel():
    """Recovering width must not overshoot into the overflow the wrapping exists to prevent."""
    fig = MatplotlibRenderer().render(_figure_with_a_width_giving_sibling())
    try:
        axes = _text_axes(fig)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        text_w = axes.texts[0].get_window_extent(renderer).width
        panel_w = axes.get_window_extent().width
        assert text_w <= panel_w, f"the rewrapped text is {text_w:.0f}px wide in a {panel_w:.0f}px panel"
        assert text_w > panel_w * 0.75, f"only {100 * text_w / panel_w:.0f}% of the panel is used; the rewrap did not take"
    finally:
        plt.close(fig)


def test_the_rewrap_runs_once_and_does_not_re_enter():
    """The rewrap mutates the artist, which schedules a draw; without disconnecting, that recurses."""
    import mlframe.reporting.renderers.matplotlib as renderer_mod

    calls = {"n": 0}
    real = renderer_mod.wrap_annotation_text

    def counted(*args, **kwargs):
        """Count every wrap, both the build-time one and any post-layout rewraps."""
        calls["n"] += 1
        return real(*args, **kwargs)

    renderer_mod.wrap_annotation_text = counted
    try:
        fig = MatplotlibRenderer().render(_figure_with_a_width_giving_sibling())
        fig.canvas.draw()
        fig.canvas.draw()  # a second draw must not wrap again either
        plt.close(fig)
    finally:
        renderer_mod.wrap_annotation_text = real
    assert calls["n"] == 2, f"wrapped {calls['n']} times; expected the build-time wrap plus exactly one rewrap"


def test_a_panel_that_was_already_wide_enough_is_left_alone():
    """Guard: the rewrap is for recovering width, not for rewriting text that was already right."""
    spec = FigureSpec(panels=((AnnotationPanelSpec(text="short note", title="t", fontsize=9),),), figsize=(8.0, 3.0))
    fig = MatplotlibRenderer().render(spec)
    try:
        before = _text_axes(fig).texts[0].get_text()
        fig.canvas.draw()
        assert _text_axes(fig).texts[0].get_text() == before
    finally:
        plt.close(fig)


@pytest.mark.parametrize("monospace", [False, True])
def test_the_rewrap_keeps_the_panels_own_font(monospace):
    """A rewrap that measured against the wrong font would fold the text to the wrong width."""
    spec = FigureSpec(
        panels=((AnnotationPanelSpec(text=LONG_TEXT, title="t", fontsize=9, monospace=monospace),),),
        figsize=(6.0, 3.0),
        suptitle="S",
    )
    fig = MatplotlibRenderer().render(spec)
    try:
        fig.canvas.draw()
        artist = _text_axes(fig).texts[0]
        assert artist.get_text(), "the panel lost its text entirely"
        renderer = fig.canvas.get_renderer()
        assert artist.get_window_extent(renderer).width <= _text_axes(fig).get_window_extent().width
    finally:
        plt.close(fig)
