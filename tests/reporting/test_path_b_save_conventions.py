"""The raw-matplotlib charts saved with different conventions than the FigureSpec renderer.

SHAP and confusion-matrix images sit beside renderer output in one report. The renderer saves with
``bbox_inches="tight", pad_inches=0.15`` and honours ``FigureSpec.dpi`` so ``ReportingConfig.plot_dpi``
sets one pixel density for the run; the Path-B saver cropped with no pad and ignored the DPI entirely, so
those images were cropped tighter and rendered at a different text weight than every neighbouring chart.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from mlframe.reporting.charts.shap_panels import _SAVE_PAD_INCHES, _save_figure


def _figure() -> Figure:
    """A small figure with axes, so the tight bbox has something to crop to."""
    fig = Figure(figsize=(4.0, 3.0))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    ax.set_title("t")
    return fig


def test_the_path_b_pad_matches_the_renderer(tmp_path: Path):
    """One number, one definition: what the renderer actually passes to savefig is the reference.

    Read off the call rather than out of the source text: a source-text assertion passes for an
    implementation that computes the right-looking literal and then saves with something else, and breaks on
    a harmless reformat.
    """
    from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer

    seen = {}
    fig = _figure()
    real_savefig = fig.savefig

    def _spy(*args, **kwargs):
        """Record the kwargs the renderer saves with, then save for real."""
        seen.update(kwargs)
        return real_savefig(*args, **kwargs)

    fig.savefig = _spy  # type: ignore[method-assign]
    MatplotlibRenderer().save(fig, str(tmp_path / "fig.png"), "png")
    assert seen.get("pad_inches") == _SAVE_PAD_INCHES, (
        f"the renderer saved with pad_inches={seen.get('pad_inches')!r}, not {_SAVE_PAD_INCHES}; " "the Path-B saver is now the odd one out"
    )


def test_a_saved_figure_honours_the_requested_dpi(tmp_path: Path):
    """``plot_dpi`` sets one pixel density for the run, and these images were opting out of it."""
    from PIL import Image

    small = _save_figure(_figure(), str(tmp_path / "small"), "matplotlib[png]", 50)
    large = _save_figure(_figure(), str(tmp_path / "large"), "matplotlib[png]", 200)
    assert small and large, "nothing was written"
    with Image.open(small[0]) as im_s, Image.open(large[0]) as im_l:
        assert im_l.size[0] > im_s.size[0] * 2, f"200 dpi produced {im_l.size} against {im_s.size} at 50 dpi"


def test_no_dpi_keeps_matplotlib_s_own_default(tmp_path: Path):
    """Guard: passing nothing must not silently pin a density the caller did not ask for."""
    from PIL import Image

    written = _save_figure(_figure(), str(tmp_path / "default"), "matplotlib[png]")
    assert written
    with Image.open(written[0]) as im:
        assert im.size[0] == pytest.approx(4.0 * plt.rcParams["figure.dpi"], abs=40), f"unexpected default size {im.size}"


@pytest.mark.parametrize(
    "func_path",
    [
        ("mlframe.reporting.diagnostics_dispatch", "render_shap_diagnostic"),
        ("mlframe.reporting.charts.shap_panels", "shap_summary_and_dependence"),
    ],
)
def test_the_dpi_reaches_the_chart_from_the_dispatch(func_path):
    """A parameter nothing passes is not a fix; every hop of the chain has to carry it."""
    import importlib

    module, name = func_path
    fn = getattr(importlib.import_module(module), name)
    assert "plot_dpi" in inspect.signature(fn).parameters, f"{name} cannot receive the run's DPI"
