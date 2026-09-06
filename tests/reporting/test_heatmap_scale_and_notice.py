"""A diverging heatmap must be anchored, and the skip notice must not print its own escape sequence.

Both defects were found by rendering the chart and looking at it, and neither is visible from the spec: the
correlation matrix carried the right numbers with the wrong colours, and the SHAP notice carried the right
reason with a mangled separator in front of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.model_comparison import compose_model_comparison_figure
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer


def _per_model(n: int = 1500, n_models: int = 5, seed: int = 0):
    """Several models scoring the same rows, agreeing only weakly -- rho well inside (0, 1)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    return {f"model_{i}": {"y_true": y, "y_score": np.clip(0.25 * y + rng.random(n) * 0.7, 0, 1)} for i in range(n_models)}


def _correlation_panel(spec):
    """The Spearman panel out of the composed figure."""
    for row in spec.panels:
        for panel in row:
            if panel is not None and "correlation" in getattr(panel, "title", "").lower():
                return panel
    raise AssertionError("the model-comparison figure has no correlation panel")


def test_the_correlation_scale_is_pinned_to_the_range_rho_can_take():
    """A diverging colormap's midpoint has to mean rho=0, which only holds if the scale is anchored.

    Autoscaled, a matrix of weak POSITIVE correlations (0.2 off-diagonal, 1.0 on it) puts the white midpoint
    near 0.6 and paints every off-diagonal cell in the colormap's negative half -- the reader sees strong
    anti-correlation where the data says weak agreement. Nothing about the numbers changes; the colours invert
    their meaning.
    """
    panel = _correlation_panel(compose_model_comparison_figure(_per_model(), "binary_classification"))
    assert panel.color_vmin == -1.0 and panel.color_vmax == 1.0, f"correlation scale is not pinned to [-1, 1]: {panel.color_vmin}..{panel.color_vmax}"

    off_diagonal = panel.matrix[~np.eye(panel.matrix.shape[0], dtype=bool)]
    assert np.nanmax(np.abs(off_diagonal)) < 0.9, "fixture no longer produces weakly-correlated models, so this test cannot detect the inversion"


def test_the_renderer_draws_the_heatmap_on_the_pinned_scale():
    """The spec's bounds have to reach the image, or pinning them changes nothing on screen."""
    spec = compose_model_comparison_figure(_per_model(), "binary_classification")
    fig = MatplotlibRenderer().render(spec)
    try:
        images = [im for ax in fig.axes for im in ax.get_images()]
        assert images, "no heatmap image was drawn"
        clims = [im.get_clim() for im in images]
        assert (-1.0, 1.0) in clims, f"no image is drawn on the pinned [-1, 1] scale; got {clims}"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_cell_text_is_readable_against_the_pinned_scale():
    """Text colour samples the colormap, so it must sample the SAME scale the cells were drawn on.

    Pinning the image without pinning the text leaves white labels on the near-white fill that weak
    correlations now produce -- the numbers become unreadable exactly where they were made honest.
    """
    spec = compose_model_comparison_figure(_per_model(), "binary_classification")
    fig = MatplotlibRenderer().render(spec)
    try:
        heat_axes = [ax for ax in fig.axes if ax.get_images()]
        assert heat_axes, "no heatmap axes"
        ax = heat_axes[0]
        # Off-diagonal cells sit near rho=0.2, which on a [-1, 1] RdBu_r scale is a pale fill: their labels
        # must NOT be white. "1.00" on the dark diagonal is the control -- it should still be light.
        pale = [t for t in ax.texts if t.get_text() not in ("", "1.00")]
        assert pale, "no off-diagonal cell labels were drawn"
        import matplotlib.colors as mcolors

        for t in pale[:8]:
            r, g, b, _a = mcolors.to_rgba(t.get_color())
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            assert luminance < 0.6, f"label {t.get_text()!r} is drawn at luminance {luminance:.2f} on a pale cell; it will not be readable"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


@pytest.mark.parametrize("reason", ["no explainer available", "the model exposes no tree structure"])
def test_the_skip_notice_prints_a_line_break_not_its_own_source(reason, tmp_path, monkeypatch):
    """The notice separator was a mangled escape that printed literally onto the delivered figure.

    It read ``SHAP panels not produced:' + BS + 'n<reason>`` -- source-level debris on a chart handed to a
    user. The text is read back off the FIGURE the production code built, not off a string this test composes:
    rebuilding the expression here would assert against itself and pass whatever ships.
    """
    from mlframe.reporting.charts import shap_panels

    captured = {}
    real_save = shap_panels._save_figure

    def _capture(fig, *args, **kwargs):
        """Grab the figure's text before the caller closes it."""
        captured["texts"] = [t.get_text() for t in fig.texts]
        return real_save(fig, *args, **kwargs)

    monkeypatch.setattr(shap_panels, "_save_figure", _capture)
    shap_panels._write_skip_notice(reason, str(tmp_path / "notice"), "matplotlib[png]")

    assert captured.get("texts"), "the notice drew no text onto the figure"
    drawn = captured["texts"][0]
    assert "BS" not in drawn and "' + " not in drawn, f"source-level debris printed onto the figure: {drawn!r}"
    assert drawn == f"SHAP panels not produced:\n{reason}", f"unexpected notice text: {drawn!r}"
    assert drawn.count("\n") == 1, "the reason must sit on its own line, below the heading"
