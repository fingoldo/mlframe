"""Regression tests for the reporting_core audit findings.

Grouped by the defect CLASS rather than the module, because most of these recur: a recorded outcome that can
only ever be one value, a catalogue that drifts from the code it describes, and "not requested" reported as
"failed".
"""

import os
import re
from pathlib import Path

import numpy as np
import pytest

from mlframe.reporting import colors
from mlframe.reporting.auto_dispatch import select_binary_emphasis_panels
from mlframe.reporting.diagnostics_dispatch import _save_figure, render_target_acf_diagnostic

_SRC = Path(colors.__file__).parent


class TestCatalogueMatchesTheCode:
    """`describe_available_panels()` is the user-facing index of what the suite produces."""

    def test_every_wired_diagnostic_is_listed(self):
        """A diagnostic the dispatcher records must appear in the catalogue, or the index under-reports."""
        cat = (_SRC / "catalog.py").read_text(encoding="utf-8")
        listed = set(re.findall(r'^\s*\("([a-z_0-9]+)",', cat, flags=re.M))
        recorded: set = set()
        for name in ("diagnostics_dispatch.py", "_diagnostics_dispatch_extra.py"):
            src = (_SRC / name).read_text(encoding="utf-8")
            recorded |= set(re.findall(r'_record\(charts,\s*"([a-z_0-9]+)"', src))
        recorded.discard("useful")  # a local bool in a verdict string, not a chart name
        # Pre-fix, 19 of these were missing -- the catalogue described under half the suite's real output.
        assert not (recorded - listed), f"wired but absent from the catalogue: {sorted(recorded - listed)}"


class TestNotRequestedIsNotAFailure:
    """A backend that does not write PNGs has not failed to write one."""

    def test_save_figure_returns_none_when_png_was_never_requested(self, tmp_path):
        """Tri-state: True saved, False failed, None not applicable."""
        assert _save_figure(object(), "plotly[html]", str(tmp_path / "unused")) is None

    def test_save_figure_returns_true_on_a_real_save(self, tmp_path):
        """The success path is unchanged."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = Figure()
        fig.add_subplot(1, 1, 1).plot([0, 1], [0, 1])
        assert _save_figure(fig, "matplotlib[png]", str(tmp_path / "f")) is True
        assert os.path.exists(str(tmp_path / "f.png"))


class TestPositivesAreCountedByClassNotByNonZero:
    """`count_nonzero` calls every nonzero label positive, which is wrong for {-1,+1} and {1,2}."""

    @pytest.mark.parametrize(
        ("name", "neg", "pos"),
        [("zero_one", 0.0, 1.0), ("minus_plus", -1.0, 1.0), ("one_two", 1.0, 2.0)],
    )
    def test_imbalanced_data_gets_the_imbalanced_panel_order(self, name, neg, pos):
        """Emphasis must key on the positive CLASS, not on nonzero-ness."""
        y = np.r_[np.full(480, neg), np.full(20, pos)]
        out = select_binary_emphasis_panels(y, "ROC PR KS GAIN THRESHOLD SCORE_DIST", emphasis="data_aware")
        # Pre-fix, {-1,+1} and {1,2} reported n_pos == n and short-circuited, so ROC stayed in front.
        assert out.split()[0] == "PR", f"{name}: expected PR-led imbalanced order, got {out!r}"

    def test_balanced_data_still_leads_with_roc(self):
        """The balanced branch is unchanged."""
        y = np.r_[np.zeros(250), np.ones(250)]
        assert select_binary_emphasis_panels(y, "ROC PR KS", emphasis="data_aware").split()[0] == "ROC"


class TestAcfAlignsInsteadOfSkipping:
    """Every other entry point trims to the common prefix; this one refused outright."""

    def test_shorter_timestamps_still_render(self, tmp_path):
        """A split with partial temporal coverage must still get its ACF panel."""
        rng = np.random.default_rng(0)
        metrics: dict = {}
        ok = render_target_acf_diagnostic(
            y_true=rng.normal(0.0, 1.0, 200),
            timestamps=np.arange(150.0),  # shorter than y -- used to skip entirely
            plot_outputs="matplotlib[png]",
            base_path=str(tmp_path / "a"),
            metrics_dict=metrics,
        )
        assert ok is True
        assert "target_acf" in metrics["charts"]["saved"]

    def test_a_genuinely_too_short_series_is_still_skipped(self, tmp_path):
        """Below 8 points an ACF says nothing, so the guard that matters is kept."""
        assert (
            render_target_acf_diagnostic(
                y_true=np.arange(5.0), timestamps=np.arange(5.0),
                plot_outputs="matplotlib[png]", base_path=str(tmp_path / "b"), metrics_dict={},
            )
            is False
        )


class TestColorsSurface:
    """Small contracts a star-importing consumer relies on."""

    def test_batch_helper_is_exported(self):
        """The documented fast path was missing from __all__, so star-import consumers silently lost it."""
        assert "auto_text_colors_batch" in colors.__all__

    def test_both_branches_return_the_same_dtype(self):
        """The result's dtype must not depend on whether matplotlib happened to load."""
        arr = np.array([[0.1, 0.9]])
        good = colors.auto_text_colors_batch(arr, 0.0, 1.0, "viridis")
        fallback = colors.auto_text_colors_batch(arr, 0.0, 1.0, "definitely_not_a_colormap")
        assert good.dtype == fallback.dtype

    def test_line_style_varies_on_each_palette_wrap(self):
        """Past 20 series the colour repeats, so something else has to distinguish them."""
        n = len(colors.LINE_PALETTE)
        assert colors.line_color(0) == colors.line_color(n)  # colour genuinely repeats
        assert colors.line_style(0) != colors.line_style(n)  # but the dash pattern does not
        assert colors.line_style(0) == "-"  # wrap 0 is solid: unchanged for the common K <= 20 case
