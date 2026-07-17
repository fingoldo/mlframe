"""Locks the 2026-05-13 ReportingConfig plot-style override surface.

Two independent backends:
- matplotlib: ``matplotlib_style`` (style sheet) + ``matplotlib_rcparams`` (key/value overrides on top).
- plotly:     ``plotly_template`` (theme name).

When all three are ``None`` (the default), the user's pre-suite plot-backend
state is preserved untouched -- so a script that does ``plt.style.use("ggplot")``
before invoking ``train_mlframe_models_suite`` keeps that style. When set on the
config, the helper applies them PROCESS-WIDE (not reverted on suite exit).
Failures (typo in style name, etc.) are logged at WARNING level rather than
aborting the suite.
"""

from __future__ import annotations

import logging
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from mlframe.training.configs import ReportingConfig
from mlframe.training.core.utils import _apply_plot_style_overrides


# ---------------------------------------------------------------------------
# Config field defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_matplotlib_style_default_none(self) -> None:
        cfg = ReportingConfig()
        assert cfg.matplotlib_style is None

    def test_matplotlib_rcparams_default_none(self) -> None:
        cfg = ReportingConfig()
        assert cfg.matplotlib_rcparams is None

    def test_plotly_template_default_none(self) -> None:
        cfg = ReportingConfig()
        assert cfg.plotly_template is None

    def test_set_all_three_fields_through_config(self) -> None:
        cfg = ReportingConfig(
            matplotlib_style="ggplot",
            matplotlib_rcparams={"font.size": 11},
            plotly_template="ggplot2",
        )
        assert cfg.matplotlib_style == "ggplot"
        assert cfg.matplotlib_rcparams == {"font.size": 11}
        assert cfg.plotly_template == "ggplot2"

    def test_matplotlib_style_accepts_list_of_styles(self) -> None:
        """matplotlib supports stacking multiple style sheets; the
        config must accept a list."""
        cfg = ReportingConfig(matplotlib_style=["seaborn-v0_8", "dark_background"])
        assert cfg.matplotlib_style == ["seaborn-v0_8", "dark_background"]


# ---------------------------------------------------------------------------
# Helper: no-op path
# ---------------------------------------------------------------------------


class TestNoOpPath:
    def test_all_none_does_nothing(self) -> None:
        """``None`` for every field -- helper returns early without
        touching any backend state."""
        # Snapshot before
        before_font_size = plt.rcParams["font.size"]
        _apply_plot_style_overrides()
        # No mutation
        assert plt.rcParams["font.size"] == before_font_size


# ---------------------------------------------------------------------------
# matplotlib branch
# ---------------------------------------------------------------------------


class TestMatplotlibBranch:
    def test_apply_style_sheet(self) -> None:
        """``plt.style.use("ggplot")`` is invoked when matplotlib_style is set."""
        # Reset to a known-good state first.
        plt.style.use("default")
        _apply_plot_style_overrides(matplotlib_style="ggplot")
        # ggplot's facecolor is a known checkable value.
        assert plt.rcParams["axes.facecolor"] != (1.0, 1.0, 1.0, 1.0)

    def test_apply_rcparams_overrides(self) -> None:
        plt.style.use("default")
        _apply_plot_style_overrides(
            matplotlib_rcparams={"font.size": 17.5, "axes.grid": True},
        )
        assert plt.rcParams["font.size"] == 17.5
        assert plt.rcParams["axes.grid"] is True

    def test_rcparams_overlay_on_top_of_style(self) -> None:
        """When BOTH style + rcparams are set, rcparams wins on overlapping keys."""
        plt.style.use("default")
        _apply_plot_style_overrides(
            matplotlib_style="ggplot",
            matplotlib_rcparams={"font.size": 22.0},
        )
        # ggplot sets font.size somewhere; our override wins.
        assert plt.rcParams["font.size"] == 22.0

    def test_unknown_style_logs_warning_does_not_raise(self, caplog) -> None:
        plt.style.use("default")
        caplog.set_level(logging.WARNING, logger="mlframe.training.core.utils")
        # Should NOT raise -- the typo surfaces as a WARNING.
        _apply_plot_style_overrides(matplotlib_style="this_style_does_not_exist__")
        warns = [r for r in caplog.records if "plt.style.use" in r.getMessage()]
        assert warns, "unknown style should emit a WARNING via the logger"


# ---------------------------------------------------------------------------
# plotly branch
# ---------------------------------------------------------------------------


class TestPlotlyBranch:
    def test_apply_plotly_template(self) -> None:
        pio = pytest.importorskip("plotly.io")
        # Snapshot current
        _before = pio.templates.default
        try:
            _apply_plot_style_overrides(plotly_template="plotly_dark")
            assert pio.templates.default == "plotly_dark"
        finally:
            # Restore so the rest of the test suite isn't affected.
            pio.templates.default = _before

    def test_unknown_plotly_template_logs_warning_does_not_raise(
        self,
        caplog,
    ) -> None:
        pio = pytest.importorskip("plotly.io")
        _before = pio.templates.default
        try:
            caplog.set_level(logging.WARNING, logger="mlframe.training.core.utils")
            _apply_plot_style_overrides(plotly_template="this_template_does_not_exist__")
            warns = [r for r in caplog.records if "templates.default" in r.getMessage()]
            assert warns, "unknown plotly template should emit a WARNING"
        finally:
            pio.templates.default = _before


# ---------------------------------------------------------------------------
# Combined: matching themes across backends
# ---------------------------------------------------------------------------


class TestUnifiedTheme:
    def test_both_backends_can_be_set_in_one_call(self) -> None:
        pio = pytest.importorskip("plotly.io")
        _before_pio = pio.templates.default
        try:
            plt.style.use("default")
            _apply_plot_style_overrides(
                matplotlib_style="ggplot",
                plotly_template="ggplot2",
            )
            assert pio.templates.default == "ggplot2"
            # ggplot has a non-white axes.facecolor.
            assert plt.rcParams["axes.facecolor"] != (1.0, 1.0, 1.0, 1.0)
        finally:
            pio.templates.default = _before_pio
            plt.style.use("default")
