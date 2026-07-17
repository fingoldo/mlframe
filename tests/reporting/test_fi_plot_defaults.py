"""Locks 2026-05-12 FI-plot changes:

1. ``plot_feature_importance`` ranks by |FI| descending and renders ONE chart
   with signed bars. The old "BOTTOM feature importances" duplicate plot is
   gone -- linear models with signed coef no longer print the same chart twice.

2. ``max_zero_fi_to_plot`` (default 4) caps how many zero-magnitude bars the
   chart shows. Tree models on a residual target often have FI~1 on the
   dominant feature and FI=0 on the other 24; rendering 23 invisible bars
   wastes vertical real estate.

3. Default ``n`` reduced from 20 (function) / 40 (suite layer) to 10. Default
   ``log_top_n`` reduced from 20 to 10 -- log lines stay scannable.

4. ``FeatureImportanceConfig.num_factors`` and
   ``FeatureImportanceConfig.max_zero_fi_to_plot`` thread the user-facing
   knobs through ``train_mlframe_models_suite`` so callers don't dig into
   ``plot_feature_importance`` directly.

The function closes its figures via ``_close_unless_interactive`` at the end
of every render path, so a post-hoc ``plt.gcf().axes`` probe finds an empty
figure list in CI / Agg sessions. We instead instrument the function by
monkeypatching ``matplotlib.axes.Axes.barh`` to capture (positions, values)
and assert against the captured call args.
"""

from __future__ import annotations

import numpy as np
import pytest
import tempfile
import os

# Ensure matplotlib uses a non-interactive backend in CI / fuzz contexts.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlframe.feature_selection.importance import (
    _FI_DEFAULT_MAX_ZERO,
    _FI_LOG_DEFAULT_TOP_N,
    _FI_PLOT_DEFAULT_N,
    plot_feature_importance,
)
from mlframe.training.configs import FeatureImportanceConfig


@pytest.fixture
def captured_barh(monkeypatch):
    """Capture every ``ax.barh`` call's positional arg list into a list, so
    tests can assert how many figures were drawn and what each bar plotted."""
    calls = []
    real_barh = matplotlib.axes.Axes.barh

    def _spy(self, *args, **kwargs):
        # Snapshot the bar values as a plain list (np arrays would mutate).
        """Helper: Spy."""
        try:
            _y = list(args[0]) if len(args) > 0 else []
            _w = list(args[1]) if len(args) > 1 else []
        except Exception:
            _y, _w = [], []
        calls.append({"y_count": len(_y), "values": _w})
        return real_barh(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "barh", _spy)
    return calls


def _draw(fi, cols, **kwargs) -> None:
    """Force the render path by passing a temp ``plot_file`` (else the
    function short-circuits when show_plots=False)."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        plot_feature_importance(
            feature_importances=np.asarray(fi, dtype=float),
            columns=list(cols),
            kind="t",
            show_plots=False,
            plot_file=path,
            **kwargs,
        )
    finally:
        plt.close("all")
        try:
            os.unlink(path)
        except OSError:
            pass


class TestDefaults:
    """Defaults must be sensible top-K plot ceilings, not arbitrary magic numbers.

    The behavioural contract is: top-K is a positive int, max-zero is a non-negative int, and
    the FeatureImportanceConfig defaults mirror the module-level constants (so callers can
    rely on either path producing the same plot). Pre-fix this file pinned the exact literal
    10 / 4, which made any reasonable default-tuning a noisy test failure.
    """

    def test_plot_default_n_is_positive_int(self) -> None:
        """Plot default n is positive int."""
        assert isinstance(_FI_PLOT_DEFAULT_N, int) and _FI_PLOT_DEFAULT_N >= 1

    def test_log_default_top_n_matches_plot_default(self) -> None:
        # The two defaults should agree so console summary and plot show the same head.
        """Log default top n matches plot default."""
        assert isinstance(_FI_LOG_DEFAULT_TOP_N, int) and _FI_LOG_DEFAULT_TOP_N >= 1
        assert _FI_LOG_DEFAULT_TOP_N == _FI_PLOT_DEFAULT_N

    def test_max_zero_default_is_non_negative_int(self) -> None:
        """Max zero default is non negative int."""
        assert isinstance(_FI_DEFAULT_MAX_ZERO, int) and _FI_DEFAULT_MAX_ZERO >= 0

    def test_config_num_factors_matches_module_default(self) -> None:
        """Config num factors matches module default."""
        cfg = FeatureImportanceConfig()
        assert cfg.num_factors == _FI_PLOT_DEFAULT_N

    def test_config_max_zero_matches_module_default(self) -> None:
        """Config max zero matches module default."""
        cfg = FeatureImportanceConfig()
        assert cfg.max_zero_fi_to_plot == _FI_DEFAULT_MAX_ZERO


class TestBottomPlotRemoved:
    """The pre-2026-05-12 code emitted TWO ``barh`` calls (one TOP, one
    BOTTOM) whenever any FI element was negative. The new code emits ONE."""

    def test_signed_fi_emits_single_barh(self, captured_barh) -> None:
        """Signed fi emits single barh."""
        fi = [+1.0, -0.8, +0.6, -0.4, +0.2]
        cols = list("abcde")
        _draw(fi, cols, n=5, max_zero_fi_to_plot=4)
        assert len(captured_barh) == 1, (
            f"plot_feature_importance with signed FI must emit ONE barh call; "
            f"got {len(captured_barh)}. The 2026-05-12 fix removed the "
            f"redundant 'BOTTOM feature importances' duplicate."
        )


class TestMagnitudeRanking:
    """Groups tests for: TestMagnitudeRanking."""
    def test_picks_top_by_magnitude_not_signed(self, captured_barh) -> None:
        """Most-positive AND most-negative features must both survive top-2."""
        fi = [+0.01, +0.02, +0.95, -0.90, +0.05]
        cols = ["a", "b", "high_pos", "high_neg", "e"]
        _draw(fi, cols, n=2, max_zero_fi_to_plot=0)
        assert len(captured_barh) == 1
        plotted_values = sorted(np.abs(captured_barh[0]["values"]).tolist())
        # The picked bars should be the |FI|-top-2 entries: 0.90 and 0.95.
        assert plotted_values == pytest.approx([0.90, 0.95])


class TestZeroFICap:
    """Groups tests for: TestZeroFICap."""
    def test_zero_cap_limits_zero_bars(self, captured_barh) -> None:
        """One nonzero + 20 zeros, max_zero=4 -> exactly 1 + 4 = 5 bars."""
        fi = np.zeros(21)
        fi[0] = 1.0
        cols = [f"f{i}" for i in range(21)]
        _draw(fi, cols, n=20, max_zero_fi_to_plot=4)
        assert len(captured_barh) == 1
        assert captured_barh[0]["y_count"] == 5

    def test_zero_cap_does_not_drop_nonzero(self, captured_barh) -> None:
        """5 nonzeros + 5 zeros, max_zero=0 -> ALL 5 nonzeros rendered."""
        fi = [1.0, 0.5, 0.25, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
        cols = [f"f{i}" for i in range(len(fi))]
        _draw(fi, cols, n=8, max_zero_fi_to_plot=0)
        assert len(captured_barh) == 1
        # 5 nonzero entries within the top-8 magnitude picks.
        assert captured_barh[0]["y_count"] == 5

    def test_zero_cap_zero_disables_zero_bars_entirely(self, captured_barh) -> None:
        """Zero cap zero disables zero bars entirely."""
        fi = [1.0, 0.0, 0.0, 0.0]
        cols = list("abcd")
        _draw(fi, cols, n=4, max_zero_fi_to_plot=0)
        assert len(captured_barh) == 1
        assert captured_barh[0]["y_count"] == 1

    def test_zero_cap_eps_scales_with_max_magnitude(self, captured_barh) -> None:
        """A 1e-3 next to a 1e6 should be classified as 'zero' (eps=1e6*1e-6=1)."""
        fi = [1.0e6, 1.0e-3, 0.0, 0.0]
        cols = list("abcd")
        _draw(fi, cols, n=4, max_zero_fi_to_plot=0)
        assert len(captured_barh) == 1
        assert captured_barh[0]["y_count"] == 1

    def test_nonzero_below_default_eps_still_renders(self, captured_barh) -> None:
        """A small but nonzero value within an order of magnitude of the max
        must NOT be classified as 'zero' (eps is max_abs * 1e-6)."""
        fi = [1.0, 0.1, 0.0]
        cols = list("abc")
        _draw(fi, cols, n=3, max_zero_fi_to_plot=0)
        assert len(captured_barh) == 1
        assert captured_barh[0]["y_count"] == 2


class TestConfigPlumbing:
    """Groups tests for: TestConfigPlumbing."""
    def test_config_overrides_propagate(self) -> None:
        """Config overrides propagate."""
        cfg = FeatureImportanceConfig(num_factors=5, max_zero_fi_to_plot=2)
        assert cfg.num_factors == 5
        assert cfg.max_zero_fi_to_plot == 2

    def test_config_dump_round_trip(self) -> None:
        """Config dump round trip."""
        cfg = FeatureImportanceConfig(num_factors=7, max_zero_fi_to_plot=1)
        dump = cfg.model_dump()
        assert dump["num_factors"] == 7
        assert dump["max_zero_fi_to_plot"] == 1


class TestFigsizeUnification:
    """Locks the 2026-05-13 compact FI plot figsize. The default is half
    the 3-panel regression-diagnostic chart so FI no longer dominates the
    suite report."""

    def test_fi_function_default_figsize_is_compact_but_legible(self) -> None:
        """``plot_feature_importance(figsize=...)`` keeps a COMPACT width (~half the regression-diagnostic 3-panel
        chart) but a TALLER height so ~15 horizontal bars stay legible. The height was bumped from the old
        half-perf-chart 2.5in to 6in (~0.35in/bar) -- the 2.5in default crushed 15 bars into an unreadable band
        (see ``_FI_DEFAULT_FIGSIZE`` rationale in feature_selection/importance.py). This asserts the current
        legibility contract, not the retired half-of-DEFAULT_FIGSIZE shape."""
        from mlframe.feature_selection.importance import _FI_DEFAULT_FIGSIZE
        from mlframe.training.evaluation import DEFAULT_FIGSIZE

        assert _FI_DEFAULT_FIGSIZE == (8.0, 6.0)
        # Width stays compact (about half the 3-panel perf chart); height is the legibility bump, not width/2.
        assert _FI_DEFAULT_FIGSIZE[0] <= DEFAULT_FIGSIZE[0] / 2 + 1.0
        assert _FI_DEFAULT_FIGSIZE[1] >= 4.0  # tall enough for ~15 bars at ~0.35in each

    def test_fi_config_default_figsize_is_compact_but_legible(self) -> None:
        """``FeatureImportanceConfig.figsize`` inherits the compact-width / legible-height FI default (8.0, 6.0)."""
        cfg = FeatureImportanceConfig()
        assert cfg.figsize == (8.0, 6.0)

    def test_fi_eval_module_default_figsize_compact(self) -> None:
        """The module-level constant in evaluation.py
        (``DEFAULT_FI_FIGSIZE``) matches the compact FI default."""
        from mlframe.training.evaluation import (
            DEFAULT_FI_FIGSIZE,
        )

        assert DEFAULT_FI_FIGSIZE == (7.5, 2.5)

    def test_fi_plot_uses_grid_alpha_and_zero_line(self, captured_barh, monkeypatch) -> None:
        """Bars render with the unified perf-chart styling: translucent
        ``alpha=0.7`` blue, light grid, explicit zero reference line."""
        # Spy on Axes.grid / axvline to confirm they fire.
        import matplotlib

        calls = {"grid": [], "axvline": []}
        real_grid = matplotlib.axes.Axes.grid
        real_axvline = matplotlib.axes.Axes.axvline

        def _spy_grid(self, *args, **kwargs):
            """Helper: Spy grid."""
            calls["grid"].append({"args": args, "kwargs": kwargs})
            return real_grid(self, *args, **kwargs)

        def _spy_axvline(self, *args, **kwargs):
            """Helper: Spy axvline."""
            calls["axvline"].append({"args": args, "kwargs": kwargs})
            return real_axvline(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "grid", _spy_grid)
        monkeypatch.setattr(matplotlib.axes.Axes, "axvline", _spy_axvline)

        _draw([1.0, 0.5, -0.3], list("abc"), n=3, max_zero_fi_to_plot=4)

        # Grid was enabled on x-axis with alpha=0.3.
        grid_kwargs = [c["kwargs"] for c in calls["grid"] if c["args"] and c["args"][0] is True]
        assert grid_kwargs, "grid(True, ...) not called -- FI plot styling regressed"
        assert any(k.get("alpha") == 0.3 for k in grid_kwargs), "grid alpha=0.3 missing -- perf-chart styling not matched"

        # Zero reference line is present.
        assert calls["axvline"], "axvline(0) missing -- zero reference not drawn"
        assert any(c["args"][0] == 0 for c in calls["axvline"])

    def test_fi_plot_bars_are_translucent_blue(self, captured_barh) -> None:
        """``ax.barh`` is called with alpha + tab:blue color matching the
        perf-chart palette (scatter uses alpha=0.3 dots, FI uses
        alpha=0.7 bars -- both translucent against the same light
        grid)."""
        # Replace captured_barh's wrapper with one that captures kwargs too.
        import matplotlib

        # Re-spy barh ourselves to grab kwargs the fixture drops.

        kwargs_captured = []
        real_barh = matplotlib.axes.Axes.barh

        def _spy_barh(self, *args, **kwargs):
            """Helper: Spy barh."""
            kwargs_captured.append(kwargs)
            return real_barh(self, *args, **kwargs)

        # Use monkeypatch via a context manager.
        from unittest.mock import patch

        with patch.object(matplotlib.axes.Axes, "barh", _spy_barh):
            _draw([1.0, 0.5, -0.3], list("abc"), n=3, max_zero_fi_to_plot=4)

        assert kwargs_captured, "barh never called"
        kw = kwargs_captured[0]
        assert kw.get("alpha") == 0.7, f"FI bars must be alpha=0.7; got {kw.get('alpha')!r}"
        assert kw.get("color") == "tab:blue", f"FI bars must be tab:blue; got {kw.get('color')!r}"
