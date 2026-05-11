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

from mlframe.feature_importance import (
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
            columns=list(cols), kind="t",
            show_plots=False, plot_file=path, **kwargs,
        )
    finally:
        plt.close("all")
        try:
            os.unlink(path)
        except OSError:
            pass


class TestDefaults:
    def test_plot_default_n_is_ten(self) -> None:
        assert _FI_PLOT_DEFAULT_N == 10

    def test_log_default_top_n_is_ten(self) -> None:
        assert _FI_LOG_DEFAULT_TOP_N == 10

    def test_max_zero_default_is_four(self) -> None:
        assert _FI_DEFAULT_MAX_ZERO == 4

    def test_config_num_factors_default_is_ten(self) -> None:
        cfg = FeatureImportanceConfig()
        assert cfg.num_factors == 10

    def test_config_max_zero_default_is_four(self) -> None:
        cfg = FeatureImportanceConfig()
        assert cfg.max_zero_fi_to_plot == 4


class TestBottomPlotRemoved:
    """The pre-2026-05-12 code emitted TWO ``barh`` calls (one TOP, one
    BOTTOM) whenever any FI element was negative. The new code emits ONE."""

    def test_signed_fi_emits_single_barh(self, captured_barh) -> None:
        fi = [+1.0, -0.8, +0.6, -0.4, +0.2]
        cols = list("abcde")
        _draw(fi, cols, n=5, max_zero_fi_to_plot=4)
        assert len(captured_barh) == 1, (
            f"plot_feature_importance with signed FI must emit ONE barh call; "
            f"got {len(captured_barh)}. The 2026-05-12 fix removed the "
            f"redundant 'BOTTOM feature importances' duplicate."
        )


class TestMagnitudeRanking:
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
    def test_config_overrides_propagate(self) -> None:
        cfg = FeatureImportanceConfig(num_factors=5, max_zero_fi_to_plot=2)
        assert cfg.num_factors == 5
        assert cfg.max_zero_fi_to_plot == 2

    def test_config_dump_round_trip(self) -> None:
        cfg = FeatureImportanceConfig(num_factors=7, max_zero_fi_to_plot=1)
        dump = cfg.model_dump()
        assert dump["num_factors"] == 7
        assert dump["max_zero_fi_to_plot"] == 1
