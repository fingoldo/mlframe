"""Tests for ``mlframe.reporting.charts.training_curve`` (INV-24).

Covers history normalization (split-key aliasing), per-metric panel content, the
early-stopping vline + post-ES shading, the empty-history placeholder, render smoke,
and the biz_value verdict (ES marker sits at the divergence point on an overfitting
synthetic history).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from mlframe.reporting.charts.training_curve import (
    compose_training_curve_figure,
    normalize_history,
)
from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import render_and_save
from mlframe.reporting.spec import AnnotationPanelSpec, LinePanelSpec


def _overfitting_history(n_iter=120, turn=70):
    """Train falls monotonically; val falls then turns UP after ``turn`` (classic overfit)."""
    it = np.arange(n_iter)
    train = 1.0 / (1.0 + 0.05 * it)  # smoothly decreasing
    val = train.copy()
    # After the turn iteration, val rises (divergence opens up).
    rise = np.clip(it - turn, 0, None) * 0.004
    val = val + rise
    return {"rmse": {"train": train.tolist(), "val": val.tolist()}}


@pytest.fixture
def overfit_history():
    """Overfit history."""
    return _overfitting_history()


# ----------------------------------------------------------------------------
# normalize_history
# ----------------------------------------------------------------------------


class TestNormalize:
    """Groups tests for: TestNormalize."""
    def test_canonical_keys_passthrough(self):
        """Canonical keys passthrough."""
        h = {"rmse": {"train": [1.0, 0.5], "val": [1.0, 0.6]}}
        norm = normalize_history(h)
        assert set(norm["rmse"]) == {"train", "val"}
        assert isinstance(norm["rmse"]["train"], np.ndarray)

    def test_alias_keys_map_to_val(self):
        """Alias keys map to val."""
        for alias in ("valid", "validation", "test", "eval", "holdout"):
            norm = normalize_history({"l2": {"learn": [1.0], alias: [2.0]}})
            assert set(norm["l2"]) == {"train", "val"}
            assert norm["l2"]["val"][0] == 2.0

    def test_unknown_split_dropped(self):
        """Unknown split dropped."""
        norm = normalize_history({"m": {"train": [1.0], "bogus": [9.0]}})
        assert set(norm["m"]) == {"train"}

    def test_metric_with_no_known_split_dropped(self):
        """Metric with no known split dropped."""
        norm = normalize_history({"m": {"bogus": [9.0]}})
        assert norm == {}

    def test_first_alias_wins_not_overwrite(self):
        # Two val-like aliases: keep the first so a caller bug surfaces rather than silently merges.
        """First alias wins not overwrite."""
        norm = normalize_history({"m": {"val": [1.0], "valid": [2.0]}})
        assert norm["m"]["val"][0] == 1.0


# ----------------------------------------------------------------------------
# Panel content
# ----------------------------------------------------------------------------


class TestPanels:
    """Groups tests for: TestPanels."""
    def test_one_panel_per_metric(self):
        """One panel per metric."""
        h = {
            "rmse": {"train": [1.0, 0.5], "val": [1.0, 0.6]},
            "mae": {"train": [0.8, 0.4], "val": [0.8, 0.5]},
        }
        spec = compose_training_curve_figure(h)
        n_set = sum(1 for r in spec.panels for c in r if c is not None)
        assert n_set == 2

    def test_train_val_series_and_styles(self, overfit_history):
        """Train val series and styles."""
        spec = compose_training_curve_figure(overfit_history)
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        assert panel.series_labels == ("train", "val")
        assert len(panel.y) == 2
        assert panel.xlabel == "Iteration"

    def test_es_vline_and_shading(self, overfit_history):
        """Es vline and shading."""
        spec = compose_training_curve_figure(overfit_history, es_iteration=70)
        panel = spec.panels[0][0]
        assert panel.vlines is not None and len(panel.vlines) == 1
        assert panel.vlines[0][0] == 70.0
        assert "early stop" in panel.vlines[0][2]
        # Post-ES shaded span runs from the ES iter to the last iter.
        assert panel.vspans is not None and len(panel.vspans) == 1
        assert panel.vspans[0][0] == 70.0
        assert panel.vspans[0][1] == 119.0

    def test_no_es_no_vline(self, overfit_history):
        """No es no vline."""
        spec = compose_training_curve_figure(overfit_history)
        panel = spec.panels[0][0]
        assert panel.vlines is None
        assert panel.vspans is None

    def test_out_of_range_es_ignored(self, overfit_history):
        """Out of range es ignored."""
        spec = compose_training_curve_figure(overfit_history, es_iteration=9999)
        panel = spec.panels[0][0]
        assert panel.vlines is None

    def test_es_at_last_iter_no_shading(self, overfit_history):
        """Es at last iter no shading."""
        spec = compose_training_curve_figure(overfit_history, es_iteration=119)
        panel = spec.panels[0][0]
        assert panel.vlines is not None
        assert panel.vspans is None  # nothing past the last iteration to shade

    def test_single_split_metric(self):
        """Single split metric."""
        spec = compose_training_curve_figure({"m": {"train": [1.0, 0.5, 0.3]}})
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        assert panel.series_labels == ("train",)

    def test_metric_subset_and_order(self):
        """Metric subset and order."""
        h = {
            "rmse": {"train": [1.0], "val": [1.0]},
            "mae": {"train": [1.0], "val": [1.0]},
        }
        spec = compose_training_curve_figure(h, metrics=["mae"])
        n_set = sum(1 for r in spec.panels for c in r if c is not None)
        assert n_set == 1
        assert spec.panels[0][0].ylabel == "mae"

    def test_empty_history_placeholder(self):
        """Empty history placeholder."""
        spec = compose_training_curve_figure({})
        panel = spec.panels[0][0]
        assert isinstance(panel, AnnotationPanelSpec)
        assert "No train/val" in panel.text

    def test_suptitle(self, overfit_history):
        """Suptitle."""
        spec = compose_training_curve_figure(overfit_history, suptitle="LGB fit")
        assert spec.suptitle == "LGB fit"


# ----------------------------------------------------------------------------
# Render smoke
# ----------------------------------------------------------------------------


class TestRender:
    """Groups tests for: TestRender."""
    def test_matplotlib_render(self, overfit_history, tmp_path):
        """Matplotlib render."""
        spec = compose_training_curve_figure(overfit_history, es_iteration=70)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"), str(tmp_path / "tc"))
        assert os.path.exists(tmp_path / "tc.png")
        assert os.path.getsize(tmp_path / "tc.png") > 5000

    def test_plotly_render(self, overfit_history, tmp_path):
        """Plotly render."""
        spec = compose_training_curve_figure(overfit_history, es_iteration=70)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("plotly[html]"), str(tmp_path / "tc"))
        assert os.path.exists(tmp_path / "tc.html")


# ----------------------------------------------------------------------------
# biz_value
# ----------------------------------------------------------------------------


class TestTrainingCurveBizValue:
    """Groups tests for: TestTrainingCurveBizValue."""
    def test_biz_es_marker_sits_at_divergence(self):
        """On an overfit history (val turns up at K=70), the ES iter passed in is the val argmin.

        Verifies the panel marks the iteration where val stops improving -- the actionable point
        an honest early-stopping fit would have stopped at, where train/val divergence begins.
        """
        turn = 70
        h = _overfitting_history(n_iter=120, turn=turn)
        val = np.asarray(h["rmse"]["val"])
        es = int(np.argmin(val))
        # The synthetic's val argmin is the turn iteration (within a small margin).
        assert abs(es - turn) <= 2, f"val argmin {es} not near turn {turn}"
        spec = compose_training_curve_figure(h, es_iteration=es)
        panel = spec.panels[0][0]
        assert panel.vlines[0][0] == float(es)

    def test_biz_divergence_detectable_after_es(self):
        """Train keeps falling while val rises after the ES point -> a clear, measurable gap.

        The mean train/val gap in the post-ES region must dwarf the pre-ES gap; if a regression
        flattened the val curve (no overfit modelled) this verdict would fail.
        """
        turn = 70
        h = _overfitting_history(n_iter=120, turn=turn)
        train = np.asarray(h["rmse"]["train"])
        val = np.asarray(h["rmse"]["val"])
        pre_gap = float(np.mean(val[:turn] - train[:turn]))
        post_gap = float(np.mean(val[turn:] - train[turn:]))
        assert pre_gap < 0.01, f"pre-ES gap should be ~0, got {pre_gap}"
        assert post_gap > pre_gap + 0.05, f"post-ES divergence too small: {post_gap} vs {pre_gap}"


# ----------------------------------------------------------------------------
# metric_period sampling + layout
# ----------------------------------------------------------------------------


def _catboost_sampled_history(n=600, period=5):
    """CatBoost with metric_period=k logs learn at 0, k, 2k, ... plus the last iteration, validation every iteration."""
    it = np.arange(n, dtype=np.float64)
    train_full = 0.33 - 0.00002 * it  # linear, so interpolating the sampled points must reproduce it exactly
    val = 0.285 + 0.02 * np.exp(-it / 15.0) + 0.000002 * it
    pos = list(range(0, n, period))
    if pos[-1] != n - 1:
        pos.append(n - 1)
    return {"Huber": {"learn": train_full[pos].tolist(), "validation": val.tolist()}}, train_full, val


class TestMetricPeriodAlignment:
    """A train curve logged every k-th iteration must be drawn at its real iterations, not its array index."""

    def test_sampled_train_curve_spans_all_iterations(self):
        """121 learn points over 600 iterations cover the whole x range instead of stopping at x=120."""
        h, train_full, _ = _catboost_sampled_history()
        panel = compose_training_curve_figure(h, es_iteration=133, metric_period=5).panels[0][0]
        train = np.asarray(panel.y[panel.series_labels.index("train")])
        assert train.shape[0] == 600 and np.isfinite(train).all()
        np.testing.assert_allclose(train, train_full, rtol=1e-12)

    def test_gap_is_measured_at_the_early_stop(self):
        """The title's gap pairs train and val at the SAME (early-stop) iteration, not train[i] with val[i]."""
        h, train_full, val = _catboost_sampled_history()
        panel = compose_training_curve_figure(h, es_iteration=133, metric_period=5).panels[0][0]
        expected = f"{val[133] - train_full[133]:+.3g} at early stop"
        assert expected in panel.title, panel.title
        assert f"{val[599] - train_full[599]:+.3g} at iter 599" in panel.title

    def test_short_series_without_period_keeps_nan_tail(self):
        """Without a known metric_period the short series is not stretched (it may have genuinely stopped early)."""
        h, _, _ = _catboost_sampled_history()
        train = np.asarray(compose_training_curve_figure(h).panels[0][0].y[0])
        assert np.isfinite(train[:121]).all() and np.isnan(train[121:]).all()

    def test_truncated_series_not_matching_the_period_grid_is_not_stretched(self):
        """A series whose length does not match the period grid is a real early stop: keep the NaN tail."""
        val = np.linspace(1.0, 0.5, 600)
        h = {"m": {"train": np.linspace(1.0, 0.4, 50).tolist(), "val": val.tolist()}}
        train = np.asarray(compose_training_curve_figure(h, metric_period=5).panels[0][0].y[0])
        assert np.isnan(train[50:]).all()


class TestLayout:
    """Single-metric figures use the full width; the title does not repeat what the legend says."""

    def test_single_panel_is_not_padded_to_two_columns(self, overfit_history):
        """A lone panel must not share a 2-wide grid inside a 1-wide figure (half-width plot, shredded title)."""
        spec = compose_training_curve_figure(overfit_history)
        assert len(spec.panels) == 1 and len(spec.panels[0]) == 1
        assert spec.figsize[0] == pytest.approx(9.0)

    def test_two_metrics_fill_two_columns(self):
        """Two metrics still share a row."""
        h = {"a": {"train": [1.0, 0.5], "val": [1.0, 0.6]}, "b": {"train": [1.0, 0.5], "val": [1.0, 0.6]}}
        spec = compose_training_curve_figure(h)
        assert len(spec.panels[0]) == 2 and spec.figsize[0] == pytest.approx(18.0)

    def test_early_stop_named_once_in_the_legend_not_the_title(self, overfit_history):
        """The ES iteration lives in the vline legend label; the title no longer repeats it."""
        panel = compose_training_curve_figure(overfit_history, es_iteration=70).panels[0][0]
        assert "ES @" not in panel.title
        assert panel.vlines[0][2] == "early stop @ 70"
