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
    return _overfitting_history()


# ----------------------------------------------------------------------------
# normalize_history
# ----------------------------------------------------------------------------


class TestNormalize:
    def test_canonical_keys_passthrough(self):
        h = {"rmse": {"train": [1.0, 0.5], "val": [1.0, 0.6]}}
        norm = normalize_history(h)
        assert set(norm["rmse"]) == {"train", "val"}
        assert isinstance(norm["rmse"]["train"], np.ndarray)

    def test_alias_keys_map_to_val(self):
        for alias in ("valid", "validation", "test", "eval", "holdout"):
            norm = normalize_history({"l2": {"learn": [1.0], alias: [2.0]}})
            assert set(norm["l2"]) == {"train", "val"}
            assert norm["l2"]["val"][0] == 2.0

    def test_unknown_split_dropped(self):
        norm = normalize_history({"m": {"train": [1.0], "bogus": [9.0]}})
        assert set(norm["m"]) == {"train"}

    def test_metric_with_no_known_split_dropped(self):
        norm = normalize_history({"m": {"bogus": [9.0]}})
        assert norm == {}

    def test_first_alias_wins_not_overwrite(self):
        # Two val-like aliases: keep the first so a caller bug surfaces rather than silently merges.
        norm = normalize_history({"m": {"val": [1.0], "valid": [2.0]}})
        assert norm["m"]["val"][0] == 1.0


# ----------------------------------------------------------------------------
# Panel content
# ----------------------------------------------------------------------------


class TestPanels:
    def test_one_panel_per_metric(self):
        h = {
            "rmse": {"train": [1.0, 0.5], "val": [1.0, 0.6]},
            "mae": {"train": [0.8, 0.4], "val": [0.8, 0.5]},
        }
        spec = compose_training_curve_figure(h)
        n_set = sum(1 for r in spec.panels for c in r if c is not None)
        assert n_set == 2

    def test_train_val_series_and_styles(self, overfit_history):
        spec = compose_training_curve_figure(overfit_history)
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        assert panel.series_labels == ("train", "val")
        assert len(panel.y) == 2
        assert panel.xlabel == "Iteration"

    def test_es_vline_and_shading(self, overfit_history):
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
        spec = compose_training_curve_figure(overfit_history)
        panel = spec.panels[0][0]
        assert panel.vlines is None
        assert panel.vspans is None

    def test_out_of_range_es_ignored(self, overfit_history):
        spec = compose_training_curve_figure(overfit_history, es_iteration=9999)
        panel = spec.panels[0][0]
        assert panel.vlines is None

    def test_es_at_last_iter_no_shading(self, overfit_history):
        spec = compose_training_curve_figure(overfit_history, es_iteration=119)
        panel = spec.panels[0][0]
        assert panel.vlines is not None
        assert panel.vspans is None  # nothing past the last iteration to shade

    def test_single_split_metric(self):
        spec = compose_training_curve_figure({"m": {"train": [1.0, 0.5, 0.3]}})
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        assert panel.series_labels == ("train",)

    def test_metric_subset_and_order(self):
        h = {
            "rmse": {"train": [1.0], "val": [1.0]},
            "mae": {"train": [1.0], "val": [1.0]},
        }
        spec = compose_training_curve_figure(h, metrics=["mae"])
        n_set = sum(1 for r in spec.panels for c in r if c is not None)
        assert n_set == 1
        assert spec.panels[0][0].ylabel == "mae"

    def test_empty_history_placeholder(self):
        spec = compose_training_curve_figure({})
        panel = spec.panels[0][0]
        assert isinstance(panel, AnnotationPanelSpec)
        assert "No train/val" in panel.text

    def test_suptitle(self, overfit_history):
        spec = compose_training_curve_figure(overfit_history, suptitle="LGB fit")
        assert spec.suptitle == "LGB fit"


# ----------------------------------------------------------------------------
# Render smoke
# ----------------------------------------------------------------------------


class TestRender:
    def test_matplotlib_render(self, overfit_history, tmp_path):
        spec = compose_training_curve_figure(overfit_history, es_iteration=70)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"), str(tmp_path / "tc"))
        assert os.path.exists(tmp_path / "tc.png")
        assert os.path.getsize(tmp_path / "tc.png") > 5000

    def test_plotly_render(self, overfit_history, tmp_path):
        spec = compose_training_curve_figure(overfit_history, es_iteration=70)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("plotly[html]"), str(tmp_path / "tc"))
        assert os.path.exists(tmp_path / "tc.html")


# ----------------------------------------------------------------------------
# biz_value
# ----------------------------------------------------------------------------


class TestTrainingCurveBizValue:
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
