"""Tests for ``mlframe.reporting.charts.multiclass``.

Covers all 7 panel tokens (CONFUSION / PR_F1 / ROC / PR_CURVES /
CALIB_GRID / PROB_DIST / TOP_K_ACC) + composer + token routing +
matplotlib + plotly render smoke.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from mlframe.reporting.charts.multiclass import (
    ALLOWED_MULTICLASS_PANEL_TOKENS,
    compose_multiclass_figure,
)
from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import render_and_save
from mlframe.reporting.spec import (
    BarPanelSpec,
    HeatmapPanelSpec,
    LinePanelSpec,
    ViolinPanelSpec,
)


@pytest.fixture
def synth_3class():
    """300 rows, 3 classes, planted signal so model isn't random."""
    rng = np.random.default_rng(42)
    n = 300
    K = 3
    classes = ["cat", "dog", "bird"]
    pos = rng.integers(0, K, n)
    y_proba = rng.dirichlet(alpha=[1] * K, size=n)
    # Bump true class so signal is non-trivial.
    for i, t in enumerate(pos):
        y_proba[i, t] += 0.7
        y_proba[i] /= y_proba[i].sum()
    # y_true carries the actual class identifiers (matching ``classes``), per
    # the sklearn convention compose_multiclass_figure expects -- NOT bare
    # positional ints (which would all miss the string-keyed label map and be
    # remapped to -1 / excluded, collapsing every one-vs-rest panel).
    y_true = np.array([classes[t] for t in pos])
    return y_true, y_proba, classes


@pytest.fixture
def synth_4class():
    rng = np.random.default_rng(42)
    n = 400
    K = 4
    classes = ["a", "b", "c", "d"]
    pos = rng.integers(0, K, n)
    y_proba = rng.dirichlet(alpha=[1] * K, size=n)
    for i, t in enumerate(pos):
        y_proba[i, t] += 0.7
        y_proba[i] /= y_proba[i].sum()
    # y_true holds class identifiers matching ``classes`` (sklearn convention),
    # not positional ints -- see synth_3class.
    y_true = np.array([classes[t] for t in pos])
    return y_true, y_proba, classes


# ----------------------------------------------------------------------------
# Allowed token set
# ----------------------------------------------------------------------------


class TestAllowedTokens:
    def test_allowed_set_matches_documented(self):
        assert ALLOWED_MULTICLASS_PANEL_TOKENS == frozenset(
            {
                "CONFUSION",
                "CONFUSION_MARGINS",
                "CONFUSED_PAIRS",
                "PR_F1",
                "ROC",
                "PR_CURVES",
                "CALIB_GRID",
                "PROB_DIST",
                "TOP_K_ACC",
            }
        )


# ----------------------------------------------------------------------------
# Per-token spec shape
# ----------------------------------------------------------------------------


class TestPanelTypes:
    def test_confusion_returns_heatmap(self, synth_3class):
        y, p, c = synth_3class
        spec = compose_multiclass_figure(y, p, c, panels_template="CONFUSION")
        assert isinstance(spec.panels[0][0], HeatmapPanelSpec)
        # K x K matrix
        assert spec.panels[0][0].matrix.shape == (3, 3)

    def test_confusion_uses_cb_safe_sequential_cmap_not_diverging(self, synth_3class):
        """Confusion values are unsigned (counts / row-rates), so the heatmap must use the CB-safe sequential
        viridis -- a diverging red/blue map (RdBu_r) wrongly implies a meaningful zero-centre."""
        from mlframe.reporting.colors import HEATMAP_CMAP, resolve_heatmap_cmap

        y, p, c = synth_3class
        panel = compose_multiclass_figure(y, p, c, panels_template="CONFUSION").panels[0][0]
        assert panel.colormap == HEATMAP_CMAP
        assert resolve_heatmap_cmap(panel.colormap) == "viridis"
        assert panel.colormap != "RdBu_r"

    def test_pr_f1_returns_grouped_bar(self, synth_3class):
        y, p, c = synth_3class
        spec = compose_multiclass_figure(y, p, c, panels_template="PR_F1")
        panel = spec.panels[0][0]
        assert isinstance(panel, BarPanelSpec)
        # 3 series (precision, recall, F1) × 3 categories
        assert isinstance(panel.values, tuple) and len(panel.values) == 3
        assert len(panel.categories) == 3

    def test_roc_returns_line_with_K_series(self, synth_3class):
        y, p, c = synth_3class
        spec = compose_multiclass_figure(y, p, c, panels_template="ROC")
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        # 1 chance diagonal + K per-class curves, each on the same x-grid.
        assert isinstance(panel.y, tuple) and len(panel.y) == 4
        assert panel.series_labels[0] == "chance"
        # AUC value in each per-class legend label.
        assert all("AUC=" in lbl for lbl in panel.series_labels[1:])

    def test_pr_curves_returns_line_with_K_series(self, synth_3class):
        y, p, c = synth_3class
        spec = compose_multiclass_figure(y, p, c, panels_template="PR_CURVES")
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        # K curves + K dotted prevalence baselines.
        assert isinstance(panel.y, tuple) and len(panel.y) == 6
        assert all("AP=" in lbl for lbl in panel.series_labels[:3])

    def test_calib_grid_returns_line_with_K_plus_diagonal(self, synth_3class):
        y, p, c = synth_3class
        spec = compose_multiclass_figure(y, p, c, panels_template="CALIB_GRID")
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        # K class curves + 1 perfect-calibration diagonal
        assert isinstance(panel.y, tuple) and len(panel.y) == 4
        assert panel.series_labels[0] == "perfect"

    def test_prob_dist_returns_violin(self, synth_3class):
        y, p, c = synth_3class
        spec = compose_multiclass_figure(y, p, c, panels_template="PROB_DIST")
        panel = spec.panels[0][0]
        assert isinstance(panel, ViolinPanelSpec)
        assert len(panel.groups) == 3  # K groups
        # Group labels carry per-class N counts.
        assert all("(n=" in lbl for lbl in panel.group_labels)

    def test_top_k_acc_returns_line(self, synth_4class):
        y, p, c = synth_4class
        spec = compose_multiclass_figure(y, p, c, panels_template="TOP_K_ACC")
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        assert len(panel.x) == 4  # K
        # Top-k accuracy is monotone non-decreasing in k.
        y_arr = panel.y if not isinstance(panel.y, tuple) else panel.y[0]
        assert np.all(np.diff(y_arr) >= -1e-9)
        # Top-K = 1.0 (true class is always in top K=K).
        assert y_arr[-1] == pytest.approx(1.0)


# ----------------------------------------------------------------------------
# Composer + token routing
# ----------------------------------------------------------------------------


class TestComposer:
    def test_default_template_returns_6_panels(self, synth_3class):
        y, p, c = synth_3class
        spec = compose_multiclass_figure(y, p, c)  # default template
        # 6 tokens -> 3 rows × 2 cols
        assert len(spec.panels) == 3
        assert len(spec.panels[0]) == 2

    def test_subset_template_returns_fewer_panels(self, synth_3class):
        y, p, c = synth_3class
        spec = compose_multiclass_figure(y, p, c, panels_template="CONFUSION ROC")
        # 2 tokens -> 1 row × 2 cols
        assert len(spec.panels) == 1
        assert spec.panels[0][0] is not None
        assert spec.panels[0][1] is not None

    def test_unknown_token_raises(self, synth_3class):
        y, p, c = synth_3class
        with pytest.raises(ValueError, match="Unknown multiclass"):
            compose_multiclass_figure(y, p, c, panels_template="CONFUSION FOO")

    def test_suptitle_propagated(self, synth_3class):
        y, p, c = synth_3class
        spec = compose_multiclass_figure(y, p, c, panels_template="CONFUSION", suptitle="my model")
        assert spec.suptitle == "my model"

    def test_max_cols_controls_grid_width(self, synth_3class):
        y, p, c = synth_3class
        spec = compose_multiclass_figure(y, p, c, panels_template="CONFUSION PR_F1 ROC PR_CURVES", max_cols=4)
        assert len(spec.panels) == 1
        assert len(spec.panels[0]) == 4


# ----------------------------------------------------------------------------
# Render smoke
# ----------------------------------------------------------------------------


class TestRender:
    def test_render_via_matplotlib(self, synth_3class, tmp_path):
        y, p, c = synth_3class
        spec = compose_multiclass_figure(y, p, c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"), str(tmp_path / "mc"))
        assert os.path.exists(tmp_path / "mc.png")
        assert os.path.getsize(tmp_path / "mc.png") > 5000

    def test_render_via_plotly(self, synth_3class, tmp_path):
        y, p, c = synth_3class
        spec = compose_multiclass_figure(y, p, c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("plotly[html]"), str(tmp_path / "mc"))
        assert os.path.exists(tmp_path / "mc.html")
