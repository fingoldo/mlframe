"""Tests for ``mlframe.reporting.charts.multilabel``.

Covers all 7 panel tokens (PR_F1 / ROC / CALIB_GRID / COOCCURRENCE /
CARDINALITY / JACCARD_DIST / HAMMING_DIST) + composer + token routing +
matplotlib + plotly render smoke.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from mlframe.reporting.charts.multilabel import (
    ALLOWED_MULTILABEL_PANEL_TOKENS, compose_multilabel_figure,
)
from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import render_and_save
from mlframe.reporting.spec import (
    BarPanelSpec, FigureSpec, HeatmapPanelSpec, HistogramPanelSpec,
    LinePanelSpec,
)


@pytest.fixture
def synth_3label():
    """300 rows × 3 labels, planted signal (label probs correlate with label truth)."""
    rng = np.random.default_rng(42)
    n = 300
    K = 3
    y_true = rng.integers(0, 2, (n, K)).astype(np.int8)
    # Build proba so that label k is more likely when y_true[k]==1.
    base = rng.uniform(0.0, 0.5, (n, K))
    boost = y_true * 0.4
    y_proba = np.clip(base + boost + rng.normal(0, 0.05, (n, K)), 0.01, 0.99)
    return y_true, y_proba, ["spam", "promo", "social"]


@pytest.fixture
def synth_4label():
    rng = np.random.default_rng(42)
    n = 400
    K = 4
    y_true = rng.integers(0, 2, (n, K)).astype(np.int8)
    base = rng.uniform(0.0, 0.5, (n, K))
    boost = y_true * 0.4
    y_proba = np.clip(base + boost + rng.normal(0, 0.05, (n, K)), 0.01, 0.99)
    return y_true, y_proba, ["a", "b", "c", "d"]


# ----------------------------------------------------------------------------
# Allowed token set
# ----------------------------------------------------------------------------


class TestAllowedTokens:
    def test_allowed_set_matches_documented(self):
        assert ALLOWED_MULTILABEL_PANEL_TOKENS == frozenset({
            "PR_F1", "ROC", "CALIB_GRID", "COOCCURRENCE",
            "CARDINALITY", "JACCARD_DIST", "HAMMING_DIST",
        })


# ----------------------------------------------------------------------------
# Per-token spec shape
# ----------------------------------------------------------------------------


class TestPanelTypes:
    def test_pr_f1_returns_grouped_bar(self, synth_3label):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl, panels_template="PR_F1")
        panel = spec.panels[0][0]
        assert isinstance(panel, BarPanelSpec)
        # 3 series (precision/recall/F1) × 3 labels
        assert isinstance(panel.values, tuple) and len(panel.values) == 3
        assert len(panel.categories) == 3

    def test_roc_returns_line_with_K_series(self, synth_3label):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl, panels_template="ROC")
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        assert isinstance(panel.y, tuple) and len(panel.y) == 3
        assert all("AUC=" in s or "n/a" in s for s in panel.series_labels)

    def test_calib_grid_returns_line_with_K_plus_diagonal(self, synth_3label):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl, panels_template="CALIB_GRID")
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        # K label curves + 1 diagonal
        assert isinstance(panel.y, tuple) and len(panel.y) == 4
        assert panel.series_labels[0] == "perfect"

    def test_cooccurrence_returns_heatmap(self, synth_3label):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl, panels_template="COOCCURRENCE")
        panel = spec.panels[0][0]
        assert isinstance(panel, HeatmapPanelSpec)
        # K x K matrix
        assert panel.matrix.shape == (3, 3)
        # Diagonal cells are P(predicted=k | true=k), should be in [0, 1].
        for k in range(3):
            assert 0.0 <= panel.matrix[k, k] <= 1.0

    def test_cardinality_returns_grouped_bar(self, synth_3label):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl, panels_template="CARDINALITY")
        panel = spec.panels[0][0]
        assert isinstance(panel, BarPanelSpec)
        # 2 series (true / predicted), K+1 = 4 categories (0..K labels)
        assert isinstance(panel.values, tuple) and len(panel.values) == 2
        assert len(panel.categories) == 4
        # Total counts conserved.
        n = y.shape[0]
        assert int(panel.values[0].sum()) == n
        assert int(panel.values[1].sum()) == n

    def test_jaccard_dist_returns_histogram(self, synth_3label):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl, panels_template="JACCARD_DIST")
        panel = spec.panels[0][0]
        assert isinstance(panel, HistogramPanelSpec)
        # Jaccard ∈ [0, 1].
        assert panel.values.min() >= 0.0
        assert panel.values.max() <= 1.0
        assert len(panel.values) == y.shape[0]

    def test_hamming_dist_returns_histogram(self, synth_4label):
        y, p, lbl = synth_4label
        spec = compose_multilabel_figure(y, p, lbl, panels_template="HAMMING_DIST")
        panel = spec.panels[0][0]
        assert isinstance(panel, HistogramPanelSpec)
        # Hamming ∈ {0..K}.
        assert panel.values.min() >= 0.0
        assert panel.values.max() <= y.shape[1]
        assert len(panel.values) == y.shape[0]


# ----------------------------------------------------------------------------
# Composer + token routing
# ----------------------------------------------------------------------------


class TestComposer:
    def test_default_template_returns_5_panels(self, synth_3label):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl)  # default template
        # default = 5 tokens -> 3 rows × 2 cols
        assert len(spec.panels) == 3
        assert len(spec.panels[0]) == 2

    def test_subset_template_returns_fewer_panels(self, synth_3label):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl, panels_template="PR_F1 ROC")
        assert len(spec.panels) == 1
        assert spec.panels[0][0] is not None
        assert spec.panels[0][1] is not None

    def test_unknown_token_raises(self, synth_3label):
        y, p, lbl = synth_3label
        with pytest.raises(ValueError, match="Unknown multilabel"):
            compose_multilabel_figure(y, p, lbl, panels_template="PR_F1 BOGUS")

    def test_suptitle_propagated(self, synth_3label):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl, panels_template="PR_F1",
                                          suptitle="ml model")
        assert spec.suptitle == "ml model"

    def test_max_cols_controls_grid_width(self, synth_3label):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl,
                                          panels_template="PR_F1 ROC CALIB_GRID COOCCURRENCE",
                                          max_cols=4)
        assert len(spec.panels) == 1
        assert len(spec.panels[0]) == 4

    def test_shape_mismatch_raises(self, synth_3label):
        y, p, lbl = synth_3label
        # Truncate y_proba to wrong shape.
        with pytest.raises(ValueError, match="y_true .* != y_proba"):
            compose_multilabel_figure(y, p[:, :2], lbl, panels_template="PR_F1")

    def test_1d_input_raises(self, synth_3label):
        y, p, lbl = synth_3label
        with pytest.raises(ValueError, match="2-D"):
            compose_multilabel_figure(y[:, 0], p, lbl, panels_template="PR_F1")


# ----------------------------------------------------------------------------
# Render smoke
# ----------------------------------------------------------------------------


class TestRender:
    def test_render_via_matplotlib(self, synth_3label, tmp_path):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"),
                            str(tmp_path / "ml"))
        assert os.path.exists(tmp_path / "ml.png")
        assert os.path.getsize(tmp_path / "ml.png") > 5000

    def test_render_via_plotly(self, synth_3label, tmp_path):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("plotly[html]"),
                            str(tmp_path / "ml"))
        assert os.path.exists(tmp_path / "ml.html")
