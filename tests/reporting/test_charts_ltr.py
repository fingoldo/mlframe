"""Tests for ``mlframe.reporting.charts.ltr``.

Covers all 6 panel tokens (NDCG_K / NDCG_DIST / LIFT / MRR_DIST /
SCORE_BY_REL / TOP1_BY_QSIZE) + composer + token routing +
matplotlib + plotly render smoke.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from mlframe.reporting.charts.ltr import (
    ALLOWED_LTR_PANEL_TOKENS, compose_ltr_figure,
)
from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import render_and_save
from mlframe.reporting.spec import (
    FigureSpec, HistogramPanelSpec, LinePanelSpec, ViolinPanelSpec,
)


@pytest.fixture
def synth_ltr():
    """50 queries × variable size (3..15 docs) with graded relevance 0..3.

    Predicted score correlates with relevance + noise so the ranker is
    non-trivial but imperfect.
    """
    rng = np.random.default_rng(42)
    y_true: list = []
    y_score: list = []
    group_ids: list = []
    n_queries = 50
    for q in range(n_queries):
        sz = int(rng.integers(3, 16))
        rels = rng.integers(0, 4, sz)
        # Score = relevance + noise -> correlated but noisy.
        scores = rels.astype(float) + rng.normal(0, 0.5, sz)
        y_true.extend(rels.tolist())
        y_score.extend(scores.tolist())
        group_ids.extend([q] * sz)
    return (np.asarray(y_true), np.asarray(y_score, dtype=np.float64),
            np.asarray(group_ids))


@pytest.fixture
def synth_ltr_large():
    """Larger LTR fixture (200 queries) for buckets-by-qsize coverage."""
    rng = np.random.default_rng(7)
    y_true: list = []
    y_score: list = []
    group_ids: list = []
    for q in range(200):
        sz = int(rng.integers(2, 25))
        rels = rng.integers(0, 4, sz)
        scores = rels.astype(float) + rng.normal(0, 0.7, sz)
        y_true.extend(rels.tolist())
        y_score.extend(scores.tolist())
        group_ids.extend([q] * sz)
    return (np.asarray(y_true), np.asarray(y_score, dtype=np.float64),
            np.asarray(group_ids))


# ----------------------------------------------------------------------------
# Allowed token set
# ----------------------------------------------------------------------------


class TestAllowedTokens:
    def test_allowed_set_matches_documented(self):
        assert ALLOWED_LTR_PANEL_TOKENS == frozenset({
            "NDCG_K", "NDCG_DIST", "NDCG_BY_QSIZE", "LIFT", "MRR_DIST",
            "SCORE_BY_REL", "TOP1_BY_QSIZE",
        })


# ----------------------------------------------------------------------------
# Per-token spec shape
# ----------------------------------------------------------------------------


class TestPanelTypes:
    def test_ndcg_k_returns_line(self, synth_ltr):
        y, s, g = synth_ltr
        spec = compose_ltr_figure(y, s, g, panels_template="NDCG_K")
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        # x = 1..max_per_query; values in [0,1].
        assert panel.x[0] == 1.0
        y_arr = panel.y if not isinstance(panel.y, tuple) else panel.y[0]
        assert np.all((y_arr >= 0.0) & (y_arr <= 1.0))

    def test_ndcg_dist_returns_violin(self, synth_ltr):
        y, s, g = synth_ltr
        spec = compose_ltr_figure(y, s, g, panels_template="NDCG_DIST")
        panel = spec.panels[0][0]
        assert isinstance(panel, ViolinPanelSpec)
        assert len(panel.groups) == 1  # single aggregated violin
        assert panel.groups[0].min() >= 0.0
        assert panel.groups[0].max() <= 1.0

    def test_lift_returns_line(self, synth_ltr):
        y, s, g = synth_ltr
        spec = compose_ltr_figure(y, s, g, panels_template="LIFT")
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        # Lift values are normalised; should be in [0, 1].
        y_arr = panel.y if not isinstance(panel.y, tuple) else panel.y[0]
        assert np.all((y_arr >= 0.0) & (y_arr <= 1.0 + 1e-9))

    def test_mrr_dist_returns_histogram(self, synth_ltr):
        y, s, g = synth_ltr
        spec = compose_ltr_figure(y, s, g, panels_template="MRR_DIST")
        panel = spec.panels[0][0]
        assert isinstance(panel, HistogramPanelSpec)
        # Reciprocal rank ∈ [0, 1].
        assert panel.values.min() >= 0.0
        assert panel.values.max() <= 1.0
        # Title carries MRR.
        assert "MRR=" in panel.title

    def test_score_by_rel_returns_violin(self, synth_ltr):
        y, s, g = synth_ltr
        spec = compose_ltr_figure(y, s, g, panels_template="SCORE_BY_REL")
        panel = spec.panels[0][0]
        assert isinstance(panel, ViolinPanelSpec)
        # 4 grades (0..3) -> 4 violins.
        assert len(panel.groups) == 4
        # Group labels carry per-grade N.
        assert all("(n=" in lbl for lbl in panel.group_labels)

    def test_top1_by_qsize_returns_line(self, synth_ltr_large):
        y, s, g = synth_ltr_large
        spec = compose_ltr_figure(y, s, g, panels_template="TOP1_BY_QSIZE")
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        # 5 buckets per the implementation contract.
        assert len(panel.x) == 5
        # Top-1 acc in [0, 1] (or NaN for empty buckets).
        y_arr = panel.y if not isinstance(panel.y, tuple) else panel.y[0]
        finite = y_arr[~np.isnan(y_arr)]
        assert np.all((finite >= 0.0) & (finite <= 1.0))


# ----------------------------------------------------------------------------
# Composer + token routing
# ----------------------------------------------------------------------------


class TestComposer:
    def test_default_template_returns_5_panels(self, synth_ltr):
        y, s, g = synth_ltr
        spec = compose_ltr_figure(y, s, g)  # default template
        # default = 5 tokens -> 3 rows × 2 cols
        assert len(spec.panels) == 3
        assert len(spec.panels[0]) == 2

    def test_subset_template_returns_fewer_panels(self, synth_ltr):
        y, s, g = synth_ltr
        spec = compose_ltr_figure(y, s, g, panels_template="NDCG_K LIFT")
        assert len(spec.panels) == 1
        assert spec.panels[0][0] is not None
        assert spec.panels[0][1] is not None

    def test_unknown_token_raises(self, synth_ltr):
        y, s, g = synth_ltr
        with pytest.raises(ValueError, match="Unknown LTR"):
            compose_ltr_figure(y, s, g, panels_template="NDCG_K NOPE")

    def test_suptitle_propagated(self, synth_ltr):
        y, s, g = synth_ltr
        spec = compose_ltr_figure(y, s, g, panels_template="NDCG_K",
                                   suptitle="ranker baseline")
        assert spec.suptitle == "ranker baseline"

    def test_max_cols_controls_grid_width(self, synth_ltr):
        y, s, g = synth_ltr
        spec = compose_ltr_figure(y, s, g,
                                   panels_template="NDCG_K NDCG_DIST LIFT MRR_DIST",
                                   max_cols=4)
        assert len(spec.panels) == 1
        assert len(spec.panels[0]) == 4

    def test_length_mismatch_raises(self, synth_ltr):
        y, s, g = synth_ltr
        with pytest.raises(ValueError, match="length mismatch"):
            compose_ltr_figure(y[:-1], s, g, panels_template="NDCG_K")

    def test_2d_input_raises(self, synth_ltr):
        y, s, g = synth_ltr
        y2d = y.reshape(-1, 1)
        with pytest.raises(ValueError, match="1-D"):
            compose_ltr_figure(y2d, s, g, panels_template="NDCG_K")


# ----------------------------------------------------------------------------
# Render smoke
# ----------------------------------------------------------------------------


class TestRender:
    def test_render_via_matplotlib(self, synth_ltr, tmp_path):
        y, s, g = synth_ltr
        spec = compose_ltr_figure(y, s, g)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"),
                            str(tmp_path / "ltr"))
        assert os.path.exists(tmp_path / "ltr.png")
        assert os.path.getsize(tmp_path / "ltr.png") > 5000

    def test_render_via_plotly(self, synth_ltr, tmp_path):
        y, s, g = synth_ltr
        spec = compose_ltr_figure(y, s, g)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("plotly[html]"),
                            str(tmp_path / "ltr"))
        assert os.path.exists(tmp_path / "ltr.html")
