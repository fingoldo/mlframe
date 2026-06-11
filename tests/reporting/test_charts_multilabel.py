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
            "CARDINALITY", "JACCARD_DIST", "HAMMING_DIST", "THRESHOLD_SWEEP",
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
        # 1 chance diagonal + K per-label curves.
        assert isinstance(panel.y, tuple) and len(panel.y) == 4
        assert panel.series_labels[0] == "chance"
        assert all("AUC=" in s or "n/a" in s for s in panel.series_labels[1:])

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

    def test_jaccard_dist_returns_prebinned_histogram(self, synth_3label):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl, panels_template="JACCARD_DIST")
        panel = spec.panels[0][0]
        assert isinstance(panel, HistogramPanelSpec)
        # Pre-binned at spec-build: spec carries O(bins) data, not length-n raw values.
        assert panel.bin_centers is not None and panel.bin_width is not None
        assert len(panel.values) == len(panel.bin_centers) <= 20
        # Jaccard bin centers span [0, 1].
        assert panel.bin_centers.min() >= 0.0
        assert panel.bin_centers.max() <= 1.0

    def test_hamming_dist_returns_prebinned_histogram(self, synth_4label):
        y, p, lbl = synth_4label
        spec = compose_multilabel_figure(y, p, lbl, panels_template="HAMMING_DIST")
        panel = spec.panels[0][0]
        assert isinstance(panel, HistogramPanelSpec)
        assert panel.bin_centers is not None and panel.bin_width is not None
        assert len(panel.values) == len(panel.bin_centers)
        # Hamming ∈ {0..K}: bin centers stay within that range.
        assert panel.bin_centers.min() >= 0.0
        assert panel.bin_centers.max() <= y.shape[1]


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


# ----------------------------------------------------------------------------
# THRESHOLD_SWEEP (per-label F1 x threshold heatmap)
# ----------------------------------------------------------------------------

from sklearn.metrics import f1_score  # noqa: E402

from mlframe.reporting.charts.multilabel import (  # noqa: E402
    _per_label_f1_sweep, _SWEEP_N_THRESHOLDS, _threshold_sweep_panel,
)
from mlframe.reporting.spec import AnnotationPanelSpec  # noqa: E402


def _planted_optima(n: int, true_optima, seed: int):
    """Build a K-label synthetic where label k's F1-optimal threshold sits near ``true_optima[k]``.

    Positives get probabilities centred above their target threshold, negatives below it, with enough
    separation that the F1-maximising cut lands near the planted value. Base rates vary per label so a
    single global 0.5 cutoff is demonstrably wrong.
    """
    rng = np.random.default_rng(seed)
    K = len(true_optima)
    y = np.zeros((n, K), dtype=np.int8)
    proba = np.zeros((n, K), dtype=np.float64)
    base_rates = np.linspace(0.15, 0.55, K)
    for k, (t_opt, br) in enumerate(zip(true_optima, base_rates)):
        yk = (rng.random(n) < br).astype(np.int8)
        y[:, k] = yk
        # Positives concentrate above t_opt, negatives below -> the F1 optimum sits near t_opt.
        pos = np.clip(t_opt + 0.18 + rng.normal(0, 0.08, n), 0.01, 0.99)
        neg = np.clip(t_opt - 0.18 + rng.normal(0, 0.08, n), 0.01, 0.99)
        proba[:, k] = np.where(yk == 1, pos, neg)
    return y, proba, [f"label{k}" for k in range(K)]


class TestThresholdSweep:
    def test_token_registered(self):
        assert "THRESHOLD_SWEEP" in ALLOWED_MULTILABEL_PANEL_TOKENS

    def test_sweep_shape(self, synth_3label):
        y, p, lbl = synth_3label
        fig = compose_multilabel_figure(y, p, lbl, panels_template="THRESHOLD_SWEEP")
        panel = [pp for row in fig.panels for pp in row if pp is not None][0]
        assert isinstance(panel, HeatmapPanelSpec)
        assert panel.matrix.shape == (3, _SWEEP_N_THRESHOLDS)
        assert len(panel.row_labels) == 3
        # Each row label carries the per-label optimal threshold marker.
        assert all("@t*=" in rl for rl in panel.row_labels)

    def test_sweep_f1_matches_sklearn(self):
        rng = np.random.default_rng(0)
        n, K = 2000, 3
        y = rng.integers(0, 2, (n, K))
        proba = np.clip(y * 0.5 + rng.random((n, K)) * 0.6, 0, 1)
        thresholds = np.linspace(0.0, 1.0, _SWEEP_N_THRESHOLDS)
        f1 = _per_label_f1_sweep(y, proba, thresholds)
        for k in range(K):
            for j in range(0, _SWEEP_N_THRESHOLDS, 17):
                ref = f1_score(y[:, k], (proba[:, k] >= thresholds[j]).astype(int), zero_division=0)
                assert abs(ref - f1[k, j]) < 1e-9

    def test_sweep_empty_labels_annotates(self):
        y = np.zeros((10, 0))
        p = np.zeros((10, 0))
        panel = _threshold_sweep_panel(y, p, [])
        assert isinstance(panel, AnnotationPanelSpec)

    def test_sweep_njit_kernel_matches_numpy_fallback(self):
        """The njit fast path MUST be bit-identical to the numpy fallback (selection-altering index)."""
        from mlframe.reporting.charts._threshold_sweep_kernel import (
            _NUMBA_AVAILABLE, _f1_sweep_numba, _f1_sweep_numpy,
        )
        if not _NUMBA_AVAILABLE:
            pytest.skip("numba unavailable; only the numpy fallback path exists")
        rng = np.random.default_rng(7)
        n, K = 5000, 4
        y = rng.integers(0, 2, (n, K)).astype(np.uint8)
        proba = np.clip(y * 0.5 + rng.random((n, K)) * 0.6, 0.0, 1.0)
        kn = _f1_sweep_numba(np.ascontiguousarray(y), np.ascontiguousarray(proba), _SWEEP_N_THRESHOLDS)
        kp = _f1_sweep_numpy(np.ascontiguousarray(y), np.ascontiguousarray(proba), _SWEEP_N_THRESHOLDS)
        assert np.array_equal(kn, kp)

    def test_biz_val_threshold_sweep_recovers_per_label_optimum(self):
        """The F1-optimal threshold per label MUST be recovered near its planted value.

        Three labels with planted optima at 0.30 / 0.50 / 0.70 and differing base rates. The sweep's
        argmax-F1 threshold for each label must land within 0.10 of the planted optimum -- proving the
        per-label cutoff (not a single global 0.5) is what the heatmap surfaces.
        """
        true_opt = [0.30, 0.50, 0.70]
        y, proba, lbl = _planted_optima(8000, true_opt, seed=11)
        thresholds = np.linspace(0.0, 1.0, _SWEEP_N_THRESHOLDS)
        f1 = _per_label_f1_sweep(y, proba, thresholds)
        recovered = thresholds[np.argmax(f1, axis=1)]
        for k, t_opt in enumerate(true_opt):
            assert abs(recovered[k] - t_opt) <= 0.10, (
                f"label{k}: recovered t*={recovered[k]:.3f} not within 0.10 of planted {t_opt}"
            )
        # The recovered optima must genuinely differ -> a single global cutoff would be wrong.
        assert recovered.max() - recovered.min() >= 0.25

    def test_sweep_render_matplotlib(self, synth_3label, tmp_path):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl, panels_template="THRESHOLD_SWEEP")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"), str(tmp_path / "sweep"))
        assert os.path.exists(tmp_path / "sweep.png")

    def test_sweep_render_plotly(self, synth_3label, tmp_path):
        y, p, lbl = synth_3label
        spec = compose_multilabel_figure(y, p, lbl, panels_template="THRESHOLD_SWEEP")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("plotly[html]"), str(tmp_path / "sweep"))
        assert os.path.exists(tmp_path / "sweep.html")
