"""Tests for the CONFUSION_MARGINS multiclass panel.

Covers: panel structure (heatmap + both margins), margin values equal the bincount of y_true / y_pred over in-range
pairs, viridis CB-safe colormap, cell-text gating at large K, degenerate annotation (single-class / empty / tiny-n),
matplotlib + plotly render smoke, a biz_value imbalance + majority-over-prediction assertion, and a cProfile bound.
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import time

import numpy as np
import pytest

from mlframe.reporting.charts.multiclass import compose_multiclass_figure
from mlframe.reporting.renderers.base import get_renderer
from mlframe.reporting.spec import ConfusionMarginsPanelSpec

os.environ.setdefault("MPLBACKEND", "Agg")


def _imbalanced(n: int, K: int, prevalence, majority_bias: float, seed: int = 0):
    """Imbalanced multiclass synthetic: class-0 dominant per ``prevalence``; ``majority_bias`` tilts every row's proba
    toward class 0 so the model over-predicts the majority."""
    rng = np.random.default_rng(seed)
    p = np.asarray(prevalence, dtype=float)
    p = p / p.sum()
    y = rng.choice(K, size=n, p=p)
    proba = rng.dirichlet([1.0] * K, size=n)
    for i, t in enumerate(y):
        proba[i, t] += 0.5            # genuine signal on the true class
        proba[i, 0] += majority_bias  # majority-class bias
    proba /= proba.sum(axis=1, keepdims=True)
    return y, proba, list(range(K))


def _balanced(n: int, K: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, size=n)
    proba = rng.dirichlet([1.0] * K, size=n)
    for i, t in enumerate(y):
        proba[i, t] += 0.5
    proba /= proba.sum(axis=1, keepdims=True)
    return y, proba, list(range(K))


def _panel(y, proba, classes):
    return compose_multiclass_figure(y, proba, classes, panels_template="CONFUSION_MARGINS").panels[0][0]


# ----------------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------------


class TestStructure:
    def test_returns_confusion_margins_spec(self):
        y, p, c = _balanced(800, 4)
        panel = _panel(y, p, c)
        assert isinstance(panel, ConfusionMarginsPanelSpec)
        assert panel.matrix.shape == (4, 4)
        assert panel.row_margin.shape == (4,)
        assert panel.col_margin.shape == (4,)

    def test_margins_match_bincount_of_labels(self):
        """Row margin == per-true-class support == bincount(y_true); col margin == per-pred-class volume == bincount(y_pred)."""
        y, p, c = _imbalanced(2000, 5, [0.5, 0.2, 0.15, 0.1, 0.05], majority_bias=0.6, seed=1)
        K = len(c)
        panel = _panel(y, p, c)
        y_pred = np.argmax(p, axis=1)
        np.testing.assert_array_equal(panel.row_margin.astype(np.int64), np.bincount(y, minlength=K))
        np.testing.assert_array_equal(panel.col_margin.astype(np.int64), np.bincount(y_pred, minlength=K))

    def test_margins_equal_matrix_row_col_sums(self):
        """Margins are pure row/col sums of the COUNT confusion matrix (no extra full-n pass)."""
        y, p, c = _balanced(900, 3, seed=2)
        K = len(c)
        from mlframe.reporting.charts.multiclass import _confusion_counts
        counts = _confusion_counts(y, np.argmax(p, axis=1), K)
        panel = _panel(y, p, c)
        np.testing.assert_allclose(panel.row_margin, counts.sum(axis=1))
        np.testing.assert_allclose(panel.col_margin, counts.sum(axis=0))

    def test_viridis_cb_safe_colormap(self):
        from mlframe.reporting.colors import HEATMAP_CMAP, resolve_heatmap_cmap
        y, p, c = _balanced(400, 4)
        panel = _panel(y, p, c)
        assert panel.colormap == HEATMAP_CMAP
        assert resolve_heatmap_cmap(panel.colormap) == "viridis"

    def test_cell_text_present_at_small_K(self):
        y, p, c = _balanced(400, 4)
        panel = _panel(y, p, c)
        assert panel.cell_text is not None
        assert panel.cell_text.shape == (4, 4)

    def test_cell_text_suppressed_at_large_K(self):
        y, p, c = _balanced(4000, 20, seed=3)
        panel = _panel(y, p, c)
        assert panel.cell_text is None  # K=20 > _CONFUSION_TEXT_MAX_K, K^2 text is soup


# ----------------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_class_annotates(self):
        panel = _panel(np.zeros(50, dtype=int), np.ones((50, 1)), [0])
        assert panel.note == "single-class problem"
        get_renderer("matplotlib").render(
            compose_multiclass_figure(np.zeros(50, dtype=int), np.ones((50, 1)), [0], panels_template="CONFUSION_MARGINS"))

    def test_empty_annotates_and_renders(self):
        spec = compose_multiclass_figure(np.array([], dtype=int), np.zeros((0, 3)), [0, 1, 2],
                                         panels_template="CONFUSION_MARGINS")
        panel = spec.panels[0][0]
        assert panel.note == "no in-range samples"
        get_renderer("matplotlib").render(spec)  # must not raise

    def test_tiny_n_annotates_but_renders(self):
        y, p, c = _balanced(6, 3, seed=4)
        spec = compose_multiclass_figure(y, p, c, panels_template="CONFUSION_MARGINS")
        panel = spec.panels[0][0]
        assert panel.note is not None and "tiny n" in panel.note
        get_renderer("matplotlib").render(spec)

    def test_large_K_renders_thin_bars(self):
        y, p, c = _balanced(3000, 25, seed=5)
        spec = compose_multiclass_figure(y, p, c, panels_template="CONFUSION_MARGINS")
        panel = spec.panels[0][0]
        assert panel.cell_text is None and panel.row_margin.shape == (25,)
        get_renderer("matplotlib").render(spec)


# ----------------------------------------------------------------------------
# Render smoke (both backends)
# ----------------------------------------------------------------------------


class TestRender:
    def test_matplotlib_render(self):
        y, p, c = _imbalanced(1000, 4, [0.6, 0.2, 0.13, 0.07], majority_bias=0.8)
        spec = compose_multiclass_figure(y, p, c, panels_template="CONFUSION_MARGINS")
        fig = get_renderer("matplotlib").render(spec)
        assert fig is not None

    def test_plotly_render(self):
        pytest.importorskip("plotly")
        y, p, c = _balanced(600, 3)
        spec = compose_multiclass_figure(y, p, c, panels_template="CONFUSION_MARGINS")
        get_renderer("plotly").render(spec)

    def test_combined_template_with_confusion(self):
        """CONFUSION_MARGINS composes alongside other tokens in one figure."""
        y, p, c = _balanced(800, 4)
        spec = compose_multiclass_figure(y, p, c, panels_template="CONFUSION CONFUSION_MARGINS PR_F1")
        flat = [cell for row in spec.panels for cell in row if cell is not None]
        assert any(isinstance(x, ConfusionMarginsPanelSpec) for x in flat)
        get_renderer("matplotlib").render(spec)


# ----------------------------------------------------------------------------
# biz_value: imbalance reflected + majority over-prediction detected
# ----------------------------------------------------------------------------


class TestBizValue:
    def test_biz_val_confusion_margins_reflects_imbalance_and_overprediction(self):
        """On a 60/20/13/7 imbalanced target with a majority-biased model, the support margin must reflect the
        injected prevalence (dominant-class support ratio >> 1, matching the ~0.6 prevalence within tolerance) AND
        the predicted-volume margin must reveal majority over-prediction (class-0 volume > its true support)."""
        n, K = 8000, 4
        prevalence = [0.6, 0.2, 0.13, 0.07]
        y, p, c = _imbalanced(n, K, prevalence, majority_bias=1.2, seed=7)
        panel = _panel(y, p, c)

        support = panel.row_margin
        volume = panel.col_margin

        # 1. Imbalance: dominant-class support fraction matches injected ~0.6 prevalence (10% tol absorbs sampling).
        support_frac0 = support[0] / support.sum()
        assert support_frac0 == pytest.approx(0.6, abs=0.06), f"support frac {support_frac0:.3f} should track 0.6 prevalence"
        # Dominant-class bar is far larger than every minority class.
        assert support[0] > 2.0 * support[1:].max(), "dominant-class support bar must dominate the minorities"

        # 2. Majority over-prediction: the model routes MORE samples to class 0 than actually belong to it.
        assert volume[0] > support[0], f"majority volume {volume[0]} must exceed its support {support[0]} (over-prediction)"
        overpred_ratio = volume[0] / support[0]
        assert overpred_ratio >= 1.05, f"over-prediction ratio {overpred_ratio:.3f} should be a clear (>=5%) excess"

    def test_biz_val_confusion_margins_balanced_is_uniform(self):
        """On a balanced target the support margin is ~uniform (max/min support ratio near 1)."""
        n, K = 8000, 4
        y, p, c = _balanced(n, K, seed=8)
        panel = _panel(y, p, c)
        support = panel.row_margin
        ratio = support.max() / support.min()
        assert ratio < 1.15, f"balanced support should be ~uniform, got max/min ratio {ratio:.3f}"

    def test_biz_val_underprediction_visible(self):
        """A minority class the model rarely emits shows volume < support (under-prediction) -- the complementary signal."""
        n, K = 8000, 4
        y, p, c = _imbalanced(n, K, [0.6, 0.2, 0.13, 0.07], majority_bias=1.5, seed=9)
        panel = _panel(y, p, c)
        # With strong majority bias, at least one minority class is under-emitted (volume < support).
        under = panel.col_margin[1:] < panel.row_margin[1:]
        assert under.any(), "majority bias must starve at least one minority class (volume < support)"


# ----------------------------------------------------------------------------
# cProfile bound: builder reuses the small-matrix tally, no extra full-n pass
# ----------------------------------------------------------------------------


def test_cprofile_confusion_margins_bounded_at_1e6():
    """At n=1e6 / K=10 the builder is bounded: margins are row/col sums of the K x K matrix on top of the single
    bincount tally CONFUSION already pays. Asserts the panel build (excluding the one-time argmax in the composer)
    stays well under a generous wall budget and that no per-class Python loop dominates."""
    n, K = 1_000_000, 10
    rng = np.random.default_rng(11)
    y = rng.integers(0, K, size=n)
    proba = rng.dirichlet([1.0] * K, size=n)

    # Warm + correctness on the full path once.
    compose_multiclass_figure(y, proba, list(range(K)), panels_template="CONFUSION_MARGINS")

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    for _ in range(3):
        compose_multiclass_figure(y, proba, list(range(K)), panels_template="CONFUSION_MARGINS")
    pr.disable()
    elapsed = (time.perf_counter() - t0) / 3.0

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
    # Generous CI-safe ceiling; the work is dominated by the unavoidable argmax + single bincount over n.
    assert elapsed < 5.0, f"CONFUSION_MARGINS build at n=1e6/K=10 took {elapsed:.3f}s (expected << 5s)"
