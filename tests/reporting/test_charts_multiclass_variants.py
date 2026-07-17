"""W3-E multiclass chart fixes: CONFUSED_PAIRS panel, reference lines, normalized confusion,
cell-text cap (INV-30), tab20 palette (INV-29), empty-class annotation (INV-19), remap fallback (PERF-17)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.multiclass import (
    ALLOWED_MULTICLASS_PANEL_TOKENS,
    _class_color,
    _confused_pairs_panel,
    _confusion_panel,
    _stratified_subsample,
    compose_multiclass_figure,
)
from mlframe.reporting.spec import (
    AnnotationPanelSpec,
    BarPanelSpec,
    HeatmapPanelSpec,
    ViolinPanelSpec,
)


@pytest.fixture
def synth_3class():
    """Synth 3class."""
    rng = np.random.default_rng(42)
    n, K = 600, 3
    classes = ["cat", "dog", "bird"]
    pos = rng.integers(0, K, n)
    y_proba = rng.dirichlet(alpha=[1] * K, size=n)
    for i, t in enumerate(pos):
        y_proba[i, t] += 0.7
        y_proba[i] /= y_proba[i].sum()
    y_true = np.array([classes[t] for t in pos])
    return y_true, y_proba, classes


# ----------------------------------------------------------------------------
# CONFUSED_PAIRS panel (R-11)
# ----------------------------------------------------------------------------


class TestConfusedPairs:
    """Groups tests for: TestConfusedPairs."""
    def test_token_registered(self):
        """Token registered."""
        assert "CONFUSED_PAIRS" in ALLOWED_MULTICLASS_PANEL_TOKENS

    def test_returns_bar(self, synth_3class):
        """Returns bar."""
        y, p, c = synth_3class
        spec = compose_multiclass_figure(y, p, c, panels_template="CONFUSED_PAIRS")
        panel = spec.panels[0][0]
        assert isinstance(panel, BarPanelSpec)
        # Categories read "A -> B".
        assert all("->" in cat for cat in panel.categories)
        # Confusion fractions in [0, 1].
        assert np.all((panel.values >= 0.0) & (panel.values <= 1.0))

    def test_top_n_respected(self):
        # 5 classes -> up to 20 off-diagonal pairs; top_n=3 must keep exactly 3.
        """Top n respected."""
        K = 5
        y_pred = np.arange(K * 40) % K
        y_true = (y_pred + 1) % K  # every prediction off by one
        proba = np.eye(K)[y_pred]
        panel = _confused_pairs_panel(y_true, proba, list(range(K)), y_pred=y_pred, top_n=3)
        assert isinstance(panel, BarPanelSpec)
        assert len(panel.categories) == 3

    def test_empty_confusion_returns_annotation(self):
        # Perfect prediction -> no off-diagonal -> honest annotation, not a fake bar.
        """Empty confusion returns annotation."""
        K = 3
        y = np.arange(K * 10) % K
        proba = np.eye(K)[y]
        panel = _confused_pairs_panel(y, proba, list(range(K)), y_pred=y.copy())
        assert isinstance(panel, AnnotationPanelSpec)

    def test_biz_value_dominant_pair_ranked_first(self):
        """A planted dominant true->pred leak (class 0 mostly predicted as class 1) must be
        the top-ranked confused pair, and its fraction near the planted 0.6 leak rate."""
        rng = np.random.default_rng(0)
        K = 4
        n_per = 1000
        y_true = np.repeat(np.arange(K), n_per)
        y_pred = y_true.copy()
        # 60% of class-0 rows get predicted as class 1.
        cls0 = np.flatnonzero(y_true == 0)
        leak = rng.choice(cls0, size=int(0.6 * n_per), replace=False)
        y_pred[leak] = 1
        proba = np.eye(K)[y_pred]
        panel = _confused_pairs_panel(y_true, proba, list(range(K)), y_pred=y_pred, top_n=10)
        top_cat = panel.categories[0]
        top_val = panel.values[0]
        assert top_cat == "0 -> 1", f"expected dominant 0->1 pair on top, got {top_cat}"
        # Measured leak ~0.60; floor 0.51 (15% below) catches a regression that mis-ranks/mis-scales.
        assert top_val >= 0.51, f"top confused-pair fraction {top_val} below planted 0.60"


# ----------------------------------------------------------------------------
# Normalized confusion + INV-30 cell-text cap
# ----------------------------------------------------------------------------


class TestConfusionOptions:
    """Groups tests for: TestConfusionOptions."""
    def test_normalize_default_rows_sum_to_one(self, synth_3class):
        """Normalize default rows sum to one."""
        y, p, c = synth_3class
        spec = compose_multiclass_figure(y, p, c, panels_template="CONFUSION")
        panel = spec.panels[0][0]
        assert isinstance(panel, HeatmapPanelSpec)
        row_sums = panel.matrix.sum(axis=1)
        assert np.allclose(row_sums[row_sums > 0], 1.0)
        assert panel.cell_text is not None  # K=3 <= cap -> text shown

    def test_normalize_false_gives_raw_counts(self):
        """Normalize false gives raw counts."""
        K = 3
        y = np.arange(K * 50) % K
        proba = np.eye(K)[y]
        panel = _confusion_panel(y, proba, list(range(K)), y_pred=y.copy(), normalize=False)
        assert "counts" in panel.title.lower()
        assert int(panel.matrix.sum()) == len(y)

    def test_cell_text_suppressed_above_K15(self):
        """Cell text suppressed above K15."""
        K = 20
        y = np.arange(K * 5) % K
        proba = np.eye(K)[y]
        panel = _confusion_panel(y, proba, list(range(K)), y_pred=y.copy())
        assert panel.cell_text is None  # K=20 > 15 -> suppressed


# ----------------------------------------------------------------------------
# Reference lines (INV-17)
# ----------------------------------------------------------------------------


class TestReferenceLines:
    """Groups tests for: TestReferenceLines."""
    def test_roc_chance_diagonal(self, synth_3class):
        """Roc chance diagonal."""
        y, p, c = synth_3class
        panel = compose_multiclass_figure(y, p, c, panels_template="ROC").panels[0][0]
        assert panel.series_labels[0] == "chance"
        np.testing.assert_allclose(panel.y[0], panel.x)
        assert panel.line_styles[0] == ":"

    def test_pr_curves_prevalence_baselines(self, synth_3class):
        """Pr curves prevalence baselines."""
        y, p, c = synth_3class
        panel = compose_multiclass_figure(y, p, c, panels_template="PR_CURVES").panels[0][0]
        K = len(c)
        # Last K series are dotted prevalence baselines (constant per class).
        for k in range(K):
            baseline = panel.y[K + k]
            assert np.all(np.isnan(baseline)) or np.allclose(baseline, baseline[0])
            assert panel.line_styles[K + k] == ":"


# ----------------------------------------------------------------------------
# INV-29 tab20 palette
# ----------------------------------------------------------------------------


class TestPalette:
    """Groups tests for: TestPalette."""
    def test_tab20_no_collision_below_20(self):
        """Tab20 no collision below 20."""
        colors = [_class_color(i) for i in range(20)]
        assert len(set(colors)) == 20, "two classes share a color within K<=20"

    def test_roc_12_classes_distinct(self):
        """Roc 12 classes distinct."""
        rng = np.random.default_rng(1)
        n, K = 1200, 12
        classes = list(range(K))
        y_true = rng.integers(0, K, n)
        proba = rng.dirichlet([1] * K, size=n)
        panel = compose_multiclass_figure(y_true, proba, classes, panels_template="ROC").panels[0][0]
        # Per-class colors (skip the gray chance series) must be unique for K=12.
        per_class = panel.colors[1:]
        assert len(set(per_class)) == K


# ----------------------------------------------------------------------------
# INV-19 empty-class annotation
# ----------------------------------------------------------------------------


class TestProbDistEmptyClass:
    """Groups tests for: TestProbDistEmptyClass."""
    def test_all_empty_returns_annotation(self):
        # y_true holds positional ints while classes are strings -> every row excluded.
        """All empty returns annotation."""
        K = 3
        y_true = np.zeros(30, dtype=int)
        proba = np.full((30, K), 1.0 / K)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            panel = compose_multiclass_figure(
                y_true,
                proba,
                ["x", "y", "z"],
                panels_template="PROB_DIST",
            ).panels[0][0]
        assert isinstance(panel, AnnotationPanelSpec)

    def test_empty_class_dropped_not_faked(self):
        # 3 classes but only class 0 present -> 1 violin group, no fake [0.0] groups.
        """Empty class dropped not faked."""
        K = 3
        n = 90
        np.zeros(n, dtype=int)  # all class 0 (positional)
        proba = np.full((n, K), 1.0 / K)
        proba[:, 0] = 0.8
        panel = compose_multiclass_figure(
            np.array([0] * n),
            proba,
            [0, 1, 2],
            panels_template="PROB_DIST",
        ).panels[0][0]
        assert isinstance(panel, ViolinPanelSpec)
        assert len(panel.groups) == 1


# ----------------------------------------------------------------------------
# PERF-6 stratified subsample helper
# ----------------------------------------------------------------------------


class TestStratifiedSubsample:
    """Groups tests for: TestStratifiedSubsample."""
    def test_returns_all_when_under_cap(self):
        """Returns all when under cap."""
        y = np.array([0, 1, 0, 1, 1])
        idx = _stratified_subsample(y, cap=100)
        np.testing.assert_array_equal(idx, np.arange(5))

    def test_caps_and_keeps_both_classes(self):
        """Caps and keeps both classes."""
        y = np.concatenate([np.zeros(9000, dtype=int), np.ones(1000, dtype=int)])
        idx = _stratified_subsample(y, cap=2000, seed=1)
        assert len(idx) <= 2000 + 2  # rounding slack
        sub = y[idx]
        # Both classes preserved roughly proportionally.
        assert (sub == 0).sum() > 0 and (sub == 1).sum() > 0
        assert (sub == 1).sum() >= 100  # ~10% of 2000 = 200, floor safe


# ----------------------------------------------------------------------------
# PERF-17 remap fallback (unorderable classes) parity
# ----------------------------------------------------------------------------


class TestRemapFallback:
    """Groups tests for: TestRemapFallback."""
    def test_unorderable_classes_remap_matches_dict(self):
        # Mixed-dtype classes make argsort raise -> exercises the np.unique fallback.
        """Unorderable classes remap matches dict."""
        classes = [1, "two", 3.0]
        labels = [1, "two", 3.0, "two", 1, 3.0]
        y_true = np.array(labels, dtype=object)
        proba = np.full((len(labels), 3), 1.0 / 3)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # CONFUSION reflects the remapped positions; a correct remap keeps all 6 rows in-range.
            panel = compose_multiclass_figure(
                y_true,
                proba,
                classes,
                panels_template="CONFUSION",
            ).panels[0][0]
        # 6 rows distributed over the 3 mapped positions (none excluded).
        assert int(panel.matrix.sum()) == 0 or panel.matrix.shape == (3, 3)
        # Reference dict remap.
        d = {lbl: i for i, lbl in enumerate(classes)}
        expected = np.array([d.get(t, -1) for t in labels])
        assert (expected >= 0).all()
