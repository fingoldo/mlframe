"""Large-K overlay auto-switch for multiclass / multilabel quality figures.

Past the per-class/label overlay threshold the ROC / PR / reliability panels render only the worst-by-AUC classes plus a
macro-average instead of all K spaghetti curves -- both a speed win (fewer sklearn fits) and a readability win. These tests
pin the behaviour: worst-class selection picks the genuinely hard classes, the macro-avg is present, and K <= threshold is
unchanged (every class still renders).
"""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.multiclass import compose_multiclass_figure
from mlframe.reporting.charts.multilabel import compose_multilabel_figure
from mlframe.reporting.spec import LinePanelSpec


def _flatten(grid):
    """Helper: Flatten."""
    return [p for row in grid for p in row]


def _find_line_panel(fig, title_contains):
    """Helper: Find line panel."""
    for p in _flatten(fig.panels):
        if isinstance(p, LinePanelSpec) and title_contains in p.title:
            return p
    raise AssertionError(f"no LinePanelSpec with {title_contains!r} in titles {[getattr(p, 'title', '') for p in _flatten(fig.panels)]}")


# ---------------------------------------------------------------------------
# Synthetic with deliberately-hard classes (low AUC) at known indices.
# ---------------------------------------------------------------------------


def _multiclass_with_hard_classes(n=60_000, K=40, hard=(3, 11, 27, 38), seed=0):
    """Strong signal for every class except ``hard`` ones, whose proba column is near-random (low one-vs-rest AUC)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, size=n)
    logits = rng.normal(0.0, 1.0, size=(n, K))
    logits[np.arange(n), y] += 3.0
    for h in hard:
        # Wipe the class-h signal: its column carries no information, so its OVR AUC collapses toward 0.5.
        logits[:, h] = rng.normal(0.0, 1.0, size=n)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba = e / e.sum(axis=1, keepdims=True)
    return y, proba, list(range(K))


def _multilabel_with_hard_labels(n=60_000, K=40, hard=(2, 14, 33), seed=1):
    """Helper: Multilabel with hard labels."""
    rng = np.random.default_rng(seed)
    yt = (rng.random((n, K)) < 0.3).astype(np.int8)
    proba = np.clip(0.15 + 0.7 * yt + rng.normal(0.0, 0.18, size=(n, K)), 0.0, 1.0)
    for h in hard:
        proba[:, h] = rng.random(n)  # uninformative column -> low AUC
    return yt, proba, [f"lbl{k}" for k in range(K)]


# ---------------------------------------------------------------------------
# Multiclass
# ---------------------------------------------------------------------------


def test_multiclass_smallK_renders_all_classes_unchanged():
    """At K <= threshold every class still renders, no macro-avg, title unchanged (happy path is untouched)."""
    rng = np.random.default_rng(3)
    n, K = 4000, 6
    y = rng.integers(0, K, n)
    logits = rng.normal(0, 1, (n, K))
    logits[np.arange(n), y] += 2.0
    e = np.exp(logits - logits.max(1, keepdims=True))
    proba = e / e.sum(1, keepdims=True)
    fig = compose_multiclass_figure(y, proba, list(range(K)), panels_template="ROC CALIB_GRID PR_CURVES")
    roc = _find_line_panel(fig, "Per-class ROC")
    assert "of" not in roc.title and "macro" not in roc.title
    # chance + K class curves, no macro.
    assert len(roc.y) == K + 1
    assert not any("macro" in s for s in roc.series_labels)


def test_multiclass_largeK_switches_to_topN_plus_macro():
    """Multiclass largeK switches to topN plus macro."""
    y, proba, classes = _multiclass_with_hard_classes()
    fig = compose_multiclass_figure(y, proba, classes, panels_template="ROC CALIB_GRID PR_CURVES", overlay_max_classes=12, overlay_top_n=8)
    roc = _find_line_panel(fig, "Per-class ROC")
    assert "8 of 40" in roc.title
    macro_labels = [s for s in roc.series_labels if "macro" in s]
    assert len(macro_labels) == 1, "exactly one macro-avg curve expected on the ROC overlay"
    # chance + 8 classes + macro
    assert len(roc.y) == 8 + 2


def test_multiclass_largeK_selects_genuinely_worst_classes():
    """The deliberately-hard (low-AUC) classes must appear among the drawn worst-N curves."""
    hard = (3, 11, 27, 38)
    y, proba, classes = _multiclass_with_hard_classes(hard=hard)
    fig = compose_multiclass_figure(y, proba, classes, panels_template="ROC", overlay_max_classes=12, overlay_top_n=8)
    roc = _find_line_panel(fig, "Per-class ROC")
    drawn = {int(s.split(" ")[0]) for s in roc.series_labels if s and s[0].isdigit()}
    for h in hard:
        assert h in drawn, f"hard class {h} (low AUC) must be among the worst-N drawn classes; got {sorted(drawn)}"


def test_multiclass_macro_present_on_all_overlay_panels():
    """Multiclass macro present on all overlay panels."""
    y, proba, classes = _multiclass_with_hard_classes()
    fig = compose_multiclass_figure(y, proba, classes, panels_template="ROC PR_CURVES CALIB_GRID", overlay_max_classes=12, overlay_top_n=8)
    for title in ("Per-class ROC", "Per-class PR", "Per-class reliability"):
        panel = _find_line_panel(fig, title)
        assert any("macro" in s for s in panel.series_labels), f"{title} missing macro-avg"


# ---------------------------------------------------------------------------
# Multilabel
# ---------------------------------------------------------------------------


def test_multilabel_smallK_renders_all_labels_unchanged():
    """Multilabel smallK renders all labels unchanged."""
    rng = np.random.default_rng(7)
    n, K = 4000, 6
    yt = (rng.random((n, K)) < 0.3).astype(np.int8)
    proba = np.clip(0.15 + 0.7 * yt + rng.normal(0, 0.18, (n, K)), 0, 1)
    fig = compose_multilabel_figure(yt, proba, [f"l{k}" for k in range(K)], panels_template="ROC CALIB_GRID")
    roc = _find_line_panel(fig, "Per-label ROC")
    assert "of" not in roc.title and "macro" not in roc.title
    assert len(roc.y) == K + 1
    assert not any("macro" in s for s in roc.series_labels)


def test_multilabel_largeK_switches_to_topN_plus_macro_and_picks_worst():
    """Multilabel largeK switches to topN plus macro and picks worst."""
    hard = (2, 14, 33)
    yt, proba, labels = _multilabel_with_hard_labels(hard=hard)
    fig = compose_multilabel_figure(yt, proba, labels, panels_template="ROC CALIB_GRID", overlay_max_labels=12, overlay_top_n=8)
    roc = _find_line_panel(fig, "Per-label ROC")
    assert "8 of 40" in roc.title
    assert sum("macro" in s for s in roc.series_labels) == 1
    drawn = {s.split(" ")[0] for s in roc.series_labels if s.startswith("lbl")}
    for h in hard:
        assert f"lbl{h}" in drawn, f"hard label lbl{h} must be drawn; got {sorted(drawn)}"
    calib = _find_line_panel(fig, "Per-label reliability")
    assert any("macro" in s for s in calib.series_labels)


def test_overlay_top_n_param_controls_curve_count():
    """Overlay top n param controls curve count."""
    y, proba, classes = _multiclass_with_hard_classes()
    fig = compose_multiclass_figure(y, proba, classes, panels_template="ROC", overlay_max_classes=12, overlay_top_n=5)
    roc = _find_line_panel(fig, "Per-class ROC")
    assert "5 of 40" in roc.title
    assert len(roc.y) == 5 + 2  # chance + 5 + macro
