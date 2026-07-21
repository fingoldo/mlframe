"""Tests for the engineered-pair separability scatter (charts/engineered_separability.py).

Covers: the Fisher 2-D score kernel (degenerate one-class -> 0, invariance to point order), narrow column pull +
seeded subsample, top-2 feature pick by importance, spec shape, and biz_value -- a cleanly separable 2-blob synthetic
must score well above a fully-overlapping same-mean control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import pytest

from mlframe.reporting.charts.engineered_separability import (
    compose_separability_figure,
    separability_panel,
    separability_score,
)
from mlframe.reporting.spec import FigureSpec, ScatterPanelSpec


def _flat(fig: FigureSpec):
    """Helper: Flat."""
    return [p for row in fig.panels for p in row if p is not None]


# ----------------------------------------------------------------------------
# Unit: score kernel
# ----------------------------------------------------------------------------


def test_single_class_scores_zero():
    """Single class scores zero."""
    z2 = np.random.default_rng(0).random((500, 2))
    y = np.ones(500)  # only one class present
    assert separability_score(z2, y) == 0.0


def test_score_is_order_invariant():
    """Score is order invariant."""
    rng = np.random.default_rng(1)
    z2 = rng.random((2000, 2))
    y = rng.integers(0, 2, size=2000).astype(float)
    s1 = separability_score(z2, y)
    perm = rng.permutation(2000)
    s2 = separability_score(z2[perm], y[perm])
    assert abs(s1 - s2) < 1e-8


def test_overlapping_classes_score_near_zero():
    """Overlapping classes score near zero."""
    rng = np.random.default_rng(2)
    z2 = rng.standard_normal((4000, 2))  # both classes share the same distribution
    y = rng.integers(0, 2, size=4000).astype(float)
    assert separability_score(z2, y) < 0.05


# ----------------------------------------------------------------------------
# Unit: panel / compose
# ----------------------------------------------------------------------------


def test_panel_subsamples_and_labels():
    """Panel subsamples and labels."""
    rng = np.random.default_rng(3)
    n = 20_000
    X = pd.DataFrame({"a": rng.random(n), "b": rng.random(n)})
    y = rng.integers(0, 2, size=n).astype(float)
    panel = separability_panel(X, y, ["a", "b"], sample=5000)
    assert isinstance(panel, ScatterPanelSpec)
    assert panel.x.shape[0] == 5000 and panel.y.shape[0] == 5000
    assert "Fisher J=" in panel.title
    assert panel.xlabel == "a" and panel.ylabel == "b"


def test_panel_raises_clean_error_on_x_y_length_mismatch():
    """A ``y`` shorter than ``X`` (e.g. an ensembling COARSE-gate rescoring call that passes the full-frame X
    alongside a subset y) used to hit ``rng.choice(n, ...)`` sampling indices up to ``len(X)-1`` then index ``yv``
    (length ``len(y)``) with them, raising a bare ``IndexError`` deep inside numpy fancy indexing. It now raises a
    clear ``ValueError`` at the function boundary instead.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.random(2000), "b": rng.random(2000)})
    y = rng.integers(0, 2, size=1800).astype(float)
    with pytest.raises(ValueError, match="length mismatch"):
        separability_panel(X, y, ["a", "b"], sample=200, seed=0)


def test_panel_survives_text_feature_column():
    """A free-text feature column (e.g. a fuzz-injected 'text_0' skills-keyword column) can't be cast to float64
    -- CatBoost's own text-feature detection rejects the symmetric cast too, raising "could not convert string
    to float" (caught live via a fuzz combo with a text feature column reaching this scatter). Must label-encode
    it via np.unique codes instead of crashing, mirroring error_analysis.py's non-numeric column handling."""
    rng = np.random.default_rng(0)
    n = 500
    texts = rng.choice(["python cloud swift", "quantum nlp robotics", "rust wasm edge"], size=n)
    X = pd.DataFrame({"text_0": texts, "num_0": rng.random(n)})
    y = rng.integers(0, 2, size=n).astype(float)
    panel = separability_panel(X, y, ["text_0", "num_0"], sample=200, seed=0)
    assert isinstance(panel, ScatterPanelSpec)
    assert np.isfinite(panel.x).all()  # label-encoded, not NaN-substituted (it's a plain string column, not embedding)


def test_compose_picks_top2_by_importance():
    """Compose picks top2 by importance."""
    rng = np.random.default_rng(4)
    n = 3000
    X = pd.DataFrame({"a": rng.random(n), "b": rng.random(n), "c": rng.random(n)})
    y = rng.integers(0, 2, size=n).astype(float)
    fig = compose_separability_figure(X, y, feature_importances=np.array([0.1, 0.9, 0.5]))
    panel = _flat(fig)[0]
    # Importances rank b (0.9) then c (0.5) as the top-2.
    assert panel.xlabel == "b" and panel.ylabel == "c"


# ----------------------------------------------------------------------------
# biz_value: separable blobs beat overlapping control
# ----------------------------------------------------------------------------


def test_biz_val_separable_blobs_beat_overlap_control():
    """A cleanly linearly-separable 2-blob synthetic must score >= 20.0 (measured Fisher J ~32) AND at least 3x the
    fully-overlapping same-mean control. A regression that mis-computes the pooled within-class scatter or the class
    means collapses J toward the overlapping baseline.
    """
    rng = np.random.default_rng(42)
    n = 2000
    half = n // 2
    z_sep = np.vstack(
        [
            rng.standard_normal((half, 2)) + np.array([0.0, 0.0]),
            rng.standard_normal((half, 2)) + np.array([4.0, 4.0]),
        ]
    )
    y = np.concatenate([np.zeros(half), np.ones(half)])
    sep = separability_score(z_sep, y)

    z_ctrl = rng.standard_normal((n, 2))  # same mean for both classes
    ctrl = separability_score(z_ctrl, y)

    assert sep >= 20.0, sep
    assert sep >= 3.0 * max(ctrl, 1e-6), (sep, ctrl)
