"""Numeric-equivalence regression for the vectorized reporting panels.

The confusion / calibration / co-occurrence / cardinality panels were
converted from per-sample Python loops to bincount / GEMM tallies
(50x / 1.8x / 5.5x / 14x on 200k rows). The existing chart tests only
assert the returned spec TYPE and shape; these assert the NUMERIC matrices
against independent loop-based references, so a future "optimization" that
silently changes the math is caught.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.multiclass import _confusion_panel, _calib_grid_panel
from mlframe.reporting.charts.multilabel import _cooccurrence_panel, _cardinality_panel


# ----- independent reference implementations (the pre-vectorization math) -----
def _confusion_ref(y_true, y_pred, K):
    m = np.zeros((K, K), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        m[int(t), int(p)] += 1.0
    rs = m.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return m / rs


def _calib_ref(y_true, y_proba, K, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for k in range(K):
        proba_k = y_proba[:, k]
        true_k = (np.asarray(y_true) == k).astype(np.float64)
        bin_idx = np.clip(np.digitize(proba_k, edges[1:-1]), 0, n_bins - 1)
        observed = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                observed[b] = float(true_k[mask].mean())
        out.append(observed)
    return out


def _cooc_ref(y_true, y_proba, K):
    y_pred = (y_proba >= 0.5).astype(np.int8)
    m = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        true_mask = y_true[:, i] == 1
        n_true = float(true_mask.sum())
        if n_true == 0:
            continue
        for j in range(K):
            m[i, j] = float(y_pred[true_mask, j].sum()) / n_true
    return m


def _card_ref(y_true, y_proba, K):
    y_pred = (y_proba >= 0.5).astype(np.int8)
    true_card = y_true.sum(axis=1).astype(np.int32)
    pred_card = y_pred.sum(axis=1).astype(np.int32)
    tc = np.zeros(K + 1, dtype=np.int64)
    pc = np.zeros(K + 1, dtype=np.int64)
    for c in true_card:
        if 0 <= c <= K:
            tc[c] += 1
    for c in pred_card:
        if 0 <= c <= K:
            pc[c] += 1
    return tc.astype(np.float64), pc.astype(np.float64)


@pytest.fixture
def mc_data():
    rng = np.random.default_rng(7)
    n, K = 800, 4
    y_true = rng.integers(0, K, size=n)
    proba = rng.random((n, K))
    proba /= proba.sum(axis=1, keepdims=True)
    return y_true, proba, list(range(K)), K


@pytest.fixture
def ml_data():
    rng = np.random.default_rng(11)
    n, K = 800, 5
    y_true = (rng.random((n, K)) < 0.35).astype(np.int8)
    # force one all-false true column to exercise the n_true == 0 branch
    y_true[:, 0] = 0
    proba = rng.random((n, K))
    return y_true, proba, list(range(K)), K


def test_confusion_matches_reference(mc_data):
    y_true, proba, classes, K = mc_data
    y_pred = np.argmax(proba, axis=1)
    spec = _confusion_panel(y_true, proba, classes)
    assert np.allclose(spec.matrix, _confusion_ref(y_true, y_pred, K))


def test_calib_matches_reference(mc_data):
    y_true, proba, classes, K = mc_data
    spec = _calib_grid_panel(y_true, proba, classes)
    # spec.y = (perfect_diagonal, *per_class_series)
    series = spec.y[1:]
    ref = _calib_ref(y_true, proba, K)
    assert len(series) == K
    for got, exp in zip(series, ref):
        assert np.allclose(np.asarray(got), exp, equal_nan=True)


def test_cooccurrence_matches_reference(ml_data):
    y_true, proba, labels, K = ml_data
    spec = _cooccurrence_panel(y_true, proba, labels)
    ref = _cooc_ref(y_true, proba, K)
    assert np.allclose(spec.matrix, ref)
    # the all-false true column (row 0) must stay all-zero, not divide-by-zero
    assert np.all(spec.matrix[0] == 0.0)


def test_compose_warns_on_total_label_mismatch():
    """Positional-int y_true + string classes => every label unseen => empty
    panels. The remap must warn loudly rather than silently render blanks."""
    from mlframe.reporting.charts.multiclass import compose_multiclass_figure

    rng = np.random.default_rng(3)
    n, K = 200, 3
    pos = rng.integers(0, K, n)  # 0..K-1 positional ints, NOT in classes
    proba = rng.random((n, K))
    proba /= proba.sum(axis=1, keepdims=True)
    with pytest.warns(UserWarning, match="none of the .* y_true values matched"):
        compose_multiclass_figure(pos, proba, ["cat", "dog", "bird"],
                                   panels_template="CONFUSION")


def test_cardinality_matches_reference(ml_data):
    y_true, proba, labels, K = ml_data
    spec = _cardinality_panel(y_true, proba, labels)
    tc_exp, pc_exp = _card_ref(y_true, proba, K)
    tc_got, pc_got = spec.values
    assert np.array_equal(np.asarray(tc_got), tc_exp)
    assert np.array_equal(np.asarray(pc_got), pc_exp)
    # histograms must conserve total row count
    assert np.asarray(tc_got).sum() == y_true.shape[0]
    assert np.asarray(pc_got).sum() == y_true.shape[0]
