"""Tests for ``mlframe.metrics.core.fast_roc_curve`` -- our own-implementation replacement for
``sklearn.metrics.roc_curve``.

Three-part coverage per the project rule for new features:
  * unit edge cases (empty, single-class, all-ties, perfect separation, sample_weight, {-1,1} labels);
  * an sklearn-equivalence test over random + tie-heavy datasets (the one place a sklearn import is legitimate),
    including ``np.trapz(tpr, fpr) ~= fast_roc_auc``;
  * a cProfile smoke on a large input (see ``test_fast_roc_curve_large_profile``); profiling numbers are recorded
    in the module docstring below.

cProfile (n=1_000_000, continuous scores, this host, 3.14 CPython, GPU-argsort path active):
    steady-state ~115 ms/call. Dominant cost is the shared descending argsort in ``_argsort_desc_for_metrics``
    (here the cupy GPU radix sort + D2H copy, exactly the sort every AUC kernel already pays); the njit sweep
    ``_roc_curve_kernel`` is a single O(n) pass and is NOT the hotspot. The largest mlframe-side pure-Python cost
    is ``_roc_optimal_idxs`` (~32 ms cumtime: two O(n) ``np.diff`` passes over the ~n distinct-score vertices),
    already vectorised numpy -- no actionable further optimisation without changing sklearn-equivalent output.
    The sort is at the algorithmic floor (see ``fast_roc_auc`` docstring's rejected sort-fusion benches).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.core import fast_roc_curve, fast_roc_auc

# ---------------------------------------------------------------------------
# Unit / edge cases (no sklearn)
# ---------------------------------------------------------------------------


def test_empty_input():
    """Empty input."""
    fpr, tpr, thr = fast_roc_curve(np.array([]), np.array([]))
    assert thr[0] == np.inf
    assert np.isnan(fpr).all() and np.isnan(tpr).all()


def test_inf_threshold_anchor_and_monotone():
    """Inf threshold anchor and monotone."""
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, thr = fast_roc_curve(y, s)
    assert thr[0] == np.inf
    # remaining thresholds strictly decreasing
    assert np.all(np.diff(thr[1:]) < 0)
    # curve starts at origin
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    # fpr / tpr monotone non-decreasing
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)


def test_perfect_separation():
    """Perfect separation."""
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    fpr, tpr, _thr = fast_roc_curve(y, s)
    # perfect classifier reaches tpr=1 at fpr=0
    assert np.isclose(np.trapezoid(tpr, fpr), 1.0)


def test_all_ties_single_threshold():
    # Every score identical -> a single swept threshold; curve is the (0,0)->(1,1) diagonal.
    """All ties single threshold."""
    y = np.array([0, 1, 0, 1])
    s = np.full(4, 0.5)
    fpr, tpr, thr = fast_roc_curve(y, s)
    assert thr[0] == np.inf
    assert np.allclose(fpr, [0.0, 1.0])
    assert np.allclose(tpr, [0.0, 1.0])


def test_single_class_all_positive_gives_nan_fpr():
    """Single class all positive gives nan fpr."""
    y = np.array([1, 1, 1])
    s = np.array([0.2, 0.5, 0.9])
    fpr, tpr, _thr = fast_roc_curve(y, s)
    assert np.isnan(fpr).all()
    assert not np.isnan(tpr).any()


def test_single_class_all_negative_gives_nan_tpr():
    """Single class all negative gives nan tpr."""
    y = np.array([0, 0, 0])
    s = np.array([0.2, 0.5, 0.9])
    fpr, tpr, _thr = fast_roc_curve(y, s)
    assert np.isnan(tpr).all()
    assert not np.isnan(fpr).any()


def test_minus_one_plus_one_labels_treated_as_binary():
    """Minus one plus one labels treated as binary."""
    y = np.array([-1, -1, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, _ = fast_roc_curve(y, s)
    fpr01, tpr01, _ = fast_roc_curve(np.array([0, 0, 1, 1]), s)
    assert np.allclose(fpr, fpr01) and np.allclose(tpr, tpr01)


def test_sample_weight_shifts_curve():
    """Sample weight shifts curve."""
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    w = np.array([1.0, 5.0, 1.0, 1.0])
    fpr_w, _tpr_w, _ = fast_roc_curve(y, s, sample_weight=w)
    fpr_u, _tpr_u, _ = fast_roc_curve(y, s)
    # up-weighting a negative changes the fpr axis
    assert not np.allclose(fpr_w, fpr_u)


def test_2d_score_uses_last_column():
    """2d score uses last column."""
    y = np.array([0, 0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]])
    fpr, tpr, _ = fast_roc_curve(y, proba)
    fpr1, tpr1, _ = fast_roc_curve(y, proba[:, -1])
    assert np.allclose(fpr, fpr1) and np.allclose(tpr, tpr1)


# ---------------------------------------------------------------------------
# sklearn equivalence (the ONE sanctioned sklearn import)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tie_heavy", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_equivalence_vs_sklearn(tie_heavy, seed):
    """Equivalence vs sklearn."""
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(seed)
    n = int(rng.integers(20, 3000))
    y = rng.integers(0, 2, n)
    if y.sum() == 0 or y.sum() == n:
        y[0], y[-1] = 0, 1  # guarantee both classes
    s = rng.integers(0, 7, n).astype(float) if tie_heavy else rng.random(n)

    fpr, tpr, thr = fast_roc_curve(y, s)
    fpr_sk, tpr_sk, thr_sk = sk.roc_curve(y, s)
    np.testing.assert_allclose(fpr, fpr_sk)
    np.testing.assert_allclose(tpr, tpr_sk)
    assert thr[0] == np.inf and thr_sk[0] == np.inf
    np.testing.assert_allclose(thr[1:], thr_sk[1:])

    # np.trapz(tpr, fpr) matches our AUC kernel and sklearn's auc.
    assert abs(np.trapezoid(tpr, fpr) - sk.auc(fpr_sk, tpr_sk)) < 1e-9
    assert abs(np.trapezoid(tpr, fpr) - fast_roc_auc(y, s)) < 1e-9


@pytest.mark.parametrize("seed", [3, 11, 99])
def test_equivalence_weighted_vs_sklearn(seed):
    """Equivalence weighted vs sklearn."""
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 2000))
    y = rng.integers(0, 2, n)
    if y.sum() == 0 or y.sum() == n:
        y[0], y[-1] = 0, 1
    s = rng.random(n)
    w = rng.random(n) + 0.1
    fpr, tpr, _ = fast_roc_curve(y, s, sample_weight=w)
    fpr_sk, tpr_sk, _ = sk.roc_curve(y, s, sample_weight=w)
    np.testing.assert_allclose(fpr, fpr_sk)
    np.testing.assert_allclose(tpr, tpr_sk)


# ---------------------------------------------------------------------------
# cProfile smoke on a large input (kept small enough for CI; asserts it runs + profiles)
# ---------------------------------------------------------------------------


def test_fast_roc_curve_large_profile():
    """Fast roc curve large profile."""
    import cProfile
    import io
    import pstats

    rng = np.random.default_rng(0)
    n = 1_000_000
    y = rng.integers(0, 2, n)
    s = rng.random(n)

    fast_roc_curve(y[:10], s[:10])  # warm up numba compile out of the timed region

    pr = cProfile.Profile()
    pr.enable()
    fpr, tpr, thr = fast_roc_curve(y, s)
    pr.disable()

    assert fpr.shape == tpr.shape
    assert thr[0] == np.inf
    pstats.Stats(pr, stream=io.StringIO())
    # a large curve at 1M rows still produces a bounded number of distinct-score vertices
    assert fpr.shape[0] <= n + 1
