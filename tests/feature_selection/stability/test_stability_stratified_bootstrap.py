"""Regression test for S2: StabilityMRMR class-stratified bootstrap on rare/imbalanced y.

Pre-fix the bootstrap subsample was ``rng.choice(n, replace=False)`` with no stratification, so on a rare-class target a bootstrap could omit the minority class entirely,
giving a single-class fit the base selector degenerates on. The fix adds ``stratify=True`` (default) which preserves per-class proportions so every class survives every
bootstrap. Per the project rule, a 1%-prevalence class needs n ~ 5000 for a reliable split, so the fixture uses n=5000.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from mlframe.feature_selection.filters.stability import StabilityMRMR


# sklearn ``clone`` deep-copies constructor params, so an instance attribute cannot collect across the clones the wrapper fits. The class attribute is shared by reference
# across clones (deepcopy copies the instance dict, not class attributes), so each cloned fit appends to the same list. Tests reset it before use.
class _ClassCountRecordingSelector(BaseEstimator):
    """Records the number of distinct y-classes seen in each fit into a class-level list, then returns a fixed support."""

    seen_class_counts: list = []

    def __init__(self, support=(0, 1)):
        self.support = support

    def fit(self, X, y):
        _ClassCountRecordingSelector.seen_class_counts.append(int(np.unique(np.asarray(y)).size))
        self.support_ = np.asarray(self.support, dtype=np.int64)
        return self


def _rare_imbalance_data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = np.zeros(n, dtype=np.int64)
    # 1% prevalence positive class.
    n_pos = max(1, n // 100)
    y[rng.choice(n, size=n_pos, replace=False)] = 1
    return X, y


def test_stratify_default_keeps_both_classes_in_every_bootstrap():
    X, y = _rare_imbalance_data()
    _ClassCountRecordingSelector.seen_class_counts = []
    sel = StabilityMRMR(
        _ClassCountRecordingSelector(), n_bootstraps=20,
        sample_fraction=0.3, random_state=1, stratify=True,
    )
    sel.fit(X, y)
    sink = _ClassCountRecordingSelector.seen_class_counts
    assert len(sink) == 20
    assert all(c == 2 for c in sink), f"stratified bootstraps must always retain both classes; saw class counts {sink}"


def test_unstratified_can_drop_minority_class():
    """The legacy unstratified path must be able to produce a single-class bootstrap on this rare target (proves the bug the default now fixes is real)."""
    X, y = _rare_imbalance_data()
    _ClassCountRecordingSelector.seen_class_counts = []
    # Tiny per-bootstrap fraction on a 1%-prevalence target: an unstratified draw frequently misses all ~50 positives.
    sel = StabilityMRMR(
        _ClassCountRecordingSelector(), n_bootstraps=20,
        sample_fraction=0.005, random_state=1, stratify=False,
    )
    sel.fit(X, y)
    sink = _ClassCountRecordingSelector.seen_class_counts
    assert any(c < 2 for c in sink), f"expected at least one single-class unstratified bootstrap on a 1%% target; saw {sink}"


def test_stratify_falls_back_on_continuous_target():
    """A continuous (regression) y has too many distinct values to stratify; the wrapper must fall back to the plain draw and still fit."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((400, 4))
    y = rng.standard_normal(400)
    _ClassCountRecordingSelector.seen_class_counts = []
    sel = StabilityMRMR(
        _ClassCountRecordingSelector(), n_bootstraps=5,
        sample_fraction=0.5, random_state=3, stratify=True,
    )
    sel.fit(X, y)
    assert len(_ClassCountRecordingSelector.seen_class_counts) == 5
    assert sel.support_.tolist() == [0, 1]
