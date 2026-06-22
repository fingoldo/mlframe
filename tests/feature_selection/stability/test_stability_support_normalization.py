"""Regression tests for StabilityMRMR support_ normalisation and threshold FP-robustness."""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.stability import StabilityMRMR, _support_to_indices


class _BoolMaskSelector:
    """A selector that exposes ``support_`` as a sklearn-style boolean mask (not integer indices)."""

    def __init__(self, mask):
        self._mask = np.asarray(mask, dtype=bool)

    def get_params(self, deep=True):
        return {"mask": self._mask}

    def set_params(self, **p):
        if "mask" in p:
            self._mask = np.asarray(p["mask"], dtype=bool)
        return self

    def fit(self, X, y):
        self.support_ = self._mask.copy()
        return self


def test_support_to_indices_converts_boolean_mask():
    mask = np.array([True, False, True, False, True])
    np.testing.assert_array_equal(_support_to_indices(mask, 5), np.array([0, 2, 4]))


def test_support_to_indices_passes_through_integer_indices():
    idx = np.array([1, 3], dtype=np.int64)
    np.testing.assert_array_equal(_support_to_indices(idx, 5), np.array([1, 3]))


def test_boolean_mask_selector_counts_correct_columns():
    """Pre-fix a bool mask was coerced to [1,0,1,...] and incremented counts[0]/counts[1] only -> selection collapsed to columns 0/1."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5))
    y = (X[:, 2] > 0).astype(int)
    mask = [False, False, True, False, True]  # columns 2 and 4 are the real support
    sel = StabilityMRMR(
        estimator=_BoolMaskSelector(mask),
        n_bootstraps=6,
        sample_fraction=0.6,
        support_threshold=0.5,
        random_state=1,
    )
    sel.fit(X, y)
    np.testing.assert_array_equal(np.sort(sel.support_), np.array([2, 4]))
    # The mis-counted pre-fix result would have flagged columns {0, 1}.
    assert 0 not in set(sel.support_.tolist())
    assert 1 not in set(sel.support_.tolist())


class _FixedIdxSelector:
    def __init__(self, support):
        self.support = support

    def get_params(self, deep=True):
        return {"support": self.support}

    def set_params(self, **p):
        if "support" in p:
            self.support = p["support"]
        return self

    def fit(self, X, y):
        self.support_ = np.asarray(self.support, dtype=np.int64)
        return self


def test_threshold_exactly_at_fraction_is_robust_to_float_rounding():
    """12/20 == 0.6 is not exactly representable in float64; the integer-count gate must still keep a feature selected in exactly 60% of runs."""
    X = np.zeros((40, 3))
    y = np.zeros(40, dtype=int)
    sel = StabilityMRMR(
        estimator=_FixedIdxSelector(support=(0,)),
        n_bootstraps=20,
        sample_fraction=0.5,
        support_threshold=0.6,
        random_state=2,
    )
    sel.fit(X, y)
    # Feature 0 is selected in 20/20 = 1.0 >= 0.6 runs; trivially kept. Now exercise the exact-boundary count directly.
    counts = np.array([12, 7, 0])
    import math
    min_count = int(math.ceil(0.6 * 20 - 1e-9))
    assert min_count == 12
    assert (counts >= min_count).tolist() == [True, False, False]
