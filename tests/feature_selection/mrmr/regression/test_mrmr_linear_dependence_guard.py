"""Regression tests for the MRMR exact-linear-dependence guard.

Pre-fix, MRMR kept the exact combination x3 = 2*x1 - x2 beside x1 and x2 on the multicollinear fixture, so the kept
raw block had a singular Gram matrix. The guard keeps the first representative in relevance order and drops any
later column that the kept columns already span.
"""

import numpy as np

from mlframe.feature_selection.filters._mrmr_fit_impl._linear_dependence_guard import exact_dependence_keep_mask


def _singular_triple(n: int = 600, seed: int = 0):
    """Return x1, x2 and x3 = 2*x1 - x2, a rank-2 triple."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return x1, x2, 2.0 * x1 - x2


def test_exact_combination_is_dropped_when_its_parts_precede_it():
    """The exact combination is dropped when x1 and x2 come first in relevance order."""
    x1, x2, x3 = _singular_triple()
    keep = exact_dependence_keep_mask(np.column_stack([x1, x2, x3]))
    assert keep.tolist() == [True, True, False]


def test_exact_combination_is_kept_when_it_ranks_first():
    """With x3 ranked first the last column of the span, x2, is the one dropped, so the outcome follows rank order."""
    x1, x2, x3 = _singular_triple()
    keep = exact_dependence_keep_mask(np.column_stack([x3, x1, x2]))
    assert keep.tolist() == [True, True, False]


def test_independent_columns_are_all_kept():
    """Independent random columns are all kept."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 5))
    keep = exact_dependence_keep_mask(X)
    assert keep.all()


def test_near_collinear_but_not_exact_column_is_kept():
    """A column with 1e-3 noise on top of its partner is not an exact combination and is kept."""
    rng = np.random.default_rng(2)
    x1 = rng.normal(size=500)
    x2 = x1 + 1e-3 * rng.normal(size=500)
    keep = exact_dependence_keep_mask(np.column_stack([x1, x2]))
    assert keep.all()


def test_constant_column_is_not_judged():
    """A constant column has no centred norm, so the guard keeps it."""
    rng = np.random.default_rng(3)
    x1 = rng.normal(size=300)
    const = np.full(300, 7.0)
    keep = exact_dependence_keep_mask(np.column_stack([x1, const]))
    assert keep.all()
