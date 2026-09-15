"""Redundancy/orth P3 contracts.

* ``relax_mrmr_score`` rejects a negative alpha; it used to skip the signed interaction term silently, as if alpha were 0.
* The tree rescue ranks importance ties by column order; the reversed ascending argsort preferred the highest column index.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._relaxmrmr_3d import relax_mrmr_score


def _codes(n=200, seed=0):
    """Integer-coded candidate, two selected columns and a target."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 3, size=n).astype(np.int64) for _ in range(4)]


def test_relax_mrmr_rejects_negative_alpha():
    """A negative weight on the signed interaction term is an input error."""
    x, s1, s2, y = _codes()
    with pytest.raises(ValueError, match="alpha"):
        relax_mrmr_score(x, [s1, s2], y, 3, [3, 3], 3, alpha=-1.0)


def test_relax_mrmr_zero_and_positive_alpha_still_score():
    """Controls: alpha=0 and alpha=1 both return a finite score."""
    x, s1, s2, y = _codes(seed=1)
    for a in (0.0, 1.0):
        assert np.isfinite(relax_mrmr_score(x, [s1, s2], y, 3, [3, 3], 3, alpha=a))


def test_tree_rescue_ties_broken_by_column_order():
    """Equal importances keep ascending column order; zero-importance columns are excluded; higher importance still ranks first."""
    from mlframe.feature_selection.filters._mrmr_tree_rescue import _rank_by_importance

    imp = np.array([0.0, 5.0, 2.0, 5.0, 2.0, 7.0])
    assert _rank_by_importance(imp) == [5, 1, 3, 2, 4]
