"""Weighted kappa depends on the classes that occur, not on the largest label value.

The earlier code took the class count as ``max(label) + 1``. A label value that never occurs then became an empty class: it changed the
quadratic and linear weights, so the same ratings coded 0, 1, 2 and 0, 1, 3 gave different kappas, and a label of 3000 allocated a 3001 x 3001
matrix for three classes. Labels are now ranked over the distinct values that occur.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.classification._weighted_kappa import quadratic_weighted_kappa, weighted_kappa


def _ratings(seed: int, n: int = 300):
    """Three ordinal classes 0, 1, 2 with noisy agreement."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 3, size=n)
    y_pred = np.clip(y_true + rng.integers(-1, 2, size=n), 0, 2)
    return y_true, y_pred


@pytest.mark.parametrize("weights", ["quadratic", "linear"])
def test_an_unused_value_between_classes_does_not_change_kappa(weights):
    """Ratings coded 0, 1, 3 (value 2 never used) score exactly like the same ratings coded 0, 1, 2."""
    y_true, y_pred = _ratings(0)
    relabel = np.array([0, 1, 3])
    assert weighted_kappa(relabel[y_true], relabel[y_pred], weights=weights) == pytest.approx(weighted_kappa(y_true, y_pred, weights=weights), abs=1e-12)


def test_a_large_unused_label_value_does_not_change_kappa():
    """A label value far above the classes in use (3000 for three classes) does not change kappa or enlarge the computation."""
    y_true, y_pred = _ratings(1)
    relabel = np.array([0, 1, 3000])
    assert quadratic_weighted_kappa(relabel[y_true], relabel[y_pred]) == pytest.approx(quadratic_weighted_kappa(y_true, y_pred), abs=1e-12)


def test_explicit_n_classes_still_fixes_the_scale():
    """An explicit ``n_classes`` keeps the caller's scale: labels must lie below it, and the classes it adds count."""
    y = np.array([0, 1, 2, 2, 1, 0])
    assert weighted_kappa(y, y, n_classes=5) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="n_classes"):
        weighted_kappa(y, y, n_classes=2)
