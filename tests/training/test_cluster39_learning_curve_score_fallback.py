"""Cluster 39 regression: learning-curve scoring falls back only on a signature mismatch (audit 2026-11-20, finding 39-F14).

A TypeError raised inside a real ``scorer(estimator, X, y)`` used to be swallowed and replaced by ``estimator.score``, which hid scorer bugs. The
fallback now depends on whether the scorer can take the call, so that TypeError propagates.
"""

import numpy as np
import pytest

from mlframe.training.diagnostics.learning_curve import _score


class _Est:
    """Estimator stub whose own ``score`` returns a fixed value."""

    def score(self, X, y):
        """Constant score for the fallback path."""
        return 0.25


def test_real_scorer_is_used():
    """A scorer that accepts (estimator, X, y) is called and its value returned (39-F14)."""
    X = np.zeros((2, 1))
    y = np.zeros(2)
    assert _score(lambda est, X, y: 0.9, _Est(), X, y) == 0.9


def test_scorer_bug_propagates_instead_of_falling_back():
    """A TypeError raised inside a correct-signature scorer propagates (39-F14)."""
    X = np.zeros((2, 1))
    y = np.zeros(2)

    def _buggy(est, X, y):
        """A scorer with the right signature that hits an internal TypeError."""
        raise TypeError("internal bug in scorer")

    with pytest.raises(TypeError, match="internal bug"):
        _score(_buggy, _Est(), X, y)


def test_mismatched_signature_falls_back_to_estimator_score():
    """A scorer that cannot take (estimator, X, y) falls back to estimator.score (39-F14)."""
    X = np.zeros((2, 1))
    y = np.zeros(2)

    def _two_arg(est, X):
        """A scorer with only two parameters."""
        return 1.0

    assert _score(_two_arg, _Est(), X, y) == 0.25
