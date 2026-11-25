"""Cluster 04 regression: categorical draws stay inside [0, K) even when the cumulative sum rounds below 1 (audit 2026-11-20, finding 04-F34)."""

import numpy as np

from mlframe.data.datasets._targets_nonbinary import sample_categorical


def test_draw_above_rounded_cumulative_sum_is_clamped_to_last_class():
    """A row whose probabilities sum to 1 - eps must not return K for a draw above the cumulative total (04-F34)."""
    probs = np.array([[0.3, 0.3, 0.4 - 1e-12]])

    class _Fixed:
        """Stub generator returning a draw above the rounded total."""

        def random(self, n):
            """Return a draw that exceeds the cumulative sum."""
            return np.full(n, 1.0 - 1e-15)

    out = sample_categorical(probs, _Fixed())
    assert out.tolist() == [2]


def test_ordinary_draws_are_unchanged():
    """Control: in-range draws keep their class (04-F34)."""
    probs = np.array([[0.2, 0.8], [0.9, 0.1]])

    class _Fixed:
        """Stub generator returning fixed draws."""

        def random(self, n):
            """Return draws that select the first class in row 0 and the second in row 1."""
            return np.array([0.1, 0.95])

    assert sample_categorical(probs, _Fixed()).tolist() == [0, 1]
