"""The Xi batch gives each finite column the tie keys of its own number: splitting the columns into blocks or taking one alone leaves the values unchanged."""

from __future__ import annotations

import numpy as np

import mlframe.feature_selection.filters._orthogonal_xi_fe as xi
from tests.feature_selection._xi_reference import xi_reference


def test_batch_equals_the_reference_over_many_blocks(monkeypatch):
    """More columns than one block (the block is forced to 7 columns): the values equal the reference's, column by column."""
    rng = np.random.default_rng(2)
    n, k = 600, 40
    X = rng.standard_normal((n, k))
    X[:, 5] = np.round(X[:, 5], 1)
    y = (X[:, 0] + rng.standard_normal(n) > 0).astype(np.int64)
    whole = xi.xi_correlation_batch(X, y, random_state=7)
    monkeypatch.setattr(xi, "_score_block", _small_blocks(xi._score_block))
    np.testing.assert_array_equal(xi.xi_correlation_batch(X, y, random_state=7), whole)
    np.testing.assert_array_equal(whole, xi_reference(X.astype(np.float32), y.astype(np.float64), 7))


def _small_blocks(score_block):
    """A ``_score_block`` that scores at most 7 columns per call, whatever it is given."""

    def split(X, js, seeds, r_row, l_row, out):
        """Score ``js`` in pieces of 7 columns."""
        for s in range(0, len(js), 7):
            score_block(X, js[s : s + 7], seeds[s : s + 7], r_row, l_row, out)

    return split


def test_single_column_equals_the_first_column_of_a_batch():
    """``xi_correlation(x, y, random_state)`` is column 0 of the batch with the same state."""
    rng = np.random.default_rng(3)
    n = 900
    X = rng.standard_normal((n, 4)).astype(np.float32)
    X[:, 0] = np.round(X[:, 0])
    y = (X[:, 1] > 0).astype(np.int64)
    assert xi.xi_correlation(X[:, 0], y, random_state=4) == xi.xi_correlation_batch(X, y, random_state=4)[0]
