"""The cumulative-gain curve asserted a ranking the model never made.

Read off every rank, the curve steps through a tied run in whatever order the sort happened to produce --
but a model cannot rank rows it scored identically. On a 20k-row column of 2-dp tree scores 19,899 of the
20,000 curve points sat inside a tied run, and merely reshuffling the ties moved the drawn curve by up to
0.25% of all positives captured. Sampling at distinct scores (run ends) is the honest reading, and the two
orderings agree exactly there.
"""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.binary import _ScoreSort, gain_curve_points


def _tied_case(seed: int = 0, n: int = 20_000):
    """A realistic quantised-score column: 2-dp tree outputs, so nearly every rank sits inside a tie."""
    rng = np.random.default_rng(seed)
    score = np.round(rng.beta(2.0, 5.0, size=n), 2)
    y = (rng.random(n) < score).astype(np.int64)
    return y, score


def _reshuffled(y, score, seed: int):
    """The same data with the rows permuted: only the WITHIN-tie order can differ."""
    perm = np.random.default_rng(seed).permutation(len(y))
    return y[perm], score[perm]


def test_the_gain_curve_does_not_move_when_tied_rows_are_reshuffled():
    """The whole contract: within-tie order is not information, so it must not reach the drawn curve."""
    y, score = _tied_case()
    a = gain_curve_points(_ScoreSort(y, score))
    y2, s2 = _reshuffled(y, score, 7)
    b = gain_curve_points(_ScoreSort(y2, s2))
    assert np.array_equal(a[0], b[0]), "the population fractions moved under a tie reshuffle"
    assert np.max(np.abs(a[1] - b[1])) == 0.0, "the captured-positive curve moved under a tie reshuffle"


def test_reading_the_curve_at_every_rank_would_have_moved_it():
    """Pin the defect itself: the per-rank curve this replaced is NOT reshuffle-invariant."""
    y, score = _tied_case()
    y2, s2 = _reshuffled(y, score, 7)

    def per_rank(yv, sv):
        """The old formulation: one point per rank, ties walked in sort order."""
        sort = _ScoreSort(yv, sv)
        return sort.cum_tp.astype(np.float64) / sort.n_pos

    moved = np.max(np.abs(per_rank(y, score) - per_rank(y2, s2)))
    assert moved > 1e-3, f"the per-rank curve moved only {moved:.2e}; the fixture no longer carries real ties"


def test_the_curve_starts_at_the_origin_and_ends_at_full_capture():
    """A gains chart is read corner to corner; losing either endpoint misreads the lift area."""
    y, score = _tied_case()
    pop, gain = gain_curve_points(_ScoreSort(y, score))
    assert (pop[0], gain[0]) == (0.0, 0.0)
    assert pop[-1] == 1.0
    assert gain[-1] == 1.0


def test_the_curve_is_monotone_and_sampled_at_distinct_scores_only():
    """One vertex per distinct score, in increasing population order, capturing no fewer positives."""
    y, score = _tied_case()
    sort = _ScoreSort(y, score)
    pop, gain = gain_curve_points(sort)
    assert len(pop) == len(np.unique(score)) + 1, "one vertex per distinct score plus the origin"
    assert len(pop) < sort.n / 10, "the curve still carries a vertex per rank"
    assert np.all(np.diff(pop) > 0.0)
    assert np.all(np.diff(gain) >= 0.0)


def test_a_tie_free_column_keeps_every_rank():
    """Guard: the fix is about ties, not about decimation -- distinct scores must all survive."""
    rng = np.random.default_rng(3)
    score = rng.random(500)
    y = (rng.random(500) < score).astype(np.int64)
    pop, _ = gain_curve_points(_ScoreSort(y, score))
    assert len(pop) == 501


def test_an_all_tied_column_is_a_single_straight_segment():
    """Every row scored the same: the only honest curve is the random-targeting diagonal."""
    y = np.array([1, 0, 1, 0, 1, 1], dtype=np.int64)
    pop, gain = gain_curve_points(_ScoreSort(y, np.full(6, 0.5)))
    assert np.array_equal(pop, np.array([0.0, 1.0]))
    assert np.array_equal(gain, np.array([0.0, 1.0]))
