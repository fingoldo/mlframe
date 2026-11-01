"""The percentile rank equals the two-sided searchsorted midpoint for members, ties, absent values and NaN."""

import numpy as np

from mlframe.feature_selection.filters._grouped_quantile_fe import _pct_rank_in_sorted


def _reference(sorted_vals, x):
    """The midpoint of the left and right insertion ranks."""
    m = sorted_vals.size
    lo = np.searchsorted(sorted_vals, x, side="left")
    hi = np.searchsorted(sorted_vals, x, side="right")
    return (lo + hi) / (2.0 * m)


def test_equals_two_sided_search():
    """Distinct values, heavy ties, queries outside the range, NaN and infinities."""
    rng = np.random.default_rng(0)
    for base in (rng.normal(size=500), np.round(rng.normal(size=500), 1), np.zeros(50)):
        sv = np.sort(base)
        x = np.concatenate([base, rng.normal(size=100) * 3, [np.nan, np.inf, -np.inf, sv[0], sv[-1]]])
        np.testing.assert_array_equal(_pct_rank_in_sorted(sv, x), _reference(sv, x))


def test_member_ranks_equal_the_search_for_a_groups_own_values():
    """The ranks read off the sort permutation equal the two-sided search of the same values, with ties, NaN and infinities among them."""
    from mlframe.feature_selection.filters._grouped_quantile_fe import _member_pct_ranks

    rng = np.random.default_rng(3)
    for base in (rng.normal(size=400), np.round(rng.normal(size=400), 1), np.full(30, 2.0)):
        vals = base.astype(np.float32)
        vals[[0, 5, 9]] = [np.nan, np.inf, -np.inf]
        finite = np.isfinite(vals)
        fin = vals[finite]
        order = np.argsort(fin, kind="stable")
        sv = fin[order]
        np.testing.assert_array_equal(_member_pct_ranks(order, sv, vals, finite), _reference(sv, vals))
