"""biz_value + unit tests for row-wise ordinal-pattern (Bandt-Pompe) encoding (mrmr_audit_2026-07-20
fe_expansion.md "Row-wise ordinal-pattern (Bandt-Pompe permutation) encoding").

Validates ``ordinal_pattern_ids`` / ``ordinal_pattern_lexicographic_rank``
(``_ordinal_pattern_fe``) -- operator #3 in the argmax/conditional-gate family, generalizing
row-argmax (#1) to the FULL ranking.

Contracts pinned
-----------------
* ``TestLexicographicRank``: the rank function assigns exactly the K! distinct ids 0..K!-1 with no
  collisions, identity permutation -> rank 0, reverse permutation -> rank K!-1.
* ``TestOrdinalPatternIds``: for K=3, exactly 6 distinct pattern ids are realized on generic data;
  a specific hand-checked row maps to its expected id.
* ``TestBizValueArgmaxCannotDistinguish`` (biz_value): on y = 1{x1 > x2 > x3}, the full ordinal
  pattern perfectly separates the target (one pattern id is 100% positive, the other 5 are 100%
  negative) while row-argmax(x1,x2,x3) CANNOT (both target classes have the SAME argmax value,
  since x1 is the row-max in 2 of the 6 orderings and only one of those is target-positive).
* Ties (tie_policy="nan") and NaN input propagate to NaN, never silently pick an arbitrary order.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._conditional_gate_fe import apply_row_argmax
from mlframe.feature_selection.filters._ordinal_pattern_fe import (
    ordinal_pattern_ids,
    ordinal_pattern_lexicographic_rank,
)


class TestLexicographicRank:
    """The rank function must assign exactly K! distinct ids with the documented boundary cases."""

    def test_k3_gives_six_distinct_ranks_no_collisions(self):
        """Every one of the 6 permutations of range(3) must get a distinct rank in 0..5."""
        from itertools import permutations

        ranks = [ordinal_pattern_lexicographic_rank(p) for p in permutations(range(3))]
        assert sorted(ranks) == list(range(6)), f"expected ranks {{0..5}} with no collisions, got {ranks}"

    def test_identity_permutation_is_rank_zero(self):
        """(0, 1, 2), the lexicographically first permutation, must rank 0."""
        assert ordinal_pattern_lexicographic_rank((0, 1, 2)) == 0

    def test_reverse_permutation_is_last_rank(self):
        """(2, 1, 0), the lexicographically last permutation, must rank K!-1 = 5."""
        assert ordinal_pattern_lexicographic_rank((2, 1, 0)) == 5

    def test_k4_gives_24_distinct_ranks(self):
        """The same no-collision property must hold at K=4 (4! = 24 permutations)."""
        from itertools import permutations

        ranks = [ordinal_pattern_lexicographic_rank(p) for p in permutations(range(4))]
        assert sorted(ranks) == list(range(24))


class TestOrdinalPatternIds:
    """For K=3 generic data, all 6 orderings must appear, and a hand-checked row must match."""

    def test_all_six_patterns_realized_on_generic_data(self):
        """Random K=3 data over many rows must realize all 6 possible pattern ids."""
        rng = np.random.default_rng(0)
        n = 6000
        X = rng.standard_normal((n, 3))
        ids = ordinal_pattern_ids(X)
        assert set(np.unique(ids[np.isfinite(ids)]).astype(int)) == set(range(6))

    def test_hand_checked_row_maps_to_expected_id(self):
        """row = (5, 3, 1): ascending sort order is (2, 1, 0) -- the reverse permutation -> rank 5."""
        X = np.array([[5.0, 3.0, 1.0]])
        ids = ordinal_pattern_ids(X)
        assert ids[0] == 5.0

    def test_ties_return_nan_under_default_policy(self):
        """A row with an exact tie among its K values must return NaN, never an arbitrary order."""
        X = np.array([[1.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
        ids = ordinal_pattern_ids(X)
        assert np.isnan(ids[0])
        assert np.isfinite(ids[1])

    def test_nan_input_row_returns_nan(self):
        """A row containing NaN in any column must return NaN, not a spurious ordering."""
        X = np.array([[1.0, np.nan, 3.0], [1.0, 2.0, 3.0]])
        ids = ordinal_pattern_ids(X)
        assert np.isnan(ids[0])
        assert np.isfinite(ids[1])

    def test_invalid_k_raises(self):
        """K < 2 (a single column) has no ordering to encode; must raise ValueError."""
        with pytest.raises(ValueError, match="K must be"):
            ordinal_pattern_ids(np.array([[1.0], [2.0]]))

    def test_non_2d_input_raises(self):
        """A 1-D input array must raise ValueError rather than silently reshaping."""
        with pytest.raises(ValueError, match="must be 2-D"):
            ordinal_pattern_ids(np.array([1.0, 2.0, 3.0]))


class TestBizValueArgmaxCannotDistinguish:
    """biz_value: y = 1{x1 > x2 > x3} is perfectly separated by the full ordinal pattern but
    row-argmax(x1,x2,x3) cannot distinguish the two target classes at all."""

    def test_ordinal_pattern_perfectly_separates_target_argmax_cannot(self):
        """The full ordinal pattern perfectly separates y = 1{x1>x2>x3}; row-argmax cannot."""
        rng = np.random.default_rng(1)
        n = 6000
        X = rng.standard_normal((n, 3))
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        y = ((x1 > x2) & (x2 > x3)).astype(int)

        ids = ordinal_pattern_ids(X)
        # The single pattern id corresponding to the (x1>x2>x3) ordering must be PURE y=1, and every
        # other realized pattern id must be PURE y=0 -- perfect separation by construction.
        finite = np.isfinite(ids)
        for pid in np.unique(ids[finite]):
            y_here = y[finite & (ids == pid)]
            assert len(set(y_here.tolist())) == 1, f"pattern id {pid} is not pure in y: {set(y_here.tolist())}"
        # x1>x2>x3 means the ASCENDING sort order (argsort) is (col2, col1, col0) = (2, 1, 0), the
        # reverse permutation -> rank 5 (matches TestOrdinalPatternIds's hand-checked row above).
        assert set(np.unique(ids[finite & (y == 1)])) == {ordinal_pattern_lexicographic_rank((2, 1, 0))}

        # Row-argmax over (x1, x2, x3) must NOT separate the classes: x1 is the row-max in BOTH the
        # x1>x2>x3 (y=1) and x1>x3>x2 (y=0) orderings, so argmax==0 occurs in BOTH classes.
        import pandas as pd

        Xdf = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        argmax_codes = apply_row_argmax(Xdf, ["x1", "x2", "x3"])
        y_where_argmax_is_x1 = y[argmax_codes == 0]
        assert len(set(y_where_argmax_is_x1.tolist())) == 2, "row-argmax should NOT purely separate the target -- both classes share argmax==x1"
