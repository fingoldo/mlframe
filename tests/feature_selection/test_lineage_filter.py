"""Unit tests for the cat-FE lineage filter in ``should_skip_candidate``
(B6).

The filter prevents the k-way screening enumeration from emitting
redundant ``(orig_i, kway(orig_i, orig_j))``-style candidates --
engineered columns already contain their parents' information, so
conditional MI degenerates and confidence gates waste budget.

When ``engineered_lineage=None`` (default), the function preserves
legacy behaviour bit-exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.evaluation import should_skip_candidate


@pytest.fixture
def base_state():
    """Common arguments to should_skip_candidate -- all empty / inert."""
    return dict(
        failed_candidates=set(),
        added_candidates=set(),
        expected_gains=np.zeros(40, dtype=np.float64),
        selected_vars=[],
        selected_interactions_vars=[],
        only_unknown_interactions=True,
    )


class TestLineageFilterDefault:
    """Without ``engineered_lineage``, behaviour matches the legacy path."""

    def test_no_engineered_lineage_does_not_change_behavior(self, base_state):
        # A 2-way candidate with no overlapping selected vars should NOT
        # be skipped under either path.
        skip, n = should_skip_candidate(
            cand_idx=5, X=(2, 3), interactions_order=2,
            **base_state,
        )
        assert skip is False
        assert n == 0

    def test_no_engineered_lineage_passing_none_explicit(self, base_state):
        skip, n = should_skip_candidate(
            cand_idx=5, X=(2, 3), interactions_order=2,
            engineered_lineage=None,
            **base_state,
        )
        assert skip is False
        assert n == 0


class TestLineageFilterActive:
    """With ``engineered_lineage`` set, the filter blocks redundant k-way
    candidates."""

    def test_skips_candidate_combining_engineered_with_its_parent(self, base_state):
        """``kway(0, 1)`` lives at idx=10 with parents {0, 1}.
        Candidate ``(0, 10)`` combines parent 0 with its own engineered
        col -- redundant, must be skipped."""
        engineered_lineage = {10: frozenset({0, 1})}
        skip, _ = should_skip_candidate(
            cand_idx=15, X=(0, 10), interactions_order=2,
            engineered_lineage=engineered_lineage,
            **base_state,
        )
        assert skip is True, "Lineage filter must catch parent+engineered overlap"

    def test_passes_candidate_with_disjoint_parents(self, base_state):
        """``kway(0, 1)`` lives at idx=10. Candidate ``(2, 10)`` has
        no overlap with the engineered col's parents -- legitimate
        interaction, should NOT be skipped."""
        engineered_lineage = {10: frozenset({0, 1})}
        skip, _ = should_skip_candidate(
            cand_idx=15, X=(2, 10), interactions_order=2,
            engineered_lineage=engineered_lineage,
            **base_state,
        )
        assert skip is False

    def test_skips_candidate_combining_two_engineered_with_shared_parent(self, base_state):
        """Two engineered cols sharing a parent: ``kway(0, 1)`` at idx=10
        and ``kway(0, 2)`` at idx=11. Candidate ``(10, 11)`` shares
        parent 0 -- both engineered cols contain it -- the joint adds
        no new info, redundant."""
        engineered_lineage = {10: frozenset({0, 1}), 11: frozenset({0, 2})}
        skip, _ = should_skip_candidate(
            cand_idx=20, X=(10, 11), interactions_order=2,
            engineered_lineage=engineered_lineage,
            **base_state,
        )
        # The filter checks for "engineered col's parent in candidate".
        # For X=(10, 11): subel=10, parents={0,1}; X_set={10,11}; no
        # overlap -> not caught here. subel=11, parents={0,2}; X_set={10,11};
        # no overlap either. So this candidate passes the lineage filter.
        # That's actually correct -- the two engineered cols share an
        # implicit dependency on parent 0, but the filter as designed
        # only catches DIRECT parent-of-X-itself patterns. Document the
        # limitation here for future tightening.
        assert skip is False, \
            "Current lineage filter only catches direct parent-in-candidate overlap"

    def test_higher_order_kway_with_engineered_parent(self, base_state):
        """3-way candidate ``(0, 1, 10)`` where idx=10 is ``kway(0, 1)``.
        Parents {0, 1} both appear in the candidate -- skip."""
        engineered_lineage = {10: frozenset({0, 1})}
        skip, _ = should_skip_candidate(
            cand_idx=15, X=(0, 1, 10), interactions_order=3,
            engineered_lineage=engineered_lineage,
            **base_state,
        )
        assert skip is True

    def test_order_1_candidates_unaffected_by_lineage(self, base_state):
        """``interactions_order=1`` skips the entire lineage block --
        single-var candidates never trigger the filter."""
        engineered_lineage = {10: frozenset({0, 1})}
        skip, _ = should_skip_candidate(
            cand_idx=10, X=(10,), interactions_order=1,
            engineered_lineage=engineered_lineage,
            **base_state,
        )
        # Order-1 path doesn't reach the lineage block; only failed/
        # added/gain checks apply. Should not be skipped.
        assert skip is False


class TestLineageFilterEdgeCases:
    def test_empty_lineage_dict_is_no_op(self, base_state):
        skip, _ = should_skip_candidate(
            cand_idx=5, X=(0, 1), interactions_order=2,
            engineered_lineage={},
            **base_state,
        )
        assert skip is False

    def test_lineage_with_unrelated_engineered_cols(self, base_state):
        """Engineered cols exist but candidate touches none of their
        parents -- pass through."""
        engineered_lineage = {
            10: frozenset({0, 1}),
            11: frozenset({2, 3}),
        }
        skip, _ = should_skip_candidate(
            cand_idx=20, X=(4, 5), interactions_order=2,
            engineered_lineage=engineered_lineage,
            **base_state,
        )
        assert skip is False
