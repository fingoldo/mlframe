"""The dispatched tree aggregation completes ragged runs with the missing-value policy (audit 2026-11-20, finding 15-F2)."""
from mlframe.feature_selection.wrappers import _helpers_importance_agg as agg


def test_feature_missing_from_runs_ranks_below_a_full_coverage_feature():
    """A is present in one of three runs and B in all three with the same value: under 'worst' the missing runs push A below B.
    Before the fix the dispatcher ignored the policy and A tied with B."""
    fi = {
        "run0": {"A": 0.9, "B": 0.9},
        "run1": {"B": 0.9},
        "run2": {"B": 0.9},
    }
    method = next(iter(agg.VotesAggregation))
    order = agg.aggregate_importances_dispatched(fi, family="tree", votes_aggregation_method=method, fi_missing_policy="worst")
    assert order.index("B") < order.index("A")
