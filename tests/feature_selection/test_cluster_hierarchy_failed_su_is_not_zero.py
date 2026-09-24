"""A failed pair-SU must be left out, not recorded as 0.0 ("maximally non-redundant")."""

from mlframe.feature_selection.filters._cluster_hierarchy import _component_medoid, _record_pair_su


def test_failed_and_non_finite_scores_are_not_recorded():
    pair_sus: dict = {}
    _record_pair_su(pair_sus, "a", "b", float("nan"))
    _record_pair_su(pair_sus, "a", "c", float("inf"))
    _record_pair_su(pair_sus, "b", "c", 0.7)
    assert pair_sus == {("b", "c"): 0.7}


def test_a_missing_pair_does_not_drag_the_medoid_down():
    """A fake 0.0 for the failed (b, c) pair halves b's mean and hands the medoid to a."""
    measured = {("a", "b"): 0.6, ("a", "c"): 0.5}  # ("b", "c") failed
    assert _component_medoid(["a", "b", "c"], measured) == "b", "b's only measured SU is the strongest"
    with_fake_zero = {**measured, ("b", "c"): 0.0}
    assert _component_medoid(["a", "b", "c"], with_fake_zero) == "a", "the fake zero is what used to move it"
