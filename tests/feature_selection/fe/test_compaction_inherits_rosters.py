"""A compaction replacement goes into the roster of the family whose column it replaces."""

from types import SimpleNamespace

from mlframe.feature_selection.filters._mrmr_fit_impl._selection_compaction import inherit_fe_rosters


def test_replacement_takes_the_position_of_the_replaced_column_in_its_roster():
    """``cand`` joins the rosters that listed a replaced column, at its position, and no other roster."""
    est = SimpleNamespace(hybrid_orth_features_=["a", "diff__He1", "b"], modular_features_=["m"], _hinge_features_=["diff__He1"])
    inherit_fe_rosters(est, "diff__He2", ("diff__He1",))
    assert est.hybrid_orth_features_ == ["a", "diff__He2", "diff__He1", "b"]
    assert est.modular_features_ == ["m"]
    assert est._hinge_features_ == ["diff__He1"]  # the hinge roster is managed by the hinge protection block


def test_candidate_already_in_the_roster_is_not_added_twice():
    """Idempotent when the replacement is already listed."""
    est = SimpleNamespace(hybrid_orth_features_=["diff__He2", "diff__He1"])
    inherit_fe_rosters(est, "diff__He2", ("diff__He1",))
    assert est.hybrid_orth_features_ == ["diff__He2", "diff__He1"]
