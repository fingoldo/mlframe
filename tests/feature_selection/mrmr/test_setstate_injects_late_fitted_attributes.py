"""An MRMR pickle from before a fitted attribute existed must unpickle with that attribute at its neutral default."""

from __future__ import annotations

from mlframe.feature_selection.filters.mrmr._mrmr_setstate_defaults import _SETSTATE_LEGACY_DEFAULTS, build_setstate_defaults

_LATE_LISTS = ("cat_pair_features_", "lof_features_", "sir_direction_features_", "temporal_agg_features_", "_hinge_features_", "_cluster_aggregate_removals_")
_LATE_NONES = ("dcd_", "friend_graph_", "provenance_", "support_linear_", "support_universal_", "_artifacts_", "fallback_metadata_")
_LATE_FALSES = ("fallback_used_", "ran_out_of_time_", "_redundancy_emptied_raw_")


def _legacy_instance(missing):
    """An MRMR whose state lacks ``missing``, rebuilt through __setstate__ exactly as unpickling an old file would."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    state = dict(MRMR().__getstate__())
    for name in missing:
        state.pop(name, None)
    obj = MRMR.__new__(MRMR)
    obj.__setstate__(state)
    return obj


def test_late_attributes_are_injected_with_their_neutral_defaults():
    """Lists come back empty, "never computed" attributes None, flags False, the step counter 0."""
    names = _LATE_LISTS + _LATE_NONES + _LATE_FALSES + ("_fe_steps_executed_",)
    obj = _legacy_instance(names)
    assert _LATE_LISTS and _LATE_NONES and _LATE_FALSES
    for n in _LATE_LISTS:
        assert getattr(obj, n) == []
    for n in _LATE_NONES:
        assert getattr(obj, n) is None
    for n in _LATE_FALSES:
        assert getattr(obj, n) is False
    assert obj._fe_steps_executed_ == 0


def test_injected_containers_are_not_shared_between_unpickled_instances():
    """Mutating one instance's injected list/dict/set must not leak into another (the roster is copied per call)."""
    a = _legacy_instance(("lof_features_", "_gate_col_src_vars_", "_gate_raw_operands_"))
    b = _legacy_instance(("lof_features_", "_gate_col_src_vars_", "_gate_raw_operands_"))
    a.lof_features_.append("x")
    a._gate_col_src_vars_["k"] = 1
    a._gate_raw_operands_.add(3)
    assert b.lof_features_ == [] and b._gate_col_src_vars_ == {} and b._gate_raw_operands_ == set()
    assert _SETSTATE_LEGACY_DEFAULTS["lof_features_"] == []


def test_a_value_present_in_the_pickle_wins_over_the_default():
    """Negative control: setstate fills only what is missing, never overwrites a stored fitted attribute."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    state = dict(MRMR().__getstate__())
    state["lof_features_"] = ["lof_a"]
    state["ran_out_of_time_"] = True
    obj = MRMR.__new__(MRMR)
    obj.__setstate__(state)
    assert obj.lof_features_ == ["lof_a"] and obj.ran_out_of_time_ is True


def test_roster_has_no_neutral_default_for_core_outputs():
    """Core outputs every fitted pickle carries must stay out of the roster, or an unfitted estimator would look fitted for them."""
    defaults = build_setstate_defaults()
    for core in ("support_", "n_features_in_", "feature_names_in_", "mrmr_gains_"):
        assert core not in defaults
