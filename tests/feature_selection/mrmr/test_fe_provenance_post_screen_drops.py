"""fe_provenance_ names the post-screen pass that dropped an engineered column, not only the rejection-ledger gates."""

from mlframe.feature_selection.filters._mrmr_fe_provenance import _output_membership


class _Fitted:
    """A fitted-estimator stand-in with the attributes the provenance reads."""

    _fe_rejection_records_ = [{"candidate": "a", "gate": "accuracy_gate"}, {"candidate": "b", "gate": "accuracy_gate"}]
    _final_subsumption_dropped_ = ["b", "c"]
    _selection_compaction_ = [(None, ("d",)), ("z", ("b", "e"))]

    def get_feature_names_out(self):
        """The finished output."""
        return ["x", "z"]


def test_later_passes_override_earlier_gates_and_name_the_replacement():
    """Ledger gate for a; final subsumption for c; compaction for d (redundant) and b, e (replaced by z), overriding b's earlier verdicts."""
    names, last = _output_membership(_Fitted())
    assert names == {"x", "z"}
    assert last == {
        "a": "accuracy_gate",
        "c": "final_subsumption",
        "d": "selection_compaction:redundant",
        "b": "selection_compaction:replaced_by:z",
        "e": "selection_compaction:replaced_by:z",
    }
