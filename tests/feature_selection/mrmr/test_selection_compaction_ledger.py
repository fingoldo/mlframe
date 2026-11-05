"""The selection compaction writes its decisions to the rejection ledger, which is what fe_provenance_.dropped_by reads."""

from types import SimpleNamespace

from mlframe.feature_selection.filters._mrmr_fe_provenance import _output_membership
from mlframe.feature_selection.filters._mrmr_fit_impl._selection_compaction import _record_compaction


def _fitted():
    """A stand-in estimator holding what the ledger and the provenance read."""
    est = SimpleNamespace(_fe_rejection_records_=[], fe_rejection_ledger_cap=None)
    est.get_feature_names_out = lambda: ["x", "z"]
    return est


def test_redundant_and_replaced_features_reach_the_ledger_and_dropped_by():
    """A pruned feature and a replaced one are recorded under gate ``selection_compaction``; the provenance reports the gate and the reason names the replacement."""
    est = _fitted()
    _record_compaction(est, "d", None)
    _record_compaction(est, "b", "z")
    gates = {r["candidate"]: (r["gate"], r["reason"]) for r in est._fe_rejection_records_}
    assert gates["d"][0] == gates["b"][0] == "selection_compaction"
    assert "replaced by z" in gates["b"][1] and "replaced" not in gates["d"][1]
    _names, last_gate = _output_membership(est)
    assert last_gate == {"d": "selection_compaction", "b": "selection_compaction"}
