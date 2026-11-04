"""A pair the family-wise gate rejects leaves a record in the rejection ledger."""

from mlframe.feature_selection.filters._fe_rejection_ledger import FE_GATE_LABELS
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_score_helpers import record_family_gate_rejection


def test_family_gate_rejection_is_recorded_with_its_own_registered_gate():
    """The record names the family gate, the winner's operator and MI, and goes to both the pair list and the side channel."""
    records: list = []
    side: list = []
    record_family_gate_rejection((("a", "b"), "mul", 3), 0.12, (0, 1), ["fa", "fb"], records, side)
    assert records == side and len(records) == 1
    rec = records[0]
    assert rec["gate"] == "pair_family_maxt" and rec["gate"] in FE_GATE_LABELS
    assert rec["operator"] == "mul" and rec["observed"] == 0.12 and rec["operand_names"] == "(fa, fb)"
