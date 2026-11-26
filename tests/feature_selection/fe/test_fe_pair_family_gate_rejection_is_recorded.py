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


def test_operand_names_of_the_records_are_resolved_from_the_stored_text_as_before():
    """The stored texts "(0, 1)", a name, a position past the raw columns, an empty tuple, no operands and a NaN resolve to the names the piecewise parse gave."""
    import numpy as np
    import pandas as pd

    from mlframe.feature_selection.filters._fe_rejection_ledger import _resolve_index_columns

    class _Est:
        """Estimator stub with the names the fit saw."""

        feature_names_in_ = np.array(["a", "b", "c"])

    def reference(text):
        """The former per-piece rule: a position is named or marked engineered, a name is kept."""
        if text is None:
            return ""
        if isinstance(text, float):
            return str(text)
        pieces = [int(p) if p.isdigit() else p for p in text[1:-1].split(", ") if p]
        return "(" + ", ".join((("a", "b", "c")[v] if 0 <= v < 3 else f"<engineered#{v}>") if isinstance(v, int) else v for v in pieces) + ")" if pieces else ""

    texts = ["(0, 1)", "(2)", "(1, 7, 0)", "(x1*x2, 0)", "()", "(5, 5)", None, "(0, 1)", "(12)"]
    frame = pd.DataFrame({"operands": pd.Series(texts, dtype=object)})
    _resolve_index_columns(_Est(), frame)
    assert list(frame["operand_names"]) == [reference(t) for t in texts]
    assert frame["operand_names"].iloc[0] == "(a, b)" and frame["operand_names"].iloc[2] == "(b, <engineered#7>, a)"
