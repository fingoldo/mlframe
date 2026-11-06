"""A whole FE family that raises is not silent: the fit goes on without its columns and the ledger carries one ``family_failed`` row with the exception."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._fe_rejection_ledger import FE_GATE_LABELS, record_family_failure


def test_family_failure_row_carries_the_family_and_the_exception():
    """One row: the gate, the family as the operator, the exception type and message as the reason, and the step."""
    est = type("Est", (), {})()
    est._fe_steps_executed_ = 2
    record_family_failure(est, "offset_product", MemoryError("Unable to allocate 5.96 GiB"))
    (row,) = est._fe_rejection_records_
    assert row["gate"] == "family_failed" and "family_failed" in FE_GATE_LABELS
    assert row["operator"] == "offset_product" and row["candidate"] == "<family offset_product>"
    assert row["reason"].startswith("MemoryError: Unable to allocate") and row["step"] == 2


def test_offset_product_stage_swallows_a_memory_error_and_records_it(monkeypatch):
    """The stage keeps its contract (returns the frame it was given, the fit continues) and leaves the ``family_failed`` row."""
    import mlframe.feature_selection.filters._offset_product_fe as opfe
    from mlframe.feature_selection.filters._mrmr_fit_impl._fe_stage_offset_product import _stage_offset_product

    def _boom(*a, **k):
        """Stand-in for a scan that runs out of memory."""
        raise MemoryError("Unable to allocate 5.96 GiB")

    monkeypatch.setattr(opfe, "hybrid_offset_product_fe", _boom)
    est = type("Est", (), {"_fe_steps_executed_": 1})()
    X = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    out = _stage_offset_product(est, lambda *a, **k: True, X, np.arange(10) % 2, ["a", "b"], {}, 0)
    assert out is X
    rows = [r for r in est._fe_rejection_records_ if r["gate"] == "family_failed"]
    assert len(rows) == 1 and rows[0]["operator"] == "offset_product" and rows[0]["reason"].startswith("MemoryError")


def test_fit_with_a_family_that_runs_out_of_memory_keeps_the_row_in_the_public_ledger(monkeypatch):
    """End to end: the fit finishes without the family and ``fe_rejection_ledger_`` carries the ``family_failed`` row; the provenance table does not list the pseudo-candidate."""
    import mlframe.feature_selection.filters._offset_product_fe as opfe
    from mlframe.feature_selection.filters.mrmr import MRMR

    def _boom(*a, **k):
        """Stand-in for a scan that runs out of memory."""
        raise MemoryError("Unable to allocate 5.96 GiB (forced)")

    monkeypatch.setattr(opfe, "hybrid_offset_product_fe", _boom)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(6000, 8)), columns=[f"x{i}" for i in range(8)])
    y = (X["x0"] * X["x1"] + 0.3 * rng.normal(size=6000) > 0).astype(int)
    m = MRMR(verbose=0, random_seed=0, n_jobs=1, fe_max_steps=1, fe_offset_product_enable=True, full_npermutations=1, baseline_npermutations=1).fit(X, y)
    led = m.fe_rejection_ledger_
    rows = led[led["gate"] == "family_failed"]
    assert len(rows) == 1 and rows.iloc[0]["operator"] == "offset_product" and rows.iloc[0]["reason"].startswith("MemoryError")
    assert not any(str(n).startswith("<family") for n in m.fe_provenance_["feature_name"])
