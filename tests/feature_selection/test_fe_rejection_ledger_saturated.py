"""``fe_rejection_ledger_saturated`` is True only once the cap marker is in the ledger, so skipping the call after it never loses a record."""

import types

from mlframe.feature_selection.filters._fe_rejection_ledger import fe_rejection_ledger_saturated, record_fe_rejection


def test_saturation_follows_the_cap_marker():
    """Below the cap and at exactly the cap the ledger is not saturated (the next call adds the marker); after the marker it is."""
    est = types.SimpleNamespace(fe_rejection_ledger_cap=5, _fe_rejection_records_=None)
    assert fe_rejection_ledger_saturated(est) is False
    for i in range(5):
        assert fe_rejection_ledger_saturated(est) is False
        record_fe_rejection(est, gate="g", candidate=f"c{i}")
    assert len(est._fe_rejection_records_) == 5
    assert fe_rejection_ledger_saturated(est) is False  # exactly at the cap: the marker is still to come
    record_fe_rejection(est, gate="g", candidate="over")
    assert len(est._fe_rejection_records_) == 6 and est._fe_rejection_records_[-1]["gate"] == "ledger_capped"
    assert fe_rejection_ledger_saturated(est) is True
    before = list(est._fe_rejection_records_)
    record_fe_rejection(est, gate="g", candidate="dropped")
    assert est._fe_rejection_records_ == before


def test_a_guarded_caller_leaves_the_same_ledger_as_an_unguarded_one():
    """Recording 20 candidates through the guard gives the records of recording them all unguarded."""
    guarded = types.SimpleNamespace(fe_rejection_ledger_cap=7, _fe_rejection_records_=None)
    plain = types.SimpleNamespace(fe_rejection_ledger_cap=7, _fe_rejection_records_=None)
    for i in range(20):
        if not fe_rejection_ledger_saturated(guarded):
            record_fe_rejection(guarded, gate="g", candidate=f"c{i}", observed=0.1 * i, threshold=0.5)
        record_fe_rejection(plain, gate="g", candidate=f"c{i}", observed=0.1 * i, threshold=0.5)
    assert guarded._fe_rejection_records_ == plain._fe_rejection_records_
