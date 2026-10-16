"""Falsy-but-valid inputs must not be rewritten to a default by an ``or`` fallback."""
from types import SimpleNamespace

from mlframe.feature_selection.filters._fe_rejection_ledger import FE_REJECTION_LEDGER_CAP, record_fe_rejection
from mlframe.feature_selection.wrappers.rfecv._fit_fold import _resolve_n_repeats


def test_ledger_cap_zero_is_honoured_not_replaced_by_module_default():
    """An explicit cap of 0 stops recording at once instead of silently becoming the 50k default."""
    owner = SimpleNamespace(fe_rejection_ledger_cap=0)
    record_fe_rejection(owner, gate="g", candidate="c", reason="r")
    record_fe_rejection(owner, gate="g", candidate="c", reason="r")
    records = owner._fe_rejection_records_
    assert len(records) == 1 and records[0]["gate"] == "ledger_capped"


def test_ledger_cap_none_uses_module_default():
    """Negative control: an unset cap still means the module default."""
    owner = SimpleNamespace(fe_rejection_ledger_cap=None)
    record_fe_rejection(owner, gate="g", candidate="c", reason="r")
    assert owner._fe_rejection_records_[0]["gate"] == "g"
    assert FE_REJECTION_LEDGER_CAP > 1


def test_resolve_n_repeats_prefers_effective_then_user_then_five():
    """Effective value wins, then ``n_repeats``, and only a missing/zero value falls back to 5."""
    assert _resolve_n_repeats(SimpleNamespace(_effective_n_repeats=3, n_repeats=9)) == 3
    assert _resolve_n_repeats(SimpleNamespace(n_repeats=9)) == 9
    assert _resolve_n_repeats(SimpleNamespace()) == 5
    assert _resolve_n_repeats(SimpleNamespace(_effective_n_repeats=None, n_repeats=0)) == 5
