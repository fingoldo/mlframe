"""The content-hash memo keeps the hash of every live external-factor column across a pair sweep (more than the old 64-entry cap), so a column is hashed once per fit."""

import numpy as np

from mlframe.feature_selection.filters import _fe_resident_operands as R


def test_hash_memo_holds_more_than_sixty_four_live_arrays(monkeypatch):
    """Two passes over 120 live arrays hash each array once."""
    R.clear_hash_memo()
    calls = []
    real = R._content_hash
    monkeypatch.setattr(R, "_content_hash", lambda h: calls.append(1) or real(h))
    arrs = [np.full(100, float(i)) for i in range(120)]
    for _ in range(2):
        for a in arrs:
            R._content_hash_memoized(a)
    assert len(calls) == 120
    R.clear_hash_memo()


def test_entry_goes_away_with_its_array():
    """A collected array leaves no memo entry, so the table follows the live columns."""
    import gc

    R.clear_hash_memo()
    a = np.arange(50.0)
    R._content_hash_memoized(a)
    assert len(R._HASH_MEMO) == 1
    del a
    gc.collect()
    assert len(R._HASH_MEMO) == 0
