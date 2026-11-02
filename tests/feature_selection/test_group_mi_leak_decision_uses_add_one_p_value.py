"""The group-MI leak decision is an add-one permutation p-value test (tracker 2026-11-02 wave 12, STAT12-4)."""

import numpy as np

from mlframe.feature_selection.filters.info_theory import _group_mi as gm


def test_value_between_two_order_statistics_is_still_a_leak(monkeypatch):
    """With 199 permuted values 1..199 the add-one rule needs the observed value to beat the 190th; 189.5 used to pass as signal."""
    values = iter([189.5] + [float(v) for v in range(1, 200)])
    monkeypatch.setattr(gm, "group_blocked_mi", lambda *a, **k: next(values))
    codes = np.zeros(40, dtype=np.int64)
    offsets = np.array([0, 20, 40])
    observed, is_leak = gm.group_leak_verdict(codes, codes, np.arange(40), offsets, 2, 2, n_perm=199, alpha=0.05)
    assert observed == 189.5
    assert is_leak is True
