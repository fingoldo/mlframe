"""Interaction-information routing is default-OFF (bench-rejected as a default); a caller without the flag must not get it enabled.

The constructor default is ``fe_ii_routing_enable=False``, but the routing helper read the flag with ``getattr(..., True)``, so any
duck-typed estimator lacking the attribute ran the routing that the bench rejected.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection.filters import _interaction_information
from mlframe.feature_selection.filters._mrmr_fe_step_helpers import apply_interaction_information_routing


def test_ii_routing_default_off_for_bare_namespace(monkeypatch):
    """With no fe_ii_routing_enable attribute, 40 prospective pairs come back unchanged and the null floor is never computed."""
    calls = {"n": 0}

    def floor_spy(**kw):
        """Record that routing ran."""
        calls["n"] += 1
        return 0.0

    monkeypatch.setattr(_interaction_information, "pooled_pair_ii_null_floor", floor_spy)
    pairs = {((i, i + 1), 0.1 * i): 0 for i in range(40)}
    data = np.zeros((50, 42), dtype=np.int32)
    out = apply_interaction_information_routing(
        SimpleNamespace(), prospective_pairs=pairs, cached_MIs={}, nbins=np.full(42, 2), freqs_y=np.array([0.5, 0.5]),
        classes_y=np.zeros(50, dtype=np.int32), data=data, synergy_added_idx=set(), verbose=0,
    )
    assert out is pairs
    assert calls["n"] == 0, "interaction-information routing ran for an estimator that never enabled it"
