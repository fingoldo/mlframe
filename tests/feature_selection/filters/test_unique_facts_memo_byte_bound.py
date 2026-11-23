"""The distinct-value memo of the binning is bounded by bytes as well as by entries (PLAN-10: 256 entries of a 500000-row column were 1.5 GB)."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _adaptive_nbins_columns as C


def test_the_memo_never_holds_more_than_its_byte_budget(monkeypatch):
    """Columns of 4000 rows (48 KB of facts each) against a 200 KB budget: at most four stay, the newest among them."""
    monkeypatch.setattr(C, "_UNIQUE_FACTS_MAX_BYTES", 200 * 1024)
    C._UNIQUE_FACTS.clear()
    rng = np.random.default_rng(0)
    sigs = []
    for i in range(12):
        col = rng.standard_normal(4000).astype(np.float32)
        sig = ("col", i)
        sigs.append(sig)
        C._unique_facts(col, sig)
    held = sum(int(a.nbytes) + int(b.nbytes) for a, b in C._UNIQUE_FACTS.values())
    assert held <= 200 * 1024
    assert sigs[-1] in C._UNIQUE_FACTS and sigs[0] not in C._UNIQUE_FACTS
    C._UNIQUE_FACTS.clear()
