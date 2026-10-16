"""local_mi_gate drops a candidate whose MI cannot be computed; that drop must reach the reject sink, not vanish."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _orthogonal_univariate_fe as orth
from mlframe.feature_selection.filters._unified_fe_gate import local_mi_gate


def _run(monkeypatch, mis, floor=0.1):
    """Run the gate with a stubbed batch MI returning ``mis`` and return (kept, sink records)."""
    y = np.arange(30) % 2

    def _stub(arr, y_arg, nbins=10):
        """The observed y gets ``mis``; a permuted y (the family null) gets a tiny MI so the null does not raise the floor."""
        same = np.array_equal(np.asarray(y_arg), y)
        return np.asarray(mis if same else [0.001] * len(mis), dtype=np.float64)

    monkeypatch.setattr(orth, "_mi_classif_batch", _stub)
    enc = pd.DataFrame({f"c{i}": np.arange(30, dtype=float) + i for i in range(len(mis))})
    records: list[dict] = []
    kept = local_mi_gate(enc, y, floor=floor, reject_sink=lambda **kw: records.append(kw))
    return kept, records


def test_non_finite_mi_is_dropped_and_recorded(monkeypatch):
    """A NaN and an inf MI are not kept and each produces a reject record with the non-finite reason."""
    kept, records = _run(monkeypatch, [0.9, np.nan, np.inf, 0.01])
    assert kept == ["c0"]
    by_candidate = {r["candidate"]: r for r in records}
    assert set(by_candidate) == {"c1", "c2", "c3"}
    assert "non-finite" in by_candidate["c1"]["reason"] and "non-finite" in by_candidate["c2"]["reason"]
    assert "below" in by_candidate["c3"]["reason"] and by_candidate["c3"]["observed"] < by_candidate["c3"]["threshold"]


def test_finite_survivors_and_clean_runs_record_nothing_extra(monkeypatch):
    """Negative control: with every MI finite and above the floor nothing is recorded and everything is kept in MI order."""
    kept, records = _run(monkeypatch, [0.3, 0.9, 0.5])
    assert kept == ["c1", "c2", "c0"]
    assert records == []
