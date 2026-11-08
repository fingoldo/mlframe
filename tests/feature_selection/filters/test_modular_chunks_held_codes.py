"""``_ModularChunks`` keeps generated candidate codes while they fit the held-codes budget and regenerates them beyond it."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _periodic_fe as pf


def _chunks(monkeypatch, budget: int):
    """A two-chunk family whose code generator counts its calls."""
    calls: list = []

    def fake(X, cs, periods, nbins, family_elements):
        """Codes of the chunk's source columns, one row per column."""
        calls.append(tuple(cs))
        return [f"c_{c}" for c in cs], np.full((len(cs), len(X)), len(calls) % 100, dtype=np.int8), np.full(len(cs), 4)

    monkeypatch.setattr(pf, "_chunk_family_codes", fake)
    monkeypatch.setattr(pf, "_held_codes_budget", lambda: budget)
    X = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0), "c": np.arange(50.0), "d": np.arange(50.0)})
    return pf._ModularChunks(X, ["a", "b", "c", "d"], 2, (7.0,), 10, np.int8), calls


def test_chunks_within_the_budget_are_generated_once(monkeypatch):
    """Reading the two chunks alternately three times generates each of them once."""
    chunks, calls = _chunks(monkeypatch, budget=10**6)
    for _ in range(3):
        chunks.codes(0)
        chunks.codes(1)
    assert calls == [("a", "b"), ("c", "d")]


def test_chunks_beyond_the_budget_are_regenerated(monkeypatch):
    """With no budget only the last chunk is kept, so alternating reads regenerate every time (the memory bound of the 10 million row fit)."""
    chunks, calls = _chunks(monkeypatch, budget=0)
    for _ in range(3):
        chunks.codes(0)
        chunks.codes(1)
    assert len(calls) == 6
