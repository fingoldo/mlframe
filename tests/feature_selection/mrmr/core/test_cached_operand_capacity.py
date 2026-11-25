"""The operand memo of the pair ranking holds every operand that fits a tenth of the free RAM, not only the entry floor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fe_step import _step_pairs_rank as sp


def test_cache_grows_past_the_entry_floor_when_ram_allows(monkeypatch):
    """With ample RAM a pool of 40 operands under a floor of 8 extracts each operand once over two passes."""
    n, k = 500, 40
    X = pd.DataFrame(np.random.default_rng(0).standard_normal((n, k)), columns=[f"c{i}" for i in range(k)])
    calls = []
    monkeypatch.setattr(sp, "_usability_operand_continuous", lambda self, X_, cols, idx: calls.append(idx) or np.asarray(X_.iloc[:, idx], dtype=np.float64))
    monkeypatch.setattr(sp, "_host_free_bytes_safe", lambda: 8 * 2**30)
    get = sp._make_cached_operand(None, X, list(X.columns), max_entries=8)
    for _ in range(2):
        for i in range(k):
            get(i)
    assert len(calls) == k


def test_entry_floor_still_bounds_when_ram_is_scarce(monkeypatch):
    """With no free RAM the floor of 8 entries applies and the second pass extracts again."""
    n, k = 500, 20
    X = pd.DataFrame(np.random.default_rng(0).standard_normal((n, k)), columns=[f"c{i}" for i in range(k)])
    calls = []
    monkeypatch.setattr(sp, "_usability_operand_continuous", lambda self, X_, cols, idx: calls.append(idx) or np.asarray(X_.iloc[:, idx], dtype=np.float64))
    monkeypatch.setattr(sp, "_host_free_bytes_safe", lambda: 0)
    get = sp._make_cached_operand(None, X, list(X.columns), max_entries=8)
    for _ in range(2):
        for i in range(k):
            get(i)
    assert len(calls) == 2 * k


def test_operands_that_are_views_of_the_frame_are_all_kept_whatever_the_entry_floor(monkeypatch):
    """A float32 column read at float32 is a view of the frame: it costs no memory, so a pool of 40 under a floor of 8 is extracted once over two passes even with no free RAM, while copies stay bounded."""
    n, k = 500, 40
    X = pd.DataFrame(np.random.default_rng(0).standard_normal((n, k)).astype(np.float32), columns=[f"c{i}" for i in range(k)])
    calls = []
    monkeypatch.setattr(sp, "_usability_operand_continuous", lambda self, X_, cols, idx: calls.append(idx) or np.asarray(X_[cols[idx]], dtype=np.float32))
    monkeypatch.setattr(sp, "_host_free_bytes_safe", lambda: 0)
    get = sp._make_cached_operand(None, X, list(X.columns), max_entries=8)
    first = [get(i) for i in range(k)]
    again = [get(i) for i in range(k)]
    assert len(calls) == k and all(a is b for a, b in zip(first, again))
    assert not first[0].flags.owndata
