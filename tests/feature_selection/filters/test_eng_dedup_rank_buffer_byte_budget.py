"""The engineered-dedup rank buffer must be lazy and byte-bounded (mrmr_audit_2026-09-14 FEC-4).

``scan_engineered_duplicates`` allocated its rank buffer as a K x n float64 ``np.empty`` at function entry, sized
by the TOTAL appended-column count before any column was inspected -- 3.2 GB at the module's cited ~200 columns
and n=2M, 160 GB at n=100M, and on Windows ``np.empty`` commits pages against the paging file. Rows are only
ever written for fully-finite ADMITTED columns. The buffer is now allocated lazily, grown geometrically, and
capped by ``_RANK_BUF_MAX_BYTES``; a column that does not fit is compared by the existing per-pair path, so the
keep/drop decision must not change.

Allocations are observed through a numpy proxy installed only as ``_eng_dedup_scan``'s module-level ``np`` --
a global ``numpy.empty`` spy would also catch the ``(2, n)`` scratch arrays pandas/numpy allocate inside
``rank`` / ``corrcoef`` and misreport them as buffer rows.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._mrmr_fit_impl import _eng_dedup_scan as M


class _NumpyProxy:
    """Forwards everything to numpy but records the shape of each ``empty`` call made through it."""

    def __init__(self, real):
        self._real = real
        self.empty_shapes: list[tuple] = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def empty(self, shape, *args, **kwargs):
        """Record the requested shape, then allocate for real."""
        self.empty_shapes.append(tuple(shape) if isinstance(shape, (tuple, list)) else (shape,))
        return self._real.empty(shape, *args, **kwargs)


_N = 3000


@pytest.fixture(scope="module")
def frame():
    """Monotone-duplicate triples (fully finite), a NaN column plus its duplicate, a constant, and noise."""
    rng = np.random.default_rng(0)
    d = {}
    for j in range(6):
        b = rng.normal(size=_N)
        d[f"e{j}_lin"], d[f"e{j}_cube"], d[f"e{j}_shift"] = b, b**3, b * 2 + 5
    d["noisy_nan"] = np.where(rng.random(_N) < 0.1, np.nan, rng.normal(size=_N))
    d["noisy_nan_dup"] = d["noisy_nan"] * 3.0
    d["const"] = np.ones(_N)
    for k in range(8):
        d[f"noise{k}"] = rng.normal(size=_N)
    return pd.DataFrame(d)


_PREFS = {"first-wins": lambda cand, kept: False, "displace": lambda cand, kept: cand > kept}


def _scan(frame, monkeypatch, prefer, budget_bytes=None):
    """Run the scan under an optional byte budget, returning (keep, drop, buffer-shaped allocations)."""
    proxy = _NumpyProxy(np)
    monkeypatch.setattr(M, "np", proxy)
    if budget_bytes is not None:
        monkeypatch.setattr(M, "_RANK_BUF_MAX_BYTES", budget_bytes)
    keep, drop, _arrs, _ranks = M.scan_engineered_duplicates(frame, list(frame.columns), set(), prefer)
    buffers = [s for s in proxy.empty_shapes if len(s) == 2 and s[1] == len(frame)]
    return keep, drop, buffers


@pytest.mark.parametrize("pref_name", list(_PREFS))
@pytest.mark.parametrize("budget_rows", [0, 1, 2, 5])
def test_keep_and_drop_are_unchanged_under_a_tight_budget(frame, monkeypatch, pref_name, budget_rows):
    """Shrinking the budget may only shrink BATCHING coverage; the dedup decision itself must not move.

    The unbounded run is the reference: with room for every row it behaves exactly as the old upfront buffer did.
    Both preference policies are exercised so the displace branch -- which writes a buffer row for the WINNING
    candidate after evicting kept columns -- is covered as well as the plain admit branch.
    """
    prefer = _PREFS[pref_name]
    ref_keep, ref_drop, _ = _scan(frame, monkeypatch, prefer, budget_bytes=1 << 40)
    keep, drop, _ = _scan(frame, monkeypatch, prefer, budget_bytes=budget_rows * 8 * len(frame))
    assert keep == ref_keep
    assert drop == ref_drop


@pytest.mark.parametrize("budget_rows", [1, 2, 3, 7])
def test_the_rank_buffer_never_exceeds_the_byte_budget(frame, monkeypatch, budget_rows):
    """No buffer allocation may be larger than the budget allows, at any growth step."""
    budget = budget_rows * 8 * len(frame)
    _keep, _drop, buffers = _scan(frame, monkeypatch, _PREFS["displace"], budget_bytes=budget)
    assert buffers, "the fixture has fully-finite admitted columns, so a buffer should have been allocated"
    assert max(rows for rows, _n in buffers) * 8 * len(frame) <= budget


def test_a_zero_budget_never_allocates_the_buffer(frame, monkeypatch):
    """With no room for even one row, the scan must fall back to per-pair entirely and allocate nothing."""
    _keep, _drop, buffers = _scan(frame, monkeypatch, _PREFS["first-wins"], budget_bytes=8 * len(frame) - 1)
    assert buffers == []


def test_allocation_is_lazy_not_sized_by_every_appended_column(frame, monkeypatch):
    """The old code allocated exactly one row per APPENDED column up front; lazy growth must stay below that.

    Many columns here are dropped as duplicates, carry NaNs, or are constant, so fewer fully-finite columns are
    ever admitted than were appended. Reverting to the upfront ``(K, n)`` allocation makes this fail.
    """
    _keep, _drop, buffers = _scan(frame, monkeypatch, _PREFS["first-wins"], budget_bytes=1 << 40)
    assert buffers, "expected at least one lazily grown buffer"
    assert max(rows for rows, _n in buffers) < len(frame.columns)
