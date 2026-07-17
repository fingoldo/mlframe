"""Regression sensor for iter110: rankgauss average-tie-rank uses ONE searchsorted sweep on continuous (untied) data.

Pre-fix `apply_rankgauss` / `generate_rankgauss_features` always ran two `np.searchsorted` sweeps (side left + right) to
form `(lo + hi - 1)/2`. On continuous data the two sweeps are identical, so the second was pure waste. `_avg_tie_rank`
collapses to a single sweep + a cheap tie probe, falling back to the exact two-sweep path only when a real tie exists.
The output is bit-identical on both continuous and tied inputs; this test pins (a) the single-sweep fast path on
continuous data, (b) the two-sweep exact path on tied data, and (c) bit-identity against the reference on both.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _extra_fe_families as fam
from mlframe.feature_selection.filters._extra_fe_families import _avg_tie_rank


def _reference_two_sweep(fit_sorted: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Reference two sweep."""
    lo = np.searchsorted(fit_sorted, vals, side="left")
    hi = np.searchsorted(fit_sorted, vals, side="right")
    return (lo + hi - 1) / 2.0


def _count_searchsorted(monkeypatch):
    """Count searchsorted."""
    calls = {"n": 0}
    real = np.searchsorted

    def spy(a, v, side="left", sorter=None):
        """Helper that spy."""
        calls["n"] += 1
        return real(a, v, side=side, sorter=sorter)

    monkeypatch.setattr(fam.np, "searchsorted", spy)
    return calls


def test_avg_tie_rank_single_sweep_on_continuous(monkeypatch):
    """Avg tie rank single sweep on continuous."""
    rng = np.random.default_rng(0)
    fit_sorted = np.sort(rng.standard_normal(5000))
    vals = rng.standard_normal(20000)  # continuous -> no exact ties with the fit values
    ref = _reference_two_sweep(fit_sorted, vals)

    calls = _count_searchsorted(monkeypatch)
    out = _avg_tie_rank(fit_sorted, vals)

    assert calls["n"] == 1, f"continuous path must use ONE searchsorted sweep, used {calls['n']}"
    assert np.array_equal(out, ref), "continuous fast path diverged from the exact two-sweep reference"


def test_avg_tie_rank_two_sweep_on_tied():
    """Avg tie rank two sweep on tied."""
    rng = np.random.default_rng(0)
    fit_sorted = np.sort(rng.integers(0, 50, 5000).astype(np.float64))
    vals = rng.integers(0, 50, 20000).astype(np.float64)  # heavy exact ties
    ref = _reference_two_sweep(fit_sorted, vals)
    out = _avg_tie_rank(fit_sorted, vals)
    assert np.array_equal(out, ref), "tied exact path diverged from the two-sweep reference"


def test_avg_tie_rank_two_sweep_runs_when_tie_present(monkeypatch):
    """Avg tie rank two sweep runs when tie present."""
    fit_sorted = np.array([0.0, 1.0, 2.0, 3.0])
    vals = np.array([0.5, 1.0, 2.5])  # 1.0 ties a fit value -> exact two-sweep path
    calls = _count_searchsorted(monkeypatch)
    out = _avg_tie_rank(fit_sorted, vals)
    assert calls["n"] == 2, f"tied input must trigger the exact two-sweep path, used {calls['n']}"
    assert np.array_equal(out, _reference_two_sweep(fit_sorted, vals))
