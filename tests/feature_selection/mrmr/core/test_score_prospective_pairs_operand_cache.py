"""Regression test for the per-call operand memoization in ``score_prospective_pairs``
(``_step_pairs_rank.py``, 2026-07-10 perf fix).

Both loops in ``score_prospective_pairs`` call ``usability_operand_continuous`` twice per candidate
PAIR, but each raw operand appears in O(n_candidates) pairs -- the same column was re-extracted from
``X`` (pandas getitem + dtype cast + ravel) once per pair it participates in. Measured 170,160 calls /
3.27s cumtime on a 100k-row production profile, almost entirely this redundancy (the function is a pure
lookup of ``X``/``cols``/``self`` -- all fixed for one ``score_prospective_pairs`` call, so the result
for a given ``var_idx`` never changes within it).

Fixed with a local dict cache (``_cached_operand``), safe because the function is called serially (no
threading/loky dispatch at this level -- contrast the numba-typed-dict caches elsewhere in this package
that DO cross worker threads and need per-worker copies).

This test pins the memoization pattern directly (import the real ``usability_operand_continuous`` and
wrap it exactly as the fix does) rather than invoking ``score_prospective_pairs`` itself, which needs
many fixture-heavy kwargs to construct standalone -- the broader FE/usability test suite (running the
real MRMR pipeline through this code path) already covers integration; this test isolates and pins the
caching CONTRACT: identical values, fewer underlying calls."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._fe_usability_signal import (
    usability_operand_continuous,
)


def _make_cached_operand(self_obj, X, cols, call_log):
    cache: dict = {}

    def _cached_operand(idx):
        if idx in cache:
            return cache[idx]
        call_log.append(idx)
        val = usability_operand_continuous(self_obj, X, cols, idx)
        cache[idx] = val
        return val

    return _cached_operand


def test_cached_operand_returns_identical_values_to_uncached():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({f"c{i}": rng.standard_normal(n) for i in range(10)})
    cols = list(X.columns)
    self_obj = object()

    call_log: list = []
    cached = _make_cached_operand(self_obj, X, cols, call_log)

    for idx in (0, 3, 0, 7, 3, 0, 9):
        got = cached(idx)
        want = usability_operand_continuous(self_obj, X, cols, idx)
        np.testing.assert_array_equal(got, want)


def test_cached_operand_avoids_redundant_extraction():
    """The whole point of the fix: repeated references to the SAME operand index must not re-extract."""
    rng = np.random.default_rng(1)
    n = 500
    X = pd.DataFrame({f"c{i}": rng.standard_normal(n) for i in range(10)})
    cols = list(X.columns)
    self_obj = object()

    call_log: list = []
    cached = _make_cached_operand(self_obj, X, cols, call_log)

    # Simulate the real access pattern: each operand referenced by many candidate pairs.
    accesses = [0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0]
    for idx in accesses:
        cached(idx)

    unique_indices = set(accesses)
    assert len(call_log) == len(unique_indices), (
        f"expected exactly one underlying extraction per unique operand ({len(unique_indices)}), "
        f"got {len(call_log)} -- caching is not eliminating redundant calls"
    )
    assert set(call_log) == unique_indices


def test_cached_operand_none_result_is_cached_too():
    """An unresolvable operand (engineered, no raw position) returns None -- must be cached as None,
    not re-attempted on every reference (which would defeat the optimization for exactly the operands
    most likely to be probed repeatedly across a wide candidate pool)."""
    rng = np.random.default_rng(2)
    n = 500
    X = pd.DataFrame({"a": rng.standard_normal(n)})
    cols = ["a", "engineered_no_raw_position"]
    self_obj = object()

    call_log: list = []
    cached = _make_cached_operand(self_obj, X, cols, call_log)

    for _ in range(5):
        result = cached(1)  # "engineered_no_raw_position" is not a column in X -> None
        assert result is None

    assert len(call_log) == 1, "None results must be cached, not re-fetched on every access"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
