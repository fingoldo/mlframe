"""Regression test for the per-operand single-corr memoization in ``score_prospective_pairs``
(``_step_pairs_rank.py``, 2026-07-11 perf fix).

``pair_is_tail_concentrated_rankaware`` runs once per candidate PAIR (~85k calls in a wide 100k-row
production FE fit) against a target fixed for the whole call. Internally it calls
``usability_form_corrs``, which computes a SINGLE-operand baseline (``_cs``) as the max of 4
correlations: each operand's raw value and its square against y. Two of those four (one operand's own
forms) are IDENTICAL every time that operand recurs in a different pair -- the same redundancy pattern
``_cached_operand`` (a prior fix in this same file) already closed for the raw operand EXTRACTION, just
one layer further down the pipeline, for the CORRELATION result.

``_single_operand_usability_corr(y, x)`` exposes that single-operand half separately (mathematically:
``max(single_corr(x0), single_corr(x1)) == _cs``, exact for float comparisons -- no accumulation), so a
per-operand-index cache can compute it ONCE and reuse it across every pair that references that operand,
then pass the pair `(cached(x0), cached(x1))` to ``usability_form_corrs``/``pair_is_tail_concentrated_rankaware``
via their new ``precomputed_single_corr`` parameter.

This test pins the memoization pattern directly (mirroring ``_cached_operand``'s own test file) rather
than invoking ``score_prospective_pairs`` itself: identical values, fewer underlying calls, and the
combined-with-precomputed-value call is bit-identical to the uncached one."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_usability_signal import (
    _single_operand_usability_corr,
    pair_is_tail_concentrated_rankaware,
    usability_form_corrs,
)


def _make_cached_single_corr(y, operands, call_log):
    """Make cached single corr."""
    cache: dict = {}

    def _cached(idx):
        """Helper that cached."""
        if idx in cache:
            return cache[idx]
        call_log.append(idx)
        val = _single_operand_usability_corr(y, operands[idx])
        cache[idx] = val
        return val

    return _cached


def test_cached_single_corr_returns_identical_values_to_uncached():
    """Cached single corr returns identical values to uncached."""
    rng = np.random.default_rng(10)
    n = 800
    y = rng.standard_normal(n)
    operands = {i: rng.standard_normal(n) + i for i in range(10)}

    call_log: list = []
    cached = _make_cached_single_corr(y, operands, call_log)

    for idx in (0, 3, 0, 7, 3, 0, 9):
        got = cached(idx)
        want = _single_operand_usability_corr(y, operands[idx])
        assert got == want


def test_cached_single_corr_avoids_redundant_computation():
    """The whole point of the fix: repeated references to the SAME operand index must not re-derive."""
    rng = np.random.default_rng(11)
    n = 800
    y = rng.standard_normal(n)
    operands = {i: rng.standard_normal(n) + i for i in range(5)}

    call_log: list = []
    cached = _make_cached_single_corr(y, operands, call_log)

    # Simulate the real access pattern: each operand referenced by many candidate pairs.
    accesses = [0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0]
    for idx in accesses:
        cached(idx)

    unique_indices = set(accesses)
    assert len(call_log) == len(unique_indices), (
        f"expected exactly one underlying derivation per unique operand ({len(unique_indices)}), "
        f"got {len(call_log)} -- caching is not eliminating redundant calls"
    )
    assert set(call_log) == unique_indices


def test_cached_values_feed_bit_identical_downstream_results():
    """End-to-end: usability_form_corrs and pair_is_tail_concentrated_rankaware, fed the CACHED per-operand
    values via precomputed_single_corr, must produce exactly what the uncached internal computation does --
    the caching wiring must be transparent to the actual selection decision."""
    rng = np.random.default_rng(12)
    n = 2000
    y = rng.standard_normal(n)
    operands = {0: rng.standard_normal(n) * 2.0 + 1.0, 1: rng.standard_normal(n) - 3.0, 2: rng.standard_normal(n) * 0.5}

    call_log: list = []
    cached = _make_cached_single_corr(y, operands, call_log)

    pairs = [(0, 1), (1, 2), (0, 2)]
    for i0, i1 in pairs:
        x0, x1 = operands[i0], operands[i1]
        sc0, sc1 = cached(i0), cached(i1)

        internal_cp, internal_cs = usability_form_corrs(x0=x0, x1=x1, y=y)
        cached_cp, cached_cs = usability_form_corrs(x0=x0, x1=x1, y=y, precomputed_single_corr=(sc0, sc1))
        assert cached_cs == internal_cs
        assert cached_cp == internal_cp

        internal_decision = pair_is_tail_concentrated_rankaware(y, x0, x1, min_corr=0.3, pairness_margin=1.0)
        cached_decision = pair_is_tail_concentrated_rankaware(
            y,
            x0,
            x1,
            min_corr=0.3,
            pairness_margin=1.0,
            precomputed_single_corr=(sc0, sc1),
        )
        assert cached_decision == internal_decision

    # Only 3 unique operand indices referenced across 3 pairs sharing operands pairwise -> exactly 3 derivations.
    assert len(call_log) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
