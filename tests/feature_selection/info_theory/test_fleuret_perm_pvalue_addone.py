"""Regression tests for the CMIM / Fleuret permutation-confidence p-value (filters/fleuret.py).

SA2: ``get_fleuret_criteria_confidence_parallel`` reports a per-candidate confidence derived from the permutation
exceedance count. Two methodological defects are fixed:

  * MISSING ADD-ONE: the caller computed ``confidence = 1 - nfailed / nchecked``. On a null feature that never fails
    this returns confidence = 1.0 (p = 0), which is impossible for a finite Monte-Carlo permutation test -- the observed
    statistic is itself one draw under the null. The fix routes through the canonical ``_perm_pvalue`` estimator
    ``(1 + nfailed) / (1 + budget)`` (Phipson & Smyth 2010), so p is never exactly 0 / confidence never exactly 1.

  * EARLY-STOP BIAS: ``max_failed`` makes the worker loop break as soon as ``nfailed`` hits the cap, so ``nchecked`` is
    data-dependent and the stopped ratio overstates the failure rate (it stops precisely where failures cluster). The
    fix passes the full ``npermutations`` budget as the denominator so the reported p does not depend on WHERE the early
    break fired, and is never anti-conservative (smaller) than a full-budget reference.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.fleuret import get_fleuret_criteria_confidence_parallel
from mlframe.feature_selection.filters.permutation import _perm_pvalue


def _build_uncorrelated_factors(n: int = 2000, seed: int = 7):
    """Build uncorrelated factors."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 2, size=n).astype(np.int32) for _ in range(4)]
    factors_data = np.column_stack(cols).astype(np.int32)
    factors_nbins = np.array([2, 2, 2, 2], dtype=np.int64)
    return factors_data, factors_nbins


def test_addone_keeps_confidence_below_one_on_null_feature():
    """On a strong-signal candidate where NO permutation fails, the add-one floor must keep confidence strictly < 1.0
    (pre-fix ``1 - nfailed/nchecked`` returned exactly 1.0). Confidence == 1 - 1/(budget+1)."""
    rng = np.random.default_rng(42)
    n = 2000
    x0 = rng.integers(0, 2, size=n).astype(np.int32)
    x1 = rng.integers(0, 2, size=n).astype(np.int32)
    x2 = rng.integers(0, 2, size=n).astype(np.int32)
    y_col = np.bitwise_xor(x0, x1).astype(np.int32)
    factors_data = np.column_stack([x0, x1, x2, y_col]).astype(np.int32)
    factors_nbins = np.array([2, 2, 2, 2], dtype=np.int64)
    budget = 100
    _, conf, _ = get_fleuret_criteria_confidence_parallel(
        data_copy=factors_data,
        factors_nbins=factors_nbins,
        x=(1,),
        y=np.asarray([3], dtype=np.int64),
        selected_vars=[0],
        bootstrapped_gain=0.1,
        npermutations=budget,
        max_failed=budget,
        nexisting=0,
        cached_cond_MIs={},
        entropy_cache={},
        n_workers=1,
    )
    assert conf < 1.0, f"add-one must keep confidence below 1.0 on a no-fail candidate, got {conf}"
    # With 0 failures over the full budget, p = 1/(budget+1) and confidence = 1 - p.
    assert abs(conf - (1.0 - 1.0 / (budget + 1.0))) < 1e-9, f"expected add-one floor confidence, got {conf}"


def test_perm_pvalue_applies_addone_and_is_never_zero():
    """Unit pin on the canonical estimator wired into the fleuret caller: add-one applied, p strictly positive."""
    # No failures: pre-fix nfailed/nchecked == 0; add-one gives 1/(B+1).
    assert _perm_pvalue(0, 100) == 1.0 / 101.0
    assert _perm_pvalue(0, 100) > 0.0


def test_stopped_p_uses_full_budget_denominator():
    """The early-stop (``nchecked < npermutations`` because ``nfailed`` hit ``max_failed``) must score the p against the
    FULL ``npermutations`` budget, so the reported significance does not depend on WHERE the early break fired.

    A candidate that, say, accrues 5 failures over a 100-permutation budget but breaks early at nchecked=8 must report the
    same p as if it had been scored against the full budget -- the de-biased ``(1 + nfailed) / (1 + budget)`` -- not the
    break-position-dependent ``(1 + 5) / (1 + 8)``."""
    budget = 100
    p_stopped = _perm_pvalue(5, 8, full_budget=budget)
    p_full = _perm_pvalue(5, budget, full_budget=budget)
    assert p_stopped == p_full == (1.0 + 5) / (1.0 + budget), f"early-stopped p must be scored against the full budget: stopped={p_stopped} full={p_full}"
    # The break-position-dependent ratio (denominator = nchecked) is the biased estimate the fix avoids.
    assert p_stopped != (1.0 + 5) / (1.0 + 8)
