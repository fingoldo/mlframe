"""Regression (MRMR critique N-F3): the ``_perm_pvalue(full_budget=)`` extrapolation is SELECTION-INERT, not anti-conservative.

N-F3 worried the full-budget denominator overstates confidence on early-break pile-ups. Calibration evidence
(bench_nf3_perm_pvalue_calibration.py) shows it cannot flip a selection decision, because of a structural invariant:

  get_fleuret_criteria_confidence breaks early ONLY when nfailed >= max_failed (fleuret.py:299), and parallel_fleuret then
  zeroes the candidate's gain (fleuret.py:123). So whenever nchecked < npermutations (the only case full_budget changes the
  denominator), the candidate is ALREADY rejected; whenever it is kept, nchecked == npermutations so full_budget == nchecked
  and the surfaced p is identical either way.

These tests pin that invariant so a future change to the early-break condition (that could make full_budget engage on a KEPT
candidate) trips here.
"""
import numpy as np
from numba.core import types
from numba.typed import Dict as NumbaDict

from mlframe.feature_selection.filters.fleuret import get_fleuret_criteria_confidence
from mlframe.feature_selection.filters.permutation import _perm_pvalue
from mlframe.feature_selection.filters.info_theory._entropy_kernels import conditional_mi


def _cache():
    return NumbaDict.empty(key_type=types.unicode_type, value_type=types.float64)


def _observed_gain(data, nbins):
    """Observed conditional gain I(x1; y | x0) the confirm loop tests against; under a true null it sits inside the
    permutation distribution so failures accrue and the loop hits max_failed (the early-break pile-up N-F3 is about)."""
    return conditional_mi(
        factors_data=data, x=np.asarray([1], dtype=np.int64), y=np.asarray([3], dtype=np.int64),
        z=np.asarray([0], dtype=np.int64), var_is_nominal=None, factors_nbins=nbins,
    )


def _true_null(n, seed):
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n).astype(np.int32)
    x1 = rng.integers(0, 3, n).astype(np.int32)
    x2 = rng.integers(0, 2, n).astype(np.int32)
    y = rng.integers(0, 2, n).astype(np.int32)
    return np.column_stack([x0, x1, x2, y]).astype(np.int32), np.array([2, 3, 2, 2], dtype=np.int64)


def test_early_break_implies_rejected_so_full_budget_is_selection_inert():
    npermutations, min_conf = 64, 0.5
    max_failed = max(1, int(npermutations * (1 - min_conf)))
    y_arr = np.asarray((3,), dtype=np.int64)
    saw_early_break = False
    saw_kept = False
    for seed in range(120):
        data, nbins = _true_null(1500, seed)
        nfailed, nchecked = get_fleuret_criteria_confidence(
            data_copy=data.copy(), factors_nbins=nbins, x=(1,), y=y_arr, selected_vars=[0],
            npermutations=npermutations, bootstrapped_gain=_observed_gain(data, nbins), max_failed=max_failed, nexisting=0,
            mrmr_relevance_algo="fleuret", mrmr_redundancy_algo="fleuret", max_veteranes_interactions_order=1,
            cached_cond_MIs=_cache(), entropy_cache=_cache(), extra_x_shuffling=True,
            base_seed=np.uint64(seed * 2654435761 + 1),
        )
        if nchecked < npermutations:
            saw_early_break = True
            assert nfailed >= max_failed, f"seed={seed}: early break WITHOUT reject condition -> full_budget would be anti-conservative"
        else:
            saw_kept = True
            # kept candidate: nchecked == budget, so full_budget denominator == truncated denominator (surfaced p identical)
            assert _perm_pvalue(nfailed, nchecked, full_budget=npermutations) == _perm_pvalue(nfailed, nchecked, full_budget=None)
    assert saw_early_break, "fixture did not exercise the early-break path"
    assert saw_kept, "fixture did not exercise the kept path"


def test_full_budget_only_changes_p_when_denominator_grows():
    # full_budget lowers the p only when full_budget > nchecked (the early-break case); on a full run it is a no-op.
    assert _perm_pvalue(5, 8, full_budget=100) < _perm_pvalue(5, 8, full_budget=None)
    assert _perm_pvalue(5, 100, full_budget=100) == _perm_pvalue(5, 100, full_budget=None)
