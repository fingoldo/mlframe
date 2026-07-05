"""Calibration bench (critique N-F3): does ``_perm_pvalue(full_budget=)`` overstate confidence / inflate the false-positive rate on early-break pile-ups?

N-F3 worry: when the Fleuret confirmation loop breaks early because ``nfailed`` piled up to ``max_failed``, scoring the p against the FULL ``npermutations`` budget
(denominator = npermutations instead of the truncated ``nchecked``) makes the reported confidence higher (p lower) than the truncated ratio -- potentially anti-conservative,
letting a true-null feature look significant.

This bench measures, under a TRUE NULL (X independent of y) across many seeds, whether the full-budget denominator can flip a selection decision. The structural fact it
demonstrates: ``get_fleuret_criteria_confidence`` breaks early ONLY at ``nfailed >= max_failed`` (fleuret.py:299), and ``parallel_fleuret`` then ZEROES the candidate's gain
(fleuret.py:123). So whenever ``nchecked < npermutations`` (the only case where full_budget changes the p), the candidate is ALREADY rejected and its confidence is discarded;
whenever the candidate is kept (no early break), ``nchecked == npermutations`` so full_budget == nchecked and the p is IDENTICAL either way. The full-budget extrapolation is
therefore SELECTION-INERT: it cannot make a null feature clear the gate. This bench proves the invariant empirically and reports the empirical FPR (denominator-invariant by
construction) plus the surfaced-p calibration on the kept population.

Run:
  python -m mlframe.feature_selection.filters._benchmarks.bench_nf3_perm_pvalue_calibration
"""
import numpy as np
from numba.core import types
from numba.typed import Dict as NumbaDict

from mlframe.feature_selection.filters.fleuret import get_fleuret_criteria_confidence
from mlframe.feature_selection.filters.permutation import _perm_pvalue
from mlframe.feature_selection.filters.info_theory._entropy_kernels import conditional_mi


def _new_cache():
    return NumbaDict.empty(key_type=types.unicode_type, value_type=types.float64)


def _observed_gain(data, nbins):
    """Observed conditional gain the confirm loop tests against: I(x1; y | x0). Under a true null this is a small
    plug-in value sitting inside the permutation distribution, so ~half the shuffles tie/exceed it -> failures accrue and
    the loop hits max_failed and breaks early (the pile-up case N-F3 is about)."""
    return conditional_mi(
        factors_data=data, x=np.asarray([1], dtype=np.int64), y=np.asarray([3], dtype=np.int64),
        z=np.asarray([0], dtype=np.int64), var_is_nominal=None, factors_nbins=nbins,
    )


def _true_null_factors(n, seed):
    """All columns iid Bernoulli/multinomial and INDEPENDENT of the target column -> any conditional gain is pure sampling noise (true null)."""
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n).astype(np.int32)
    x1 = rng.integers(0, 3, n).astype(np.int32)
    x2 = rng.integers(0, 2, n).astype(np.int32)
    y = rng.integers(0, 2, n).astype(np.int32)  # independent of x0,x1,x2
    data = np.column_stack([x0, x1, x2, y]).astype(np.int32)
    nbins = np.array([2, 3, 2, 2], dtype=np.int64)
    return data, nbins


def run(n=1500, npermutations=64, min_nonzero_confidence=0.5, n_seeds=200, alpha=0.05):
    max_failed = max(1, int(npermutations * (1 - min_nonzero_confidence)))
    y_arr = np.asarray((3,), dtype=np.int64)

    invariant_ok = True
    early_break = 0
    kept = 0  # candidates NOT rejected (nfailed < max_failed) -> false positives under a true null
    p_lt_alpha_full = 0  # kept AND surfaced p (full-budget) < alpha
    p_lt_alpha_trunc = 0  # kept AND surfaced p (truncated denominator) < alpha
    p_diff_on_kept = 0  # kept runs where full-budget and truncated p disagree

    for seed in range(n_seeds):
        data, nbins = _true_null_factors(n, seed)
        obs_gain = _observed_gain(data, nbins)
        nfailed, nchecked = get_fleuret_criteria_confidence(
            data_copy=data.copy(), factors_nbins=nbins, x=(1,), y=y_arr,
            selected_vars=[0], npermutations=npermutations, bootstrapped_gain=obs_gain,
            max_failed=max_failed, nexisting=0, mrmr_relevance_algo="fleuret", mrmr_redundancy_algo="fleuret",
            max_veteranes_interactions_order=1, cached_cond_MIs=_new_cache(), entropy_cache=_new_cache(),
            extra_x_shuffling=True, base_seed=np.uint64(seed * 2654435761 + 1),
        )
        if nchecked < npermutations:
            early_break += 1
            if nfailed < max_failed:
                invariant_ok = False  # would be a real anti-conservative bug: early break WITHOUT the reject condition
        rejected = nfailed >= max_failed
        if not rejected:
            kept += 1
            p_full = _perm_pvalue(nfailed, nchecked, full_budget=npermutations)
            p_trunc = _perm_pvalue(nfailed, nchecked, full_budget=None)
            if p_full != p_trunc:
                p_diff_on_kept += 1
            if p_full < alpha:
                p_lt_alpha_full += 1
            if p_trunc < alpha:
                p_lt_alpha_trunc += 1

    print("=== N-F3 permutation p-value full-budget calibration (true null) ===")
    print(f"seeds={n_seeds}  npermutations={npermutations}  max_failed={max_failed}  alpha={alpha}  (bootstrapped_gain = per-seed observed I(x1;y|x0))")
    print(f"early-break runs (nchecked<budget): {early_break}/{n_seeds}")
    print(f"INVARIANT (early break => nfailed>=max_failed => candidate rejected/gain-zeroed): {'HOLDS' if invariant_ok else 'VIOLATED'}")
    print(f"kept candidates (nfailed<max_failed = false positives): {kept}/{n_seeds}  empirical FPR={kept / n_seeds:.4f}")
    print(f"  of kept: runs where full-budget p != truncated p: {p_diff_on_kept} (0 == full_budget is INERT on kept candidates)")
    print(f"  of kept: full-budget p<alpha: {p_lt_alpha_full}   truncated p<alpha: {p_lt_alpha_trunc}")
    print("NOTE: the reject decision uses the RAW nfailed>=max_failed count, NOT the p-value; on kept candidates nchecked==budget so full_budget==nchecked.")
    verdict = ("CORRECTLY CALIBRATED -> DOC (full_budget is selection-inert; identical FPR and identical p on kept candidates)"
               if invariant_ok and p_diff_on_kept == 0 else "ANTI-CONSERVATIVE -> implement calibrated denominator")
    print(f"VERDICT: {verdict}")
    return invariant_ok, p_diff_on_kept, kept / n_seeds


if __name__ == "__main__":
    run()
