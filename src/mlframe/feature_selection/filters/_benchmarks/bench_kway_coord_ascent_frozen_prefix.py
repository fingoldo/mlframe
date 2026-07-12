"""Bench: precompute the k-1 FROZEN members once per (position, pass) in
``_refine_kway_coordinate_ascent``'s coordinate-ascent sweep, instead of
re-scanning the FULL k-member tuple from raw columns for every candidate.

WHY
---
For a fixed swap position ``pos``, only 1 of the k tuple members (the
candidate) changes across the whole ``for cand in candidate_pool`` sweep --
the other k-1 ("frozen") members are identical for every candidate. The OLD
code still called ``merge_vars`` on the FULL re-sorted k-tuple (frozen + cand)
for every candidate, re-deriving the frozen members' contribution from raw
columns every time.

``merge_vars``'s dense renumbering is ORDER-SENSITIVE (merging the same
variable set in a different order gives a bijective but numerically DIFFERENT
label array -- verified separately), so simply appending the candidate LAST
to a once-merged "frozen" state is only correct when the candidate happens to
sort after every frozen member. The general fix precomputes INCREMENTAL
prefix states of the frozen members (ascending order) once per position, then
splices each candidate into its correct SORTED insertion point via a tiny
"cand" merge + "suffix" merge -- reproducing the exact same merge order (and
therefore the exact same dense codes) as a fresh full-tuple merge, without
re-scanning the frozen members before the insertion point.

Bit-identical (verified): 150 randomized end-to-end trials comparing the
patched ``_refine_kway_coordinate_ascent`` against a reference re-implementation
of the pre-fix algorithm (varying n, cardinality, k, candidate-pool size,
max_combined_nbins, n_passes) produced zero classes/MI/tuple mismatches,
including the "accept mid-sweep invalidates the frozen state for this same
position" case (fixed by rebuilding the prefix states immediately after any
accepted swap, not just once per position).

Run:
  CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection.filters._benchmarks.bench_kway_coord_ascent_frozen_prefix
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes, merge_vars
from mlframe.feature_selection.filters._cat_post_refine import _refine_kway_coordinate_ascent


def _refine_old(factors_data, kway_results, candidate_pool, nbins, classes_y, freqs_y, max_combined_nbins, n_passes, dtype):
    """Pre-fix reference: full merge_vars over the re-sorted tuple every candidate."""
    if n_passes <= 0 or not kway_results:
        return kway_results
    refined = []
    for orig_tuple, orig_classes, orig_nuniq, orig_mi in kway_results:
        current = list(orig_tuple)
        current_mi = orig_mi
        current_classes = orig_classes
        current_nuniq = orig_nuniq
        for _ in range(n_passes):
            improved = False
            for pos in range(len(current)):
                for cand in candidate_pool:
                    cand_int = int(cand)
                    if cand_int in current:
                        continue
                    new_tuple = current.copy()
                    new_tuple[pos] = cand_int
                    new_tuple_sorted = tuple(sorted(new_tuple))
                    card = 1
                    for kk in new_tuple_sorted:
                        card *= int(nbins[kk])
                    if card > max_combined_nbins or card >= 2**31:
                        continue
                    new_classes, new_freqs, new_nuniq = merge_vars(
                        factors_data=factors_data,
                        vars_indices=np.array(new_tuple_sorted, dtype=np.int64),
                        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
                    )
                    new_mi = compute_mi_from_classes(
                        classes_x=new_classes, freqs_x=new_freqs,
                        classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
                    )
                    if new_mi > current_mi:
                        current = list(new_tuple_sorted)
                        current_mi = new_mi
                        current_classes = new_classes
                        current_nuniq = new_nuniq
                        improved = True
            if not improved:
                break
        refined.append((tuple(sorted(current)), current_classes, current_nuniq, current_mi))
    return refined


def _make_case(n, n_cols, n_bins, k, seed):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, n_bins, size=(n, n_cols)).astype(np.int32)
    nbins = np.full(n_cols, n_bins, dtype=np.int64)
    dtype = np.int32
    y_idx = n_cols - 1
    cls_y, fq_y, _ = merge_vars(
        factors_data=data, vars_indices=np.array([y_idx], dtype=np.int64),
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    seed_cols = rng.choice(n_cols - 1, size=k, replace=False)
    seed_tuple = tuple(sorted(int(c) for c in seed_cols))
    seed_classes, seed_freqs, seed_n = merge_vars(
        factors_data=data, vars_indices=np.array(seed_tuple, dtype=np.int64),
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    seed_mi = compute_mi_from_classes(seed_classes, seed_freqs, cls_y, fq_y, dtype)
    kway_results = [(seed_tuple, seed_classes, seed_n, seed_mi)]
    candidate_pool = np.array([c for c in range(n_cols - 1) if c not in seed_tuple], dtype=np.int64)
    return data, nbins, cls_y, fq_y, kway_results, candidate_pool, dtype


def _best_of(fn, args, kwargs, reps):
    best = 1e18
    for _ in range(reps):
        t = time.perf_counter()
        fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    print(f"{'n':>7} {'k':>3} {'n_cand':>7} {'old_ms':>10} {'new_ms':>10} {'speedup':>8}  identical")
    for n in (5_000, 30_000, 150_000):
        for k in (3, 4):
            data, nbins, cls_y, fq_y, kway_results, candidate_pool, dtype = _make_case(n, n_cols=30, n_bins=5, k=k, seed=7)
            kwargs = dict(
                factors_data=data, nbins=nbins, classes_y=cls_y, freqs_y=fq_y,
                max_combined_nbins=100_000, n_passes=2, dtype=dtype, verbose=0,
            )
            # identity + warm
            old_res = _refine_old(data, kway_results, candidate_pool, nbins, cls_y, fq_y, 100_000, 2, dtype)
            new_res = _refine_kway_coordinate_ascent(kway_results=kway_results, candidate_pool=candidate_pool, **kwargs)
            for (ot, oc, on, omi), (nt, nc, nn, nmi) in zip(old_res, new_res):
                assert ot == nt and np.array_equal(oc, nc) and on == nn and omi == nmi  # nosec B101 - internal invariant check in src/mlframe/feature_selection/filters/_benchmarks, not reachable with untrusted input
            ident = True

            reps = 10 if n <= 30_000 else 4
            old_t = _best_of(_refine_old, (data, kway_results, candidate_pool, nbins, cls_y, fq_y, 100_000, 2, dtype), {}, reps)
            new_t = _best_of(_refine_kway_coordinate_ascent, (), dict(kway_results=kway_results, candidate_pool=candidate_pool, **kwargs), reps)
            print(f"{n:>7} {k:>3} {len(candidate_pool):>7} {old_t*1e3:>10.2f} {new_t*1e3:>10.2f} {old_t/new_t:>7.2f}x  {ident}")


if __name__ == "__main__":
    main()
