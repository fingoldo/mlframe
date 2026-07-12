"""Bench: precompute the PARENT set's merge_vars state once per order-extension
step in ``_greedy_expand_one_seed``, instead of re-scanning ALL parent columns
+ candidate from raw data for every candidate in the sweep.

WHY
---
For a given expansion order, ``parent_set``/``parent_classes``/``parent_nclasses``
are fixed for the WHOLE ``for k in candidate_pool`` sweep -- only the single
BEST candidate gets accepted, and only AFTER the sweep finishes. The OLD code
still called ``merge_vars`` on the full re-sorted ``parent_set | {k}`` tuple
for every candidate, re-deriving the parent's contribution from raw columns
each time even though ``parent_classes``/``parent_nclasses`` were already
sitting in scope, loop-invariant.

Same order-sensitivity caveat as the coordinate-ascent sibling fix
(``bench_kway_coord_ascent_frozen_prefix.py``): ``merge_vars``'s dense
renumbering depends on merge ORDER, so the candidate must be spliced into its
correct SORTED position (via incremental prefix states) rather than simply
appended after the parent -- verified bit-identical end-to-end below (unlike
the coordinate-ascent case, ``parent_set`` never mutates mid-sweep here, so
no "stale cache after mid-sweep accept" hazard applies).

Run:
  CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection.filters._benchmarks.bench_greedy_expand_seed_frozen_prefix
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes, merge_vars
from mlframe.feature_selection.filters._cat_kway_materialize import _greedy_expand_one_seed


def _greedy_expand_old(factors_data, seed_indices, candidate_pool, nbins, classes_y, freqs_y, marginal_mi, max_combined_nbins, max_kway_order, min_inc_ii, dtype):
    """Pre-fix reference: full merge_vars over parent_set | {k} for every candidate."""
    parent_set = set(seed_indices)
    parent_vi = np.array(sorted(parent_set), dtype=np.int64)
    parent_classes, parent_freqs, parent_nclasses = merge_vars(
        factors_data=factors_data, vars_indices=parent_vi,
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    parent_mi = compute_mi_from_classes(parent_classes, parent_freqs, classes_y, freqs_y, dtype)

    for _order in range(len(parent_set) + 1, max_kway_order + 1):
        best_inc_ii = -np.inf
        best_var = -1
        best_classes = None
        best_nclasses = 0
        best_joint_mi = 0.0

        for k in candidate_pool:
            k_int = int(k)
            if k_int in parent_set:
                continue
            new_card_estimate = parent_nclasses * int(nbins[k_int])
            if new_card_estimate > max_combined_nbins or new_card_estimate >= 2**31:
                continue

            new_vi = np.array(sorted(parent_set | {k_int}), dtype=np.int64)
            new_classes, new_freqs, new_n = merge_vars(
                factors_data=factors_data, vars_indices=new_vi,
                var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
            )
            new_joint_mi = compute_mi_from_classes(new_classes, new_freqs, classes_y, freqs_y, dtype)
            inc_ii = new_joint_mi - parent_mi - float(marginal_mi[k_int])
            if inc_ii > best_inc_ii:
                best_inc_ii, best_var, best_classes, best_nclasses, best_joint_mi = inc_ii, k_int, new_classes, new_n, new_joint_mi

        if best_var < 0 or best_inc_ii < min_inc_ii:
            break
        parent_set.add(best_var)
        parent_classes, parent_nclasses, parent_mi = best_classes, best_nclasses, best_joint_mi

    if len(parent_set) <= 2:
        return None
    return tuple(sorted(parent_set)), parent_classes, int(parent_nclasses), float(parent_mi)


def _make_case(n, n_cols, n_bins, seed):
    rng = np.random.default_rng(seed)
    dtype = np.int32
    data = rng.integers(0, n_bins, size=(n, n_cols)).astype(np.int32)
    nbins = np.full(n_cols, n_bins, dtype=np.int64)
    y_idx = n_cols - 1
    cls_y, fq_y, _ = merge_vars(
        factors_data=data, vars_indices=np.array([y_idx], dtype=np.int64),
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    marginal_mi = np.zeros(n_cols, dtype=np.float64)
    for c in range(n_cols - 1):
        cls_c, fq_c, _ = merge_vars(
            factors_data=data, vars_indices=np.array([c], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        marginal_mi[c] = compute_mi_from_classes(cls_c, fq_c, cls_y, fq_y, dtype)
    seed_cols = rng.choice(n_cols - 1, size=2, replace=False)
    seed_indices = tuple(sorted(int(c) for c in seed_cols))
    candidate_pool = np.array([c for c in range(n_cols - 1) if c not in seed_indices], dtype=np.int64)
    return data, nbins, cls_y, fq_y, marginal_mi, seed_indices, candidate_pool, dtype


def _best_of(fn, kwargs, reps):
    best = 1e18
    for _ in range(reps):
        t = time.perf_counter()
        fn(**kwargs)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    print(f"{'n':>7} {'max_order':>9} {'n_cand':>7} {'old_ms':>10} {'new_ms':>10} {'speedup':>8}  identical")
    for n in (5_000, 30_000, 150_000):
        for max_kway_order in (4, 5):
            data, nbins, cls_y, fq_y, marginal_mi, seed_indices, candidate_pool, dtype = _make_case(n, n_cols=30, n_bins=5, seed=3)
            kwargs = dict(
                factors_data=data, seed_indices=seed_indices, candidate_pool=candidate_pool,
                nbins=nbins, classes_y=cls_y, freqs_y=fq_y, marginal_mi=marginal_mi,
                max_combined_nbins=100_000, max_kway_order=max_kway_order, min_inc_ii=-1.0, dtype=dtype,
            )
            old_res = _greedy_expand_old(**kwargs)
            new_res = _greedy_expand_one_seed(**kwargs)
            if old_res is None or new_res is None:
                ident = old_res is None and new_res is None
            else:
                ot, oc, on, omi = old_res
                nt, nc, nn, nmi = new_res
                ident = ot == nt and np.array_equal(oc, nc) and on == nn and omi == nmi
            assert ident  # nosec B101 - internal invariant check in src/mlframe/feature_selection/filters/_benchmarks, not reachable with untrusted input

            reps = 10 if n <= 30_000 else 4
            old_t = _best_of(_greedy_expand_old, kwargs, reps)
            new_t = _best_of(_greedy_expand_one_seed, kwargs, reps)
            print(f"{n:>7} {max_kway_order:>9} {len(candidate_pool):>7} {old_t*1e3:>10.2f} {new_t*1e3:>10.2f} {old_t/new_t:>7.2f}x  {ident}")


if __name__ == "__main__":
    main()
