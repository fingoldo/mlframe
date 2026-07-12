"""Bench: replace the per-step ``merge_vars`` call over the FULL growing prefix
in ``_build_kway_chained_lookup`` with a single ``np.unique(pre_prune_codes,
return_inverse=True)`` call, since ``pre_prune_codes`` already carries
everything the full-prefix merge would derive from raw columns.

WHY
---
At each chain step, ``pre_prune_codes = running_classes + nxt_vals *
running_nuniq`` is ALREADY the pre-prune joint code for the growing prefix
(``running_classes``/``running_nuniq`` came from the PRIOR step's own
prune-renumber). The OLD code then re-derived the SAME post-prune dense
classes by calling ``merge_vars`` over ``idx_tuple[:step+1]`` -- re-scanning
every raw column in the prefix from scratch, even though ``merge_vars``'s own
prune-then-renumber (drop empty bins, relabel survivors 0..n_uniq-1 in
ascending old-code order) is EXACTLY what ``np.unique``'s inverse-index
produces when applied straight to ``pre_prune_codes``.

Bit-identical (verified): 60 randomized cases (varying n, n_cols, cardinality,
k) plus adversarial cases (a constant column, a heavily skewed column, and an
all-identical-rows degenerate case) all produced EXACT matches between
``merge_vars``'s per-row classes/nuniq and the ``np.unique``-based
replacement at every chain step, zero mismatches.

REJECTED (bit-identical but SLOWER): this bench's own numbers show the
``np.unique``-based replacement running ~3-4x SLOWER than the current
``merge_vars``-per-step code across n in {5k, 30k, 200k} -- ``np.unique``'s
generic sort-based dedup loses badly to ``merge_vars``'s njit-compiled direct
bincount+remap at these small-cardinality categorical joins, even though it
touches fewer columns per step. The fix actually shipped is
``_dense_renumber_codes`` (an njit kernel applying the SAME prune-then-renumber
scheme merge_vars uses, directly to ``pre_prune_codes`` instead of re-deriving
it from raw columns) -- see ``bench_kway_chained_lookup_njit_renumber.py``,
which IS a net win. Kept runnable here as the documented rejected attempt.

Run:
  CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection.filters._benchmarks.bench_kway_chained_lookup_unique
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters.info_theory import merge_vars


def _chain_old(factors_data, idx_tuple, nbins, dtype):
    """Pre-fix reference: merge_vars over the full growing prefix at every step."""
    results = []
    for step in range(1, len(idx_tuple)):
        vi_prefix = np.array(idx_tuple[: step + 1], dtype=np.int64)
        cls_next, _, n_uniq_next = merge_vars(
            factors_data=factors_data, vars_indices=vi_prefix,
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        results.append((cls_next.astype(np.int64, copy=False), int(n_uniq_next)))
    return results


def _chain_new(factors_data, idx_tuple, nbins, dtype):
    """NEW: incremental np.unique(pre_prune_codes, return_inverse=True) per step."""
    results = []
    vi_2 = np.array([idx_tuple[0], idx_tuple[1]], dtype=np.int64)
    running_classes, _, running_nuniq = merge_vars(
        factors_data=factors_data, vars_indices=vi_2,
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    running_classes = running_classes.astype(np.int64, copy=False)
    running_nuniq = int(running_nuniq)
    results.append((running_classes, running_nuniq))
    for step in range(2, len(idx_tuple)):
        nxt_idx = int(idx_tuple[step])
        nxt_vals = factors_data[:, nxt_idx].astype(np.int64, copy=False)
        pre_prune_codes = running_classes + nxt_vals * running_nuniq
        uniq_vals, inv = np.unique(pre_prune_codes, return_inverse=True)
        running_classes = np.asarray(inv, dtype=np.int64).reshape(-1)
        running_nuniq = int(uniq_vals.size)
        results.append((running_classes, running_nuniq))
    return results


def _best_of(fn, args, reps):
    best = 1e18
    for _ in range(reps):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    dtype = np.int32
    print(f"{'n':>8} {'k':>3} {'n_bins':>7} {'old_ms':>10} {'new_ms':>10} {'speedup':>8}  identical")
    for n in (5_000, 30_000, 200_000):
        for k in (3, 5):
            rng = np.random.default_rng(11)
            n_cols = max(k, 8)
            n_bins = 5
            data = rng.integers(0, n_bins, size=(n, n_cols)).astype(np.int32)
            nbins = np.full(n_cols, n_bins, dtype=np.int64)
            idx_tuple = tuple(range(k))

            old_res = _chain_old(data, idx_tuple, nbins, dtype)
            new_res = _chain_new(data, idx_tuple, nbins, dtype)
            ident = all(
                np.array_equal(oc, nc) and on == nn
                for (oc, on), (nc, nn) in zip(old_res, new_res)
            )
            assert ident  # nosec B101 - internal invariant check in src/mlframe/feature_selection/filters/_benchmarks, not reachable with untrusted input

            reps = 20 if n <= 30_000 else 6
            old_t = _best_of(_chain_old, (data, idx_tuple, nbins, dtype), reps)
            new_t = _best_of(_chain_new, (data, idx_tuple, nbins, dtype), reps)
            print(f"{n:>8} {k:>3} {n_bins:>7} {old_t*1e3:>10.3f} {new_t*1e3:>10.3f} {old_t/new_t:>7.2f}x  {ident}")


if __name__ == "__main__":
    main()
