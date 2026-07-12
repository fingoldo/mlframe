"""Bench: the ADOPTED fix for ``_build_kway_chained_lookup``'s per-step
``merge_vars``-over-the-full-growing-prefix cost -- an njit kernel
(``_dense_renumber_codes``) that dense-renumbers the ALREADY-COMBINED
``pre_prune_codes`` directly, instead of re-deriving them via ``merge_vars``
over every raw column in the prefix.

WHY
---
See ``bench_kway_chained_lookup_unique.py`` for the REJECTED first attempt
(``np.unique(pre_prune_codes, return_inverse=True)``): bit-identical but
measured ~3-4x SLOWER than ``merge_vars`` itself, because a generic
sort-based dedup loses to njit-compiled machine code at these small
categorical-join cardinalities.

This kernel instead reimplements merge_vars's OWN prune-then-renumber
algorithm (bincount via a manual loop, then an ascending-oldclass lookup
table, then a remap pass) as a SEPARATE njit function operating on the single
pre-combined ``pre_prune_codes`` array -- same generated-code style as
``merge_vars``'s inner loop, so it competes on equal footing, while still
skipping the (k-2) already-processed prefix columns that ``merge_vars`` would
otherwise re-walk from scratch every step.

Bit-identical (verified): 60 randomized cases matching
``bench_kway_chained_lookup_unique.py``'s coverage, zero mismatches.

Run:
  CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection.filters._benchmarks.bench_kway_chained_lookup_njit_renumber
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters.info_theory import merge_vars
from mlframe.feature_selection.filters._cat_kway_materialize import _dense_renumber_codes


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
    """NEW: njit direct dense-renumber of pre_prune_codes per step."""
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
        expected_size = running_nuniq * int(nbins[nxt_idx])
        cls_next, n_uniq_next = _dense_renumber_codes(pre_prune_codes, expected_size)
        results.append((cls_next, n_uniq_next))
        running_classes, running_nuniq = cls_next, n_uniq_next
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
            # warm numba
            _chain_new(data, idx_tuple, nbins, dtype)

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
