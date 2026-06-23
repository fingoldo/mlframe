"""Bench: collapse the DOUBLE merge_vars per swap-candidate in
``_refine_kway_coordinate_ascent`` into a SINGLE call.

WHY
---
The coordinate-ascent inner loop (one iteration per (position x candidate)
swap) evaluates each candidate k-way tuple by calling ``merge_vars`` on the
SAME ``new_tuple_sorted`` TWICE:

  new_classes, _, new_nuniq = merge_vars(...)          # freqs discarded
  ... new_mi = ... if False else None                  # dead line
  _, new_freqs, _ = merge_vars(...)                     # classes discarded
  new_mi = compute_mi_from_classes(new_classes, new_freqs, ...)

Both calls use byte-identical args, so they return identical
(classes, freqs, nuniq). The second is pure wasted work -- the FIRST call
already produced freqs but threw them away. Collapsing to one merge_vars +
one compute_mi_from_classes halves the dominant per-candidate cost.

Bit-identical BY CONSTRUCTION: same merge_vars output reused; no numeric change.

Run: CUDA_VISIBLE_DEVICES="" python bench_kway_coord_ascent_single_merge.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes, merge_vars


def _make(n, n_cols, n_bins, seed):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, n_bins, size=(n, n_cols)).astype(np.int32)
    nbins = np.full(n_cols, n_bins, dtype=np.int64)
    cls_y, fq_y, _ = merge_vars(
        factors_data=data, vars_indices=np.array([n_cols - 1], dtype=np.int64),
        var_is_nominal=None, factors_nbins=nbins, dtype=np.int32,
    )
    return data, nbins, cls_y, fq_y


def _eval_old(data, nbins, classes_y, freqs_y, tup, dtype):
    new_tuple_sorted = tuple(sorted(tup))
    new_classes, _, new_nuniq = merge_vars(
        factors_data=data, vars_indices=np.array(new_tuple_sorted, dtype=np.int64),
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    new_mi = compute_mi_from_classes(
        classes_x=new_classes, freqs_x=None,
        classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
    ) if False else None
    _, new_freqs, _ = merge_vars(
        factors_data=data, vars_indices=np.array(new_tuple_sorted, dtype=np.int64),
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    new_mi = compute_mi_from_classes(
        classes_x=new_classes, freqs_x=new_freqs,
        classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
    )
    return new_mi, new_classes, new_nuniq


def _eval_new(data, nbins, classes_y, freqs_y, tup, dtype):
    new_tuple_sorted = tuple(sorted(tup))
    new_classes, new_freqs, new_nuniq = merge_vars(
        factors_data=data, vars_indices=np.array(new_tuple_sorted, dtype=np.int64),
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    new_mi = compute_mi_from_classes(
        classes_x=new_classes, freqs_x=new_freqs,
        classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
    )
    return new_mi, new_classes, new_nuniq


def main():
    dtype = np.int32
    for n in (10_000, 50_000, 200_000):
        data, nbins, cls_y, fq_y = _make(n, 8, 6, seed=7)
        tuples = [(0, 1, 2), (1, 3, 4), (0, 2, 5), (2, 4, 6), (1, 5, 6)]
        # warm
        _eval_old(data, nbins, cls_y, fq_y, tuples[0], dtype)
        _eval_new(data, nbins, cls_y, fq_y, tuples[0], dtype)
        # identity
        max_diff = 0.0
        for t in tuples:
            mo, co, no = _eval_old(data, nbins, cls_y, fq_y, t, dtype)
            mn, cn, nn = _eval_new(data, nbins, cls_y, fq_y, t, dtype)
            max_diff = max(max_diff, abs(mo - mn))
            assert np.array_equal(co, cn) and no == nn, t

        N = 60
        best_old = best_new = 1e9
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(N):
                for t in tuples:
                    _eval_old(data, nbins, cls_y, fq_y, t, dtype)
            best_old = min(best_old, time.perf_counter() - t0)
            t0 = time.perf_counter()
            for _ in range(N):
                for t in tuples:
                    _eval_new(data, nbins, cls_y, fq_y, t, dtype)
            best_new = min(best_new, time.perf_counter() - t0)
        speedup = best_old / best_new
        print(f"n={n:>7}: OLD={best_old*1e3:8.2f}ms  NEW={best_new*1e3:8.2f}ms  "
              f"speedup={speedup:.3f}x  max|dMI|={max_diff:.2e}")


if __name__ == "__main__":
    main()
