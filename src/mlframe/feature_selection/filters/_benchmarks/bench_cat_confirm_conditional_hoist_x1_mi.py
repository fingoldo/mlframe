"""Bench: hoist the loop-invariant I(X1; Y) recompute out of the conditional-null
permutation loop in ``_cat_confirm_permutation._confirm_pairs_via_permutation``.

In the ``permutation_null="conditional"`` branch the inner per-permutation loop
shuffles X2 within strata of Y, then recomputes ALL THREE marginal/joint MIs
against the shuffled X2 -- INCLUDING ``i_x1_p = compute_mi_from_classes(cls_x1,
fq_x1, classes_y, freqs_y)``. But X1 (``cls_x1``/``fq_x1``) and Y
(``classes_y``/``freqs_y``) are NEVER mutated inside the loop -- only X2 is
shuffled. So ``i_x1_p`` is loop-invariant: identical float every permutation.

OLD: recompute ``i_x1_p`` every permutation (one full length-n MI pass / perm).
NEW: compute ``i_x1_p`` ONCE before the loop, reuse it.

Bit-identical BY CONSTRUCTION (same args -> same float). This bench isolates the
saving: it times only the repeated ``i_x1_p`` MI computation (OLD) vs a single
hoisted computation (NEW), at the realistic confirmation shapes
(n in {2k, 20k, 200k}, n_perms in {50, 500}).

Run:
  CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection.filters._benchmarks.bench_cat_confirm_conditional_hoist_x1_mi
"""
from __future__ import annotations

import time
import numpy as np

from mlframe.feature_selection.filters.info_theory._class_mi_kernels import (
    compute_mi_from_classes,
)


def _make_inputs(n, K_x1, K_y, seed=0):
    rng = np.random.default_rng(seed)
    cls_x1 = rng.integers(0, K_x1, size=n).astype(np.int32)
    classes_y = rng.integers(0, K_y, size=n).astype(np.int32)
    fq_x1 = np.bincount(cls_x1, minlength=K_x1).astype(np.float64) / n
    freqs_y = np.bincount(classes_y, minlength=K_y).astype(np.float64) / n
    return cls_x1, fq_x1, classes_y, freqs_y


def _old_loop(cls_x1, fq_x1, classes_y, freqs_y, n_perms, dtype):
    """Mirror the OLD conditional branch: recompute i_x1_p every permutation."""
    acc = 0.0
    for _ in range(n_perms):
        i_x1_p = compute_mi_from_classes(
            classes_x=cls_x1, freqs_x=fq_x1,
            classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
        )
        acc += i_x1_p
    return acc


def _new_loop(cls_x1, fq_x1, classes_y, freqs_y, n_perms, dtype):
    """NEW: hoist i_x1_p once (loop-invariant)."""
    i_x1_p = compute_mi_from_classes(
        classes_x=cls_x1, freqs_x=fq_x1,
        classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
    )
    acc = 0.0
    for _ in range(n_perms):
        acc += i_x1_p
    return acc


def _best_of(fn, args, reps):
    best = 1e18
    for _ in range(reps):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    dtype = np.int32
    print(f"{'n':>8} {'n_perms':>8} {'old_ms':>10} {'new_ms':>10} {'speedup':>8}  identical")
    for n in (2000, 20000, 200000):
        cls_x1, fq_x1, classes_y, freqs_y = _make_inputs(n, K_x1=8, K_y=3)
        args_base = (cls_x1, fq_x1, classes_y, freqs_y)
        # warm
        a = _old_loop(*args_base, 3, dtype)
        b = _new_loop(*args_base, 3, dtype)
        for n_perms in (50, 500):
            args = (*args_base, n_perms, dtype)
            ident = (_old_loop(*args) == _new_loop(*args))
            reps = 20 if n <= 20000 else 8
            old = _best_of(_old_loop, args, reps)
            new = _best_of(_new_loop, args, reps)
            print(f"{n:>8} {n_perms:>8} {old*1e3:>10.4f} {new*1e3:>10.4f} "
                  f"{old/new:>7.1f}x  {ident}")


if __name__ == "__main__":
    main()
