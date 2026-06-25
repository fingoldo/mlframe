"""Lazy-compiled numba kernels carved out of preprocessing/cleaning.py (single-sibling split).

Each _get_*_njit() factory lazily compiles + memoises one njit kernel (deferred so module import stays
light and does not force numba native init at an ABI-sensitive point in the py3.14 import order). cleaning.py
re-exports the factories at its top so the call sites (_get_count_distinct_njit() etc.) are unchanged.
"""

from __future__ import annotations

import numpy as np


_COUNT_DISTINCT_NJIT = None


def _get_count_distinct_njit():
    """Lazily compile the njit count-distinct kernel. Deferred so module import stays light and does not force numba's native init at the wrong point in the ABI-sensitive import order on py3.14."""
    global _COUNT_DISTINCT_NJIT
    if _COUNT_DISTINCT_NJIT is None:
        import numba

        @numba.njit(cache=True)
        def _count_distinct_sorted_float(sorted_vals, skip0, skip1):
            # sorted ascending; NaNs sort to the end for float arrays. Counts distinct finite values excluding skip0/skip1 (NaN-coded skip = "no skip").
            n = sorted_vals.shape[0]
            count = 0
            have_prev = False
            prev = 0.0
            for i in range(n):
                v = sorted_vals[i]
                if v != v:  # NaN
                    continue
                if v == skip0 or (skip1 == skip1 and v == skip1):
                    continue
                if (not have_prev) or v != prev:
                    count += 1
                    prev = v
                    have_prev = True
            return count

        _COUNT_DISTINCT_NJIT = _count_distinct_sorted_float
    return _COUNT_DISTINCT_NJIT


_COUNT_DISTINCT_ROUNDED_NJIT = None


def _get_count_distinct_rounded_njit():
    """Lazily compile the njit kernel that counts distinct of ``round(sorted_vals, d)`` for a single precision ``d`` in one O(n) pass.

    ``np.round`` is monotone non-decreasing, so ``round(sort(x), d)`` equals ``sort(round(x, d))`` elementwise; counting distinct over the already-sorted
    array (rounding each element inline) is therefore bit-identical to sorting a freshly-rounded copy. This lets ``is_variable_truly_continuous`` sort the
    fractional part ONCE and probe every rounding precision over that single sort, instead of re-sorting (and re-allocating a rounded copy) per precision.
    """
    global _COUNT_DISTINCT_ROUNDED_NJIT
    if _COUNT_DISTINCT_ROUNDED_NJIT is None:
        import numba

        @numba.njit(cache=True)
        def _count_distinct_rounded_sorted(sorted_vals, ndigits, skip0, skip1):
            scale = 10.0**ndigits
            n = sorted_vals.shape[0]
            count = 0
            have_prev = False
            prev = 0.0
            for i in range(n):
                v = sorted_vals[i]
                if v != v:  # NaN
                    continue
                # numpy round-half-to-even (banker's rounding), matching np.round semantics.
                scaled = v * scale
                r = np.rint(scaled) / scale
                if r == skip0 or (skip1 == skip1 and r == skip1):
                    continue
                if (not have_prev) or r != prev:
                    count += 1
                    prev = r
                    have_prev = True
            return count

        _COUNT_DISTINCT_ROUNDED_NJIT = _count_distinct_rounded_sorted
    return _COUNT_DISTINCT_ROUNDED_NJIT


_OUTLIER_MASK_NJIT = None


def _get_outlier_mask_njit():
    """Lazily compile the fused outlier-mask kernel. One parallel pass over the values computes the keep-mask and the two outside-fence counts together,
    replacing four separate full-array passes (``v < l``, ``v > r``, two ``.sum()``, ``(~il) & (~ir)``). The per-element comparison is exact and the
    counts are integer increments, so the result is bit-identical to the numpy expression regardless of thread scheduling; NaN compares False on both sides
    in numpy and numba alike, so NaN rows stay in the keep-mask identically (measured 2.7-3.6x on the masking segment at n=10M, 2026-06-14)."""
    global _OUTLIER_MASK_NJIT
    if _OUTLIER_MASK_NJIT is None:
        import numba

        @numba.njit(cache=True, parallel=True)
        def _outlier_mask(values, l, r):
            n = values.shape[0]
            idx = np.empty(n, dtype=np.bool_)
            n_less_l = 0
            n_more_r = 0
            for i in numba.prange(n):
                x = values[i]
                below = x < l
                above = x > r
                idx[i] = (not below) and (not above)
                if below:
                    n_less_l += 1
                if above:
                    n_more_r += 1
            return idx, n_less_l, n_more_r

        _OUTLIER_MASK_NJIT = _outlier_mask
    return _OUTLIER_MASK_NJIT


_SPAN_FENCE_NJIT = None


def _get_span_fence_njit():
    """Lazily compile the fused span-mask + fence-count kernel. One parallel pass over the values builds the in-span keep-mask
    (q0 <= v <= q1) and the two outside-fence counts (v < lo, v > hi) together, replacing the separate ``(values>=q0)&(values<=q1)``
    mask plus two ``(values<lo).sum()`` / ``(values>hi).sum()`` full-array passes. NaN compares False on every comparison in numpy
    and numba alike, so NaN rows are excluded from the span mask AND counted in neither fence — bit-identical to the numpy expression."""
    global _SPAN_FENCE_NJIT
    if _SPAN_FENCE_NJIT is None:
        import numba

        @numba.njit(cache=True, parallel=True)
        def _span_fence(values, q0, q1, lo, hi):
            n = values.shape[0]
            mask = np.empty(n, dtype=np.bool_)
            n_below = 0
            n_above = 0
            for i in numba.prange(n):
                x = values[i]
                mask[i] = (x >= q0) and (x <= q1)
                if x < lo:
                    n_below += 1
                if x > hi:
                    n_above += 1
            return mask, n_below, n_above

        _SPAN_FENCE_NJIT = _span_fence
    return _SPAN_FENCE_NJIT
