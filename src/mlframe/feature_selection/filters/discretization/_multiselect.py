"""In-place multi-order-statistic selection for the quantile-edge kernels.

``np.partition(col, kths)`` in numba runs one full quickselect per requested kth over the whole array, so a
10-bin quantile edge set (~21 kths) costs ~21 O(n) passes per column. ``multiselect_inplace`` places every kth
in ONE recursive pass: select the middle kth, then recurse into the left / right sub-ranges with only the kths
that fall there (O(n log k)). The values at the kth positions are the exact order statistics, so any consumer
reading only ``col[k]`` for ``k in kths`` is bit-identical to ``np.partition`` / ``np.sort``.
"""
from __future__ import annotations

import numpy as np
from numba import njit

_SMALL = 24


@njit(nogil=True, cache=True, inline="always")
def _insertion_sort_range(a: np.ndarray, lo: int, hi: int) -> None:
    """Sort ``a[lo:hi+1]`` in place (cheaper than partitioning for tiny ranges)."""
    for i in range(lo + 1, hi + 1):
        x = a[i]
        j = i - 1
        while j >= lo and a[j] > x:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = x


@njit(nogil=True, cache=True)
def multiselect_inplace(a: np.ndarray, kths: np.ndarray) -> None:
    """Rearrange NaN-free ``a`` in place so ``a[k]`` equals ``np.sort(a)[k]`` for every ``k`` in ``kths``.

    ``kths`` must be sorted ascending, unique, within ``[0, len(a))``. Quickselect with median-of-three pivots
    and a Hoare partition (tie-heavy columns split evenly); a per-kth iteration budget falls back to sorting the
    remaining sub-range, which bounds the worst case at O(n log n) like introselect."""
    n = a.shape[0]
    nk = kths.shape[0]
    if n < 2 or nk == 0:
        return
    # Each popped frame pushes at most two; live frames never exceed 2 * (nk + 2).
    stack = np.empty((2 * (nk + 2), 4), dtype=np.int64)
    stack[0, 0] = 0
    stack[0, 1] = n - 1
    stack[0, 2] = 0
    stack[0, 3] = nk
    sp = 1
    while sp > 0:
        sp -= 1
        lo = stack[sp, 0]
        hi = stack[sp, 1]
        klo = stack[sp, 2]
        khi = stack[sp, 3]
        if klo >= khi or lo >= hi:
            continue
        if hi - lo < _SMALL:
            _insertion_sort_range(a, lo, hi)
            continue
        kmid = (klo + khi) // 2
        k = kths[kmid]
        l = lo
        h = hi
        budget = 2 * int(np.log2(hi - lo + 1)) + 8
        while h > l:
            if h - l < _SMALL:
                _insertion_sort_range(a, l, h)
                break
            if budget <= 0:
                a[l:h + 1].sort()
                break
            budget -= 1
            m = (l + h) // 2
            if a[m] < a[l]:
                t = a[m]
                a[m] = a[l]
                a[l] = t
            if a[h] < a[l]:
                t = a[h]
                a[h] = a[l]
                a[l] = t
            if a[h] < a[m]:
                t = a[h]
                a[h] = a[m]
                a[m] = t
            p = a[m]
            i = l
            j = h
            while i <= j:
                while a[i] < p:
                    i += 1
                while a[j] > p:
                    j -= 1
                if i <= j:
                    t = a[i]
                    a[i] = a[j]
                    a[j] = t
                    i += 1
                    j -= 1
            # [l..j] <= p <= [i..h] with j < i; anything strictly between equals p and is already final.
            if k <= j:
                h = j
            elif k >= i:
                l = i
            else:
                break
        stack[sp, 0] = lo
        stack[sp, 1] = k - 1
        stack[sp, 2] = klo
        stack[sp, 3] = kmid
        sp += 1
        stack[sp, 0] = k + 1
        stack[sp, 1] = hi
        stack[sp, 2] = kmid + 1
        stack[sp, 3] = khi
        sp += 1
