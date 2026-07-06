
from __future__ import annotations

import numpy as np
from numba import njit, prange

################################################################################################
# ARRAY STATS
################################################################################################


# fastmath WITHOUT nnan/ninf: the full ``fastmath=True`` flag set tells LLVM to assume no NaN
# (the ``nnan`` flag), which would let it fold the ``v == v`` NaN test below to a constant True --
# silently defeating the NaN skip. We keep the SIMD-friendly arithmetic flags but drop the NaN/inf
# assumptions so the NaN-aware comparison survives. min/max is a comparison scan (no FP accumulator
# reduction), so dropping nnan/ninf does NOT block vectorisation the way it would for a sum-reduction.
_MINMAX_FASTMATH = {"nsz", "arcp", "contract", "afn", "reassoc"}

# Module-level empty-int32 mask singleton used as the default ``mask`` for the counting-argsort njit kernels.
# Building it once here (instead of ``np.array([], np.int32)`` in each signature) keeps the default a read-only,
# zero-length typed array with the same numba signature while avoiding a call in the argument default.
_EMPTY_INT32_MASK = np.array([], np.int32)


@njit(fastmath=_MINMAX_FASTMATH, cache=True)
def arrayMinMax(x, l=0, r=0):
    n = len(x)
    if r == 0:
        r = n
    # Empty-range guard: return NaN sentinels (numba-friendly: no Python exceptions)
    if n == 0 or r <= l:
        return (np.nan, np.nan)
    # NaN-aware seeding/comparison: a NaN seed (e.g. a leading-NaN column) would otherwise stick as
    # both min and max, and a non-leading NaN would be silently dropped -- both poison the uniform
    # discretiser's affine map. Seed from the first FINITE element and skip NaN entries; on all-finite
    # input this is bit-identical to the plain first-element seed + scan (no NaN ever takes the finite
    # branch, so the seed and every comparison match). +/-inf are finite-comparison-wise and stay normal.
    minimum = np.nan
    maximum = np.nan
    seeded = False
    for v in x[l:r]:
        if v == v:  # NaN != NaN; cheap, no np.isfinite dispatch
            if not seeded:
                minimum = v
                maximum = v
                seeded = True
            elif v > maximum:
                maximum = v
            elif v < minimum:
                minimum = v
    # All-NaN range: return NaN sentinels so the caller's _rng<=0 / NaN guard fires.
    return (minimum, maximum)


# WARNING (audit3): these two helpers use full ``fastmath=True`` (nnan/ninf), so they assume finite input and
# will NOT propagate NaN the way the serial ``arrayMinMax`` above deliberately does. They are currently UNUSED
# in src/mlframe; do NOT wire them into a NaN-sensitive hot path (e.g. the discretiser min/max scan) without a
# finite-gate at the wrapper level, or all-NaN / NaN-bearing columns will get a garbage range instead of the
# NaN sentinel the callers' ``_rng<=0 / NaN`` guards rely on.
@njit(fastmath=True, parallel=True, cache=True)
def arrayMinMaxParallel(array, l=0, r=0, maxThreads=8):
    arrLen = len(array)
    if r == 0:
        r = arrLen
    nElemsToProcess = r - l
    nThreads = min(max(nElemsToProcess, 1), maxThreads)
    chunkSize = nElemsToProcess // nThreads
    minMaxData = np.empty((nThreads, 2), array.dtype)
    for k in prange(nThreads):
        lBound = l + chunkSize * k
        rBound = l + chunkSize * (k + 1)
        if k == nThreads - 1:
            rBound = r
        minMaxData[k, :] = arrayMinMax(array, lBound, rBound)
    return np.min(minMaxData[:, 0]), np.max(minMaxData[:, 1])


@njit(fastmath=True, parallel=True, cache=True)
def npnbArrayMinMax(x):
    return x.min(), x.max()


################################################################################################
# ARRAY SORTING
################################################################################################
@njit(fastmath=True, cache=True)
def arrayCountingSort(array, maxval):
    res = np.empty(len(array), np.int32)
    m = maxval + 1
    count = np.zeros(m, np.int32)
    for a in array:
        count[a] += 1  # count occurences
    i = 0
    for a in range(m):  # emit
        for c in range(count[a]):  # - emit 'count[a]' copies of 'a'
            res[i] = a
            i += 1
    return res


################################################################################################
# ARRAY ARGSORTING
################################################################################################
@njit(fastmath=True, cache=True)
def emptyListOfInts():
    return [i for i in range(0)]


@njit(fastmath=True, cache=True)
def BinByUniqueValues(array, l, r, m, mask):
    groupedIndices = [emptyListOfInts() for k in range(m)]
    if len(mask) > 0:
        i = l
        while i < r:
            ind = mask[i]
            groupedIndices[array[ind]].append(ind)
            i += 1
    else:
        i = l
        while i < r:
            groupedIndices[array[i]].append(i)
            i += 1
    return groupedIndices
    # cGrowthFactor=2
    # if v>m:
    #    newM=m*cGrowthFactor
    #    #print ("resizing from %d to %d" %(m,newM))
    #    count+=[[i for i in range(0)] for k in range(newM-m)]
    #   m=newM


@njit(fastmath=True, cache=True)
def arrayCountingArgSort(array, maxval, mask=_EMPTY_INT32_MASK):
    m = maxval + 1

    # Allocate output array
    if len(mask) > 0:
        arrLen = len(mask)
    else:
        arrLen = len(array)
    argsorted = np.empty(arrLen, np.int32)

    # Group indices of same values
    groupedIndices = BinByUniqueValues(array, 0, arrLen, m, mask)

    position = 0
    for k in range(m):
        if len(groupedIndices[k]) > 0:
            for index in groupedIndices[k]:
                argsorted[position] = index
                position += 1
    return argsorted


@njit(fastmath=True, cache=True)
def arrayCountingArgSortAndUniqueValues(array, maxval, mask=_EMPTY_INT32_MASK):
    m = maxval + 1

    # Allocate output array
    if len(mask) > 0:
        arrLen = len(mask)
    else:
        arrLen = len(array)
    argsorted = np.empty(arrLen, np.int32)

    # Group indices of same values
    groupedIndices = BinByUniqueValues(array, 0, arrLen, m, mask)

    position = 0
    uniqueValues = emptyListOfInts()
    uniqueValuesIndices = emptyListOfInts()
    for k in range(m):
        if len(groupedIndices[k]) > 0:
            uniqueValues.append(k)
            uniqueValuesIndices.append(position)
            for index in groupedIndices[k]:
                argsorted[position] = index
                position += 1
    return np.array(uniqueValues, np.int32), np.array(uniqueValuesIndices, np.int32), argsorted


@njit(fastmath=True, parallel=True, cache=True)
def arrayCountingArgSortThreaded(array, maxval, mask=_EMPTY_INT32_MASK, maxThreads=2):
    m = maxval + 1

    # Allocate output array
    if len(mask) > 0:
        arrayLen = len(mask)
    else:
        arrayLen = len(array)
    argsorted = np.empty(arrayLen, np.int32)

    # Group indices of same values
    effectiveSize = int(m * 3)
    if arrayLen <= effectiveSize:
        nThreads = 1
    else:
        nThreads = min(max(arrayLen // effectiveSize, 1), maxThreads)
    groups = [[emptyListOfInts() for k in range(0)] for _ in range(nThreads)]
    chunkSize = arrayLen // nThreads
    for k in prange(nThreads):
        lBound = chunkSize * k
        rBound = chunkSize * (k + 1)
        if k == nThreads - 1:
            rBound = arrayLen
        groups[k] = BinByUniqueValues(array, lBound, rBound, m, mask)
    position = 0
    for k in range(m):
        for groupedIndices in groups:
            ls = groupedIndices[k]
            subLen = len(ls)
            if subLen > 0:
                for index in ls:
                    argsorted[position] = index
                    position += 1
    return argsorted


@njit(fastmath=True, parallel=True, cache=True)
def arrayCountingArgSortAndUniqueValuesThreaded(array, maxval, mask=_EMPTY_INT32_MASK, maxThreads=2):
    m = maxval + 1
    # Allocate output array
    if len(mask) > 0:
        arrayLen = len(mask)
    else:
        arrayLen = len(array)
    argsorted = np.empty(arrayLen, np.int32)

    # Group indices of same values
    effectiveSize = int(m * 3)
    if arrayLen <= effectiveSize:
        nThreads = 1
    else:
        nThreads = min(max(arrayLen // effectiveSize, 1), maxThreads)
    groups = [[emptyListOfInts() for k in range(0)] for _ in range(nThreads)]
    chunkSize = arrayLen // nThreads
    for k in prange(nThreads):
        lBound = chunkSize * k
        rBound = chunkSize * (k + 1)
        if k == nThreads - 1:
            rBound = arrayLen
        groups[k] = BinByUniqueValues(array, lBound, rBound, m, mask)

    position = 0
    uniqueValues, uniqueValuesIndices = [], []
    seen = np.zeros(m, dtype=np.bool_)  # O(1) membership test (replaces O(U) `k in uniqueValues`)
    for k in range(m):
        for groupedIndices in groups:
            if len(groupedIndices[k]) > 0:
                if not seen[k]:
                    seen[k] = True
                    uniqueValues.append(k)
                    uniqueValuesIndices.append(position)
                for index in groupedIndices[k]:
                    argsorted[position] = index
                    position += 1
    return np.array(uniqueValues, np.int32), np.array(uniqueValuesIndices, np.int32), argsorted


def topk_by_partition(arr: np.ndarray, k: int, axis: int | None = None, ascending: bool = False) -> tuple:
    """Returns indices and values of TOP-k elements of an array.

    Does NOT mutate the caller's array (previous implementation did `arr *= -1` in place).
    """
    # Copy rather than mutate caller's array; flip sign for descending.
    if not ascending:
        arr = -arr
    else:
        arr = np.asarray(arr).copy()

    # len() on multi-dim arrays gives len of first axis; use shape[axis] for per-axis cap.
    n_along_axis = arr.shape[axis] if axis is not None else arr.size
    k = min(k, n_along_axis)
    if k <= 0:
        # Empty selection; return empty arrays with matching shape.
        empty_ind = np.take(np.argsort(arr, axis=axis), np.arange(0), axis=axis)
        empty_val = np.take(arr if ascending else -arr, empty_ind, axis=axis)
        return empty_ind, empty_val

    # np.argpartition requires kth in [0, n-1]; clamp.
    part_kth = min(k - 1, n_along_axis - 1) if k == n_along_axis else k
    ind = np.argpartition(arr, min(part_kth, n_along_axis - 1), axis=axis)
    # Slice first k along axis.
    ind = np.take(ind, np.arange(k), axis=axis)
    vals_part = np.take_along_axis(arr, ind, axis=axis) if axis is not None else arr[ind]

    # sort within k elements
    ind_part = np.argsort(vals_part, axis=axis)
    ind = np.take_along_axis(ind, ind_part, axis=axis) if axis is not None else ind[ind_part]
    val = np.take_along_axis(vals_part, ind_part, axis=axis) if axis is not None else vals_part[ind_part]
    if not ascending:
        val = -val
    return ind, val
