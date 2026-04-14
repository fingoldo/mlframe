import numba
import numpy as np
import pandas as pd
from numba import cuda, njit, prange

################################################################################################
# ARRAY STATS
################################################################################################


@njit(fastmath=True)
def arrayMinMax(x, l=0, r=0):
    n = len(x)
    if r == 0:
        r = n
    # Empty-range guard: return NaN sentinels (numba-friendly: no Python exceptions)
    if n == 0 or r <= l:
        return (np.nan, np.nan)
    firstElem = x[l]
    maximum, minimum = firstElem, firstElem
    for v in x[l:r]:
        if v > maximum:
            maximum = v
        elif v < minimum:
            minimum = v
    return (minimum, maximum)


@njit(fastmath=True, parallel=True)
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


@njit(fastmath=True, parallel=True)
def npnbArrayMinMax(x):
    return x.min(), x.max()


################################################################################################
# ARRAY SORTING
################################################################################################
@njit(fastmath=True)
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
@njit(fastmath=True)
def emptyListOfInts():
    return [i for i in range(0)]


@njit(fastmath=True)
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
    # print("l=",l,"r=",r) #,groupedIndices,'\n'
    return groupedIndices
    # cGrowthFactor=2
    # if v>m:
    #    newM=m*cGrowthFactor
    #    #print ("resizing from %d to %d" %(m,newM))
    #    count+=[[i for i in range(0)] for k in range(newM-m)]
    #   m=newM


@njit(fastmath=True)
def arrayCountingArgSort(array, maxval, mask=np.array([], np.int32)):
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


@njit(fastmath=True)
def arrayCountingArgSortAndUniqueValues(array, maxval, mask=np.array([], np.int32)):
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


@njit(fastmath=True, parallel=True)
def arrayCountingArgSortThreaded(array, maxval, mask=np.array([], np.int32), maxThreads=2):
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
    # print("nThreads=",nThreads)
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


@njit(fastmath=True, parallel=True)
def arrayCountingArgSortAndUniqueValuesThreaded(array, maxval, mask=np.array([], np.int32), maxThreads=2):
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
    # print("nThreads=",nThreads)
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


def topk_by_partition(input: np.ndarray, k: int, axis: int = None, ascending: bool = False) -> tuple:
    """Returns indices and values of TOP-k elements of an array.

    Does NOT mutate the caller's array (previous implementation did `input *= -1` in place).
    """
    # Copy rather than mutate caller's array; flip sign for descending.
    if not ascending:
        input = -input
    else:
        input = np.asarray(input).copy()

    # len() on multi-dim arrays gives len of first axis; use shape[axis] for per-axis cap.
    n_along_axis = input.shape[axis] if axis is not None else input.size
    k = min(k, n_along_axis)
    if k <= 0:
        # Empty selection; return empty arrays with matching shape.
        empty_ind = np.take(np.argsort(input, axis=axis), np.arange(0), axis=axis)
        empty_val = np.take(input if ascending else -input, empty_ind, axis=axis)
        return empty_ind, empty_val

    # np.argpartition requires kth in [0, n-1]; clamp.
    part_kth = min(k - 1, n_along_axis - 1) if k == n_along_axis else k
    ind = np.argpartition(input, min(part_kth, n_along_axis - 1), axis=axis)
    # Slice first k along axis.
    ind = np.take(ind, np.arange(k), axis=axis)
    vals_part = np.take_along_axis(input, ind, axis=axis) if axis is not None else input[ind]

    # sort within k elements
    ind_part = np.argsort(vals_part, axis=axis)
    ind = np.take_along_axis(ind, ind_part, axis=axis) if axis is not None else ind[ind_part]
    val = np.take_along_axis(vals_part, ind_part, axis=axis) if axis is not None else vals_part[ind_part]
    if not ascending:
        val = -val
    return ind, val
