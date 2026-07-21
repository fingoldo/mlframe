"""Count / crossing / quantile aggregate kernels carved out of numerical.py (single-sibling split).

Holds the nunique / modes / quantiles fused kernel, the n-crossings serial + mark-parallel kernels, and the
numba nunique/mode/quantiles variant. numerical.py re-exports these at its bottom so the public import
surface (compute_nunique_modes_quantiles_numpy / compute_ncrossings / compute_nunique_mode_quantiles_numba)
is unchanged.
"""

from __future__ import annotations

from typing import Sequence, cast

import numba
import numpy as np

from ._numerical_constants import NUMBA_NJIT_PARAMS, default_quantiles


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fused_nunique_modes_quantiles_kernel(s: np.ndarray, q: np.ndarray, max_modes: int):  # pragma: no cover
    """Single-pass njit core: from the SORTED array ``s`` derive nunique, mode stats and the
    median_unbiased quantiles in one compiled traversal, replacing the per-row stack of numpy
    temporaries (``boundary``/``nonzero``/``diff``/``append``/``clip``/``floor``/``astype``).

    Returns ``(nuniques, modes_min, modes_max, modes_mean, modes_qty, quantiles)``. Mode selection
    is bit-identical to the prior ``np.lexsort((vals, -counts))`` path: the global-max count defines
    the modes, ties broken by ascending value, capped at ``max_modes`` (``s`` ascending => the lowest
    values are kept first, matching the lexsort value-tiebreak). Quantile interp uses the same
    Hyndman-Fan type-8 virtual index as the numpy code, so the result matches to <=1 ULP.
    """
    n = s.size

    # First pass: nunique + global max run-length (the mode count).
    nuniques = 1
    max_count = 1
    cur_count = 1
    for i in range(1, n):
        if s[i] != s[i - 1]:
            nuniques += 1
            if cur_count > max_count:
                max_count = cur_count
            cur_count = 1
        else:
            cur_count += 1
    if cur_count > max_count:
        max_count = cur_count

    if max_count == 1:
        modes_min = np.nan
        modes_max = np.nan
        modes_mean = np.nan
        modes_qty = np.nan
    else:
        # Second pass: collect up to max_modes lowest-valued uniques whose run-length == max_count.
        modes_min = np.inf
        modes_max = -np.inf
        modes_sum = 0.0
        modes_qty_i = 0
        cur_count = 1
        i = 1
        while i <= n:
            at_end = i == n
            if at_end or s[i] != s[i - 1]:
                if cur_count == max_count and modes_qty_i < max_modes:
                    v = s[i - 1]
                    if v < modes_min:
                        modes_min = v
                    if v > modes_max:
                        modes_max = v
                    modes_sum += v
                    modes_qty_i += 1
                cur_count = 1
            else:
                cur_count += 1
            i += 1
        modes_mean = modes_sum / modes_qty_i
        modes_qty = float(modes_qty_i)

    # median_unbiased (Hyndman-Fan type 8): virtual index h = (n + 1/3)*p + 1/3, clamp to [1, n], linear interp.
    quantiles = np.empty(q.size, dtype=np.float64)
    nf = float(n)
    for k in range(q.size):
        h = (nf + 1.0 / 3.0) * q[k] + 1.0 / 3.0
        if h < 1.0:
            h = 1.0
        elif h > nf:
            h = nf
        fl = np.floor(h)
        lo = int(fl) - 1
        hi = lo + 1
        if hi > n - 1:
            hi = n - 1
        g = h - fl
        quantiles[k] = s[lo] * (1.0 - g) + s[hi] * g

    return float(nuniques), modes_min, modes_max, modes_mean, modes_qty, quantiles


def _fused_nunique_modes_quantiles(arr: np.ndarray, q: np.ndarray, quantile_method: str, max_modes: int) -> tuple:
    """All-finite fast path for ``compute_nunique_modes_quantiles_numpy``: one ``np.sort`` feeds the sorted-unique
    values + counts AND the ``median_unbiased`` quantiles, replacing the independent ``np.unique`` sort and
    ``np.nanquantile`` partition. Caller guarantees ``arr`` is 1-D, all-finite, ``len>=2``, method=median_unbiased.

    The boundary-detection + counts + mode-pick + quantile-interp are fused into a single njit pass
    (``_fused_nunique_modes_quantiles_kernel``) to drop the per-row numpy-temporary churn.
    """
    s = np.sort(arr)
    nuniques, modes_min, modes_max, modes_mean, modes_qty, quantiles = _fused_nunique_modes_quantiles_kernel(s, q, max_modes)
    res = (nuniques, modes_min, modes_max, modes_mean, modes_qty)
    res = res + tuple(quantiles)
    res = res + tuple(compute_ncrossings(arr=arr, marks=quantiles))
    return res


def compute_nunique_modes_quantiles_numpy(
    arr: np.ndarray, q: Sequence[float] = default_quantiles, quantile_method: str = "median_unbiased", max_modes: int = 10, return_unsorted_stats: bool = True
) -> tuple:
    """For a 1d array, computes aggregates:
    nunique
    modes:min,max,mean
    list of quantiles (0 and 1 included by default, therefore, min/max)
    number of quantiles crossings
    Can NOT be numba jitted (yet).
    """
    # Fused single-sort fast path: np.unique (its own sort) + np.nanquantile (its own partition) both walk the array
    # independently. When arr is all-finite (the common numeric-column case) a single np.sort yields the sorted-unique
    # values + counts AND the quantiles, eliminating one full sort and one full partition. Gated on no-NaN because
    # np.unique collapses all NaN into a single entry while a sort keeps each NaN distinct -> the fast path would change
    # nunique/modes. Quantiles carry a ~1e-16 ULP delta vs np.nanquantile (FP order in the linear-interp), far below any
    # selection-altering threshold; nunique/modes/ncrossings are exactly identical.
    _fused_ok = return_unsorted_stats and arr.ndim == 1 and quantile_method == "median_unbiased" and arr.size >= 2 and not np.isnan(arr).any()
    if _fused_ok:
        return _fused_nunique_modes_quantiles(arr, np.asarray(q, dtype=np.float64), quantile_method, max_modes)

    if return_unsorted_stats:
        vals, counts = np.unique(arr, return_counts=True)

        max_modes = min(max_modes, len(counts))

        # lexsort with value as tiebreaker so tied counts give a deterministic mode pick across runs.
        modes_indices = np.lexsort((vals, -counts))[:max_modes]

        first_mode_count = counts[modes_indices[0]]

        if first_mode_count == 1:
            modes_min, modes_max, modes_mean, modes_qty = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )  # for higher stability. cnt=1 is not really a mode, rather a random pick.
        else:
            next_idx = modes_indices[0]
            best_modes_list = [vals[next_idx]]
            for i in range(1, max_modes):
                next_idx = modes_indices[i]
                next_mode_count = counts[next_idx]
                if next_mode_count < first_mode_count:
                    break
                else:
                    best_modes_list.append(vals[next_idx])
            best_modes = np.asarray(best_modes_list)
            modes_min = best_modes.min()
            modes_max = best_modes.max()
            modes_mean = best_modes.mean()
            modes_qty = len(best_modes)

        nuniques = len(vals)

        res: tuple = (nuniques, modes_min, modes_max, modes_mean, modes_qty)
    else:
        res = ()

    # nanquantile so a NaN in arr doesn't poison all returned quantile feature values silently.
    # The public feature aggregator runs on caller-supplied 1-D series; NaN is common in raw data.
    quantiles = np.nanquantile(arr, q, method=quantile_method)  # type: ignore[call-overload]  # quantile_method is a plain str param (any of numpy's valid method literals); over-typing it as a Literal isn't worth the API churn
    res = res + tuple(quantiles)  # .tolist()

    if return_unsorted_stats:
        res = res + tuple(compute_ncrossings(arr=arr, marks=quantiles))  # .tolist()

    return res


@numba.njit(**NUMBA_NJIT_PARAMS)
def _compute_ncrossings_serial(arr: np.ndarray, marks: np.ndarray, dtype=np.int32) -> np.ndarray:  # pragma: no cover
    """Serial reference: scan the array once, updating the prev-difference of every mark per element.

    Element-major traversal re-reads/writes the length-M ``prev_ds`` array for every sample (strided, non-sequential
    over arr per mark). Kept as the bit-exact reference and the fallback for non-int32 output dtype.
    """
    n_crossings = np.zeros(len(marks), dtype=dtype)
    prev_ds = np.full(len(marks), dtype=np.float32, fill_value=np.nan)

    for next_value in arr:
        for i, mark in enumerate(marks):
            d = next_value - mark
            if not np.isnan(prev_ds[i]):
                if d * prev_ds[i] < 0:
                    n_crossings[i] += 1
            prev_ds[i] = d

    return n_crossings


@numba.njit(parallel=True, fastmath=False, cache=True, nogil=True)
def _compute_ncrossings_marks_prange(arr: np.ndarray, marks: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Mark-major parallel variant: each ``prange`` lane owns one mark and walks ``arr`` in a single sequential pass,
    keeping its previous difference in a register. Bit-identical to the serial path -- the per-element difference is
    truncated to float32 (matching the original ``prev_ds`` float32 storage) and the crossing test stays ``< 0`` in
    float32 -- but with perfect cache locality and no shared length-M state. ~31x at n=1e6, marks=7.
    """
    m = len(marks)
    n = arr.shape[0]
    out = np.zeros(m, dtype=np.int32)
    for i in numba.prange(m):
        mark = marks[i]
        prev = np.float32(np.nan)
        c = 0
        for j in range(n):
            d = np.float32(arr[j] - mark)
            if not np.isnan(prev):
                if d * prev < np.float32(0.0):
                    c += 1
            prev = d
        out[i] = c
    return out


def compute_ncrossings(arr: np.ndarray, marks: np.ndarray, dtype: type = np.int32) -> np.ndarray:
    """Count sign-changes in ``(arr[i] - mark)`` for each ``mark`` in ``marks``.

    Returns one integer per mark; useful as a quantile-crossing feature on time series. Dispatches the common int32
    output path to a mark-parallel kernel (sequential per-mark scan, ~31x at n=1e6); other dtypes use the serial kernel.
    """
    if dtype is np.int32 or dtype == np.int32:
        return cast(np.ndarray, _compute_ncrossings_marks_prange(arr, marks))
    return cast(np.ndarray, _compute_ncrossings_serial(arr, marks, dtype))


@numba.njit(**NUMBA_NJIT_PARAMS)
def compute_nunique_mode_quantiles_numba(arr: np.ndarray, q: Sequence[float] = default_quantiles) -> tuple:  # pragma: no cover
    """
    NOT RECOMMENDED. use compute_nunique_modes_quantiles_numpy instead, it's faster and more functional.
    numUnique and mode calculation from sorted array
    CAN be numba jitted.
    """
    xsorted = np.sort(arr)

    next_unique_value = xsorted[0]
    numUnique = 1
    mode = next_unique_value
    best_count = 1
    times_occured = 0
    for next_value in xsorted:
        if next_value == next_unique_value:
            times_occured = times_occured + 1
        else:
            numUnique = numUnique + 1
            if times_occured > best_count:
                best_count = times_occured
                mode = next_unique_value
            next_unique_value = next_value
            times_occured = 1
    if times_occured > best_count:
        best_count = times_occured
        mode = next_unique_value

    factor = len(arr)
    quantiles = []
    for quantile in q:
        # Clamp index >= 0: at quantile=0.0 the formula yields -1 which would wrap to the LAST element (max) instead of the first (min).
        idx = int(np.ceil(quantile * factor)) - 1
        if idx < 0:
            idx = 0
        elif idx >= factor:
            idx = factor - 1
        quantiles.append(xsorted[idx])

    if times_occured == 1:
        mode = np.nan
    return numUnique, mode, quantiles
