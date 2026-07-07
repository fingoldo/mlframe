"""Multilabel metrics for ``mlframe.metrics.core``.

Split out from ``core.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; every moved symbol is
re-exported from ``core`` so existing
``from mlframe.metrics.core import hamming_loss`` (and the other moved
names) imports continue to work.

What lives here:
  - Elementwise @njit kernels (seq + par) for hamming, subset_accuracy,
    jaccard.
  - Popcount + bitmap-Jaccard fastpath kernels for ``K <= 64``:
    ``_popcount64``, ``_fast_jaccard_bitmap_seq``,
    ``_can_use_bitmap_jaccard``, ``_pack_for_bitmap``.
  - Shape coercion / validation: ``_coerce_multilabel_array``,
    ``_validate_multilabel_pair``.
  - Public dispatchers: ``hamming_loss``, ``subset_accuracy``,
    ``jaccard_score_multilabel``.

All three accept either ``(N, K)`` binary indicator matrices or ``(N,)``
binary arrays (auto-reshaped to ``(N, 1)`` by the public wrappers).

Sequential variants are the default. Parallel variants
(``@njit(parallel=True)``) are auto-selected by the public wrapper when
``N * K > 1_000_000`` -- benchmarked threshold on Win32 where
``numba.prange`` cold-spawn cost is ~40-80ms (rules out small-frame
parallelism).

All three follow sklearn semantics:
- ``hamming_loss``: mean fraction of incorrect labels (lower is better).
- ``subset_accuracy``: fraction of samples where ALL labels match (exact
  match).
- ``jaccard_score_multilabel``: per-row averaged ``|y_true & y_pred| /
  |y_true | y_pred|``; empty-union row counts as 1.0 (defined as "both
  empty = perfect"; sklearn ``jaccard_score(average='samples')`` raises
  in that case unless ``zero_division`` is explicit -- we pick 1.0 as
  the well-defined choice).
"""
from __future__ import annotations

import numpy as np
import numba

from ._numba_params import NUMBA_NJIT_PARAMS, _PARALLEL_MULTILABEL_THRESHOLD


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_hamming_loss_seq(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sequential mean-mismatch fraction. Both inputs (N, K) uint8."""
    N, K = y_true.shape
    err = 0.0
    for i in range(N):
        for j in range(K):
            if y_true[i, j] != y_pred[i, j]:
                err += 1.0
    return float(err / (N * K))


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_hamming_loss_par(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Parallel variant. Auto-selected by hamming_loss() when N*K > 1M.

    The prange body is a pure per-row MAP (writes one independent ``err_per_row[i]`` per iteration,
    no scalar accumulator crossing iterations); the final ``sum/N`` runs in a separate plain ``range``
    loop. This shape is required to dodge numba 0.63.1's parfor reduction-analysis bug ("unexpected
    cycle in lookup()"): an inner-loop accumulator that is read AFTER the loop and combined with a
    reduction (the old ``err_per_row[i] = local/K`` then ``.mean()``) builds a cyclic def-use chain
    that aborts compilation. Keeping the reduction out of the parfor avoids the analysis entirely.
    """
    N, K = y_true.shape
    err_per_row = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        c = 0.0
        for j in range(K):
            if y_true[i, j] != y_pred[i, j]:
                c += 1.0
        err_per_row[i] = c / K
    total = 0.0
    for i in range(N):
        total += err_per_row[i]
    return float(total / N)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_subset_accuracy_seq(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Row-wise all-equal then mean. Both inputs (N, K) uint8."""
    N, K = y_true.shape
    correct = 0.0
    for i in range(N):
        all_eq = True
        for j in range(K):
            if y_true[i, j] != y_pred[i, j]:
                all_eq = False
                break
        if all_eq:
            correct += 1.0
    return float(correct / N)


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_subset_accuracy_par(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Parallel subset accuracy. ~5.6x faster than seq at N=1M.
    Public ``subset_accuracy`` wrapper auto-selects above N=50k."""
    N, K = y_true.shape
    correct = 0
    for i in numba.prange(N):
        all_eq = True
        for j in range(K):
            if y_true[i, j] != y_pred[i, j]:
                all_eq = False
                break
        if all_eq:
            correct += 1
    return float(correct / N)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_jaccard_score_seq(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Per-row Jaccard (|A & B|/|A | B|), averaged. Empty-union -> 1.0."""
    N, K = y_true.shape
    total = 0.0
    for i in range(N):
        intersect = 0.0
        union = 0.0
        for j in range(K):
            t = y_true[i, j]
            p = y_pred[i, j]
            if t == 1 and p == 1:
                intersect += 1.0
            if t == 1 or p == 1:
                union += 1.0
        if union > 0:
            total += intersect / union
        else:
            total += 1.0  # both empty -- perfect by definition
    return float(total / N)


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_jaccard_score_par(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Parallel per-row Jaccard. ~6.2x faster than seq at N=1M.
    Public ``jaccard_score_multilabel`` wrapper auto-selects above N=50k.

    Same numba 0.63.1 parfor workaround as ``_fast_hamming_loss_par``: the prange body writes one
    independent ``row_score[i]`` per iteration (pure map, no cross-iteration scalar reduction); the
    final sum runs in a separate plain ``range`` loop. A direct ``total += intersect/union`` scalar
    reduction inside the conditional aborts compilation with "unexpected cycle in lookup()".
    """
    N, K = y_true.shape
    row_score = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        intersect = 0.0
        union = 0.0
        for j in range(K):
            t = y_true[i, j]
            p = y_pred[i, j]
            if t == 1 and p == 1:
                intersect += 1.0
            if t == 1 or p == 1:
                union += 1.0
        if union > 0:
            row_score[i] = intersect / union
        else:
            row_score[i] = 1.0  # both empty -- perfect by definition
    total = 0.0
    for i in range(N):
        total += row_score[i]
    return float(total / N)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _popcount64(x: np.uint64) -> np.int64:
    """Population-count for uint64 -- Hacker's Delight bit-twiddle.

    Numba doesn't expose ``__builtin_popcountll`` directly; this 5-instruction
    sequence is ~as fast as the intrinsic on modern x86-64. Used by the
    bitmap-Jaccard fast path for K<=64 multilabel arrays.
    """
    x = x - ((x >> 1) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> 2) & np.uint64(0x3333333333333333))
    x = (x + (x >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int(np.int64((x * np.uint64(0x0101010101010101)) >> 56) & np.int64(0x7F))


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_jaccard_bitmap_seq(y_true_packed: np.ndarray, y_pred_packed: np.ndarray, K: int) -> float:
    """Bitmap Jaccard via popcount -- ~10-50x faster than elementwise loop on K<=64.

    Inputs are already packed uint64 of shape (N,) -- caller's responsibility
    to pack via ``np.packbits`` and view-as-uint64. K is the original number
    of labels (needed for empty-union detection -- packed zero means all-zero
    labels, regardless of K).
    """
    N = y_true_packed.shape[0]
    total = 0.0
    for i in range(N):
        t = y_true_packed[i]
        p = y_pred_packed[i]
        intersect = _popcount64(t & p)
        union = _popcount64(t | p)
        if union > 0:
            total += intersect / union
        else:
            total += 1.0  # both empty
    return float(total / N)


def _can_use_bitmap_jaccard(K: int) -> bool:
    """Bitmap Jaccard fits if 16 <= K <= 64 (single uint64 per row).

    K threshold benchmarks (N=200_000, jaccard_score_multilabel, Win32 Anaconda 3.11):
        K=3  : bitmap 22ms,  elem  4ms -- bitmap LOSES 5x (pack overhead)
        K=16 : bitmap 21ms,  elem 25ms -- bitmap wins 1.2x
        K=32 : bitmap 23ms,  elem 52ms -- bitmap wins 2.3x
        K=64 : bitmap 12ms,  elem 101ms -- bitmap wins 8.6x

    Cutoff at K=16 (~breakeven) means the elementwise loop wins for the
    common 3-5-label case and bitmap kicks in for tag-cloud cases (K>=16).
    """
    return 16 <= K <= 64


@numba.njit(**NUMBA_NJIT_PARAMS)
def _pack_for_bitmap_kernel_seq(arr: np.ndarray) -> np.ndarray:
    """Direct (N, K<=64) uint8 -> (N,) uint64 packing in one pass.

    Bit-identical to ``np.packbits(pad64(arr), axis=1).view(uint64)`` on a
    little-endian host: ``np.packbits`` is big-endian WITHIN each byte and
    the LE uint64 view then reverses byte order, so label ``j`` lands at bit
    ``(j>>3)*8 + (7 - (j&7))`` of the word. Computing that index directly
    avoids the old path's full ``(N, 64)`` zero-buffer alloc + elementwise
    copy + ``packbits`` over all 64 columns when only ``K`` are populated.
    """
    N, K = arr.shape
    out = np.zeros(N, dtype=np.uint64)
    one = np.uint64(1)
    for i in range(N):
        w = np.uint64(0)
        for j in range(K):
            if arr[i, j]:
                pos = (j >> 3) * 8 + (7 - (j & 7))
                w |= one << np.uint64(pos)
        out[i] = w
    return out


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _pack_for_bitmap_kernel_par(arr: np.ndarray) -> np.ndarray:
    """Parallel twin of ``_pack_for_bitmap_kernel_seq`` (pure per-row map)."""
    N, K = arr.shape
    out = np.zeros(N, dtype=np.uint64)
    one = np.uint64(1)
    for i in numba.prange(N):
        w = np.uint64(0)
        for j in range(K):
            if arr[i, j]:
                pos = (j >> 3) * 8 + (7 - (j & 7))
                w |= one << np.uint64(pos)
        out[i] = w
    return out


def _pack_for_bitmap_numpy(arr: np.ndarray) -> np.ndarray:
    """Reference numpy packer. Kept callable for the bit-identity bench/test."""
    N, K = arr.shape
    if K < 64:
        padded = np.zeros((N, 64), dtype=np.uint8)
        padded[:, :K] = arr
    else:
        padded = arr  # K == 64 exactly
    packed_u8 = np.packbits(padded, axis=1)  # (N, 8) uint8
    return packed_u8.view(np.uint64).ravel()  # (N,) uint64


def _pack_for_bitmap(arr: np.ndarray) -> np.ndarray:
    """Pack a (N, K) uint8 binary array into (N,) uint64.

    Handles K not multiple of 8 by zero-padding to next 64-bit boundary.
    Excess bits are zero -- they contribute 0 to popcount, so safe.

    Dispatches to a fused njit kernel (bit-identical to the numpy reference,
    verified across K in 16..64 incl. non-byte-aligned K and all-zero/-one
    rows): the njit path skips the old ``(N, 64)`` zero buffer + 64-wide
    ``packbits``, giving ~8-11x at K<=32 and ~2x at K=64 (n=200k microbench).
    The parallel twin auto-selects above ``_PARALLEL_MULTILABEL_THRESHOLD``
    rows; tiny frames keep the serial njit (prange spawn ~40-80us not amortised).
    """
    if arr.shape[0] >= _PARALLEL_MULTILABEL_THRESHOLD:
        return np.asarray(_pack_for_bitmap_kernel_par(arr))
    return np.asarray(_pack_for_bitmap_kernel_seq(arr))


def _coerce_multilabel_array(arr) -> np.ndarray:
    """Single-pass cast to contiguous uint8 (N, K). Reshape (N,) -> (N, 1)."""
    a = np.ascontiguousarray(np.asarray(arr), dtype=np.uint8)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    raise ValueError(f"multilabel array must be 1-D or 2-D, got shape {a.shape}")


def _validate_multilabel_pair(y_true, y_pred) -> tuple:
    """Coerce + validate y_true / y_pred shape match BEFORE calling numba.

    Numba @njit kernels do not bounds-check inner loops; passing arrays
    of mismatched second-dimension would silently read garbage memory.
    Public wrappers MUST validate up-front.
    """
    yt = _coerce_multilabel_array(y_true)
    yp = _coerce_multilabel_array(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true shape {yt.shape} != y_pred shape {yp.shape}; " "multilabel metrics require matching shapes.")
    return yt, yp


def hamming_loss(y_true, y_pred) -> float:
    """sklearn-compatible Hamming loss for multilabel targets.

    Accepts (N,) binary or (N, K) multilabel. For N*K > 1M, auto-routes
    to the parallel numba variant.

    Same return-value semantics as ``sklearn.metrics.hamming_loss``.
    """
    yt, yp = _validate_multilabel_pair(y_true, y_pred)
    if yt.shape[0] * yt.shape[1] > 1_000_000:
        return float(_fast_hamming_loss_par(yt, yp))
    return float(_fast_hamming_loss_seq(yt, yp))


def subset_accuracy(y_true, y_pred) -> float:
    """Subset accuracy (a.k.a. exact-match) for multilabel targets.

    Equivalent to ``sklearn.metrics.accuracy_score(y_true, y_pred)`` on
    multilabel inputs (sklearn does row-wise all-equal under the hood).
    Auto-dispatches to the parallel kernel above N=50k rows
    (~5.6x speedup at 1M rows on an 8-thread runtime).
    """
    yt, yp = _validate_multilabel_pair(y_true, y_pred)
    if yt.shape[0] >= _PARALLEL_MULTILABEL_THRESHOLD:
        return float(_fast_subset_accuracy_par(yt, yp))
    return float(_fast_subset_accuracy_seq(yt, yp))


def jaccard_score_multilabel(y_true, y_pred, *, force_elementwise: bool = False) -> float:
    """Per-row averaged Jaccard score for multilabel targets.

    Equivalent to ``sklearn.metrics.jaccard_score(y_true, y_pred, average='samples')``
    with the well-defined choice of 1.0 for empty-union rows
    (sklearn's default raises a ``DivisionWarning`` on those).

    Performance: when ``K <= 64`` (the common case in multilabel tagging),
    uses a bitmap-popcount fast path (~10-50x faster than the elementwise
    loop) -- these are sequential because the per-row work is tiny on
    bit-packed input. For elementwise mode (``force_elementwise=True``
    or K>64), the parallel kernel kicks in above N=50k rows
    (~6x speedup at 1M rows). Set ``force_elementwise=True`` to bypass
    the bitmap path -- useful for benchmarks and verifying numerical
    equivalence between the two paths.
    """
    yt, yp = _validate_multilabel_pair(y_true, y_pred)
    K = yt.shape[1]
    if not force_elementwise and _can_use_bitmap_jaccard(K):
        yt_packed = _pack_for_bitmap(yt)
        yp_packed = _pack_for_bitmap(yp)
        return float(_fast_jaccard_bitmap_seq(yt_packed, yp_packed, K))
    if yt.shape[0] >= _PARALLEL_MULTILABEL_THRESHOLD:
        return float(_fast_jaccard_score_par(yt, yp))
    return float(_fast_jaccard_score_seq(yt, yp))
