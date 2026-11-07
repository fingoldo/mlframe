"""Stages of ``honest_null_columns_device``."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for
from mlframe.feature_selection.filters._honest_null_device import XLOGX_TABLE_SIZE
from mlframe.feature_selection.filters._honest_null_device import _accepted
from mlframe.feature_selection.filters._honest_null_device import _entropy
from mlframe.feature_selection.filters._honest_null_device import _get_kernel
from mlframe.feature_selection.filters._honest_null_device import _get_kernel_k2
from mlframe.feature_selection.filters._honest_null_device import _pack_bits_cached
from mlframe.feature_selection.filters._honest_null_device import _tie_flags
from mlframe.feature_selection.filters._mdlp_honest_njit import _presort_columns
# --- end imports ---


def _int_to_host(cp, dev, bound: int) -> np.ndarray:
    """``cp.asnumpy(dev)`` for an integer device array whose values lie in ``[0, bound]``, copied as int16 when ``bound`` fits and widened back on the host."""
    if int(bound) <= 32767 and dev.dtype.kind in "iu" and dev.dtype.itemsize > 2:
        return np.asarray(cp.asnumpy(dev.astype(cp.int16))).astype(dev.dtype)
    return np.asarray(cp.asnumpy(dev))


def _use_bitset_accept() -> bool:
    """Whether the binary-target acceptance runs as the sorted-space bitset scan with the provable prune (``_honest_null_accept_bits``): ``MLFRAME_HONEST_ROOT_SPLIT=accept`` (default) or ``k5`` / ``k4`` / ``k2`` for the per-step kernels."""
    import os

    return os.environ.get("MLFRAME_HONEST_ROOT_SPLIT", "accept").strip().lower() == "accept"


def _bitset_accept(cp, order, tie, y_raw_perm, n, n_perm, kc, min_split_size, alpha, h_full, tab_full, d_gain, d_ncand, d_idx) -> bool:
    """The bitset acceptance scan into the three output vectors; ``False`` (nothing written) when it does not apply or the device refuses, so the per-step kernel runs."""
    try:
        from mlframe.feature_selection.filters._honest_null_accept_bits import root_split_accept_gpu

        return bool(root_split_accept_gpu(cp, order, tie, y_raw_perm, n, n_perm, kc, min_split_size, alpha, h_full, cp.asarray(tab_full), d_gain, d_ncand, d_idx))
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("bitset honest-null acceptance failed (%s: %s); using the per-step kernel", type(exc).__name__, exc)
        return False


def _root_split_kernel(cp):
    """``(kernel, full_table)`` of the binary-target root-split kernel, chosen by ``MLFRAME_HONEST_ROOT_SPLIT``: ``k5`` (default: one float pass, exact gain once at the best cut, selection-equivalent to v2), ``k4`` (one exact pass over a full
    ``x ln x`` table, outputs equal to v2) or ``k2`` (the two-pass original). ``full_table`` says the kernel reads a table over every count."""
    import os

    mode = os.environ.get("MLFRAME_HONEST_ROOT_SPLIT", "k5").strip().lower()  # "accept" (the default of the caller) falls through to k5 when the bitset scan declines
    if mode == "k2" or os.environ.get("MLFRAME_HONEST_ROOT_SPLIT_K4", "") == "0":
        return _get_kernel_k2(cp), False
    from mlframe.feature_selection.filters._honest_null_k4 import get_kernel_k4, get_kernel_k5

    if mode == "k4":
        return get_kernel_k4(cp), True
    return get_kernel_k5(cp), False


def _order_and_ties(raw_cont):
    """``(order, tie)`` of the continuous columns: the stable sort order and the tie flags of the sorted rows, on the device for large finite blocks (the same permutation as the host's stable sort, ties included), else on the host."""
    from mlframe.feature_selection.filters._honest_null_presort_gpu import presort_and_ties_gpu, use_device_presort

    if use_device_presort(int(raw_cont.shape[0]), int(raw_cont.shape[1])):
        got = presort_and_ties_gpu(raw_cont)
        if got is not None:
            return got
    order = _presort_columns(raw_cont)
    return order, _tie_flags(raw_cont, order)


def _continuous_accept_block(kc, raw_cont, y_raw_perm, n_classes_total, n, xlogx_table, cp, n_perm, min_split_size, alpha, accepted_c, order, tie):
    """Stage of ``honest_null_columns_device``."""
    if kc:
        order, tie = _order_and_ties(raw_cont)
        counts = np.bincount(np.asarray(y_raw_perm[0], dtype=np.int64), minlength=n_classes_total)
        h_full = _entropy(counts, n)
        tab = xlogx_table(XLOGX_TABLE_SIZE)
        d_gain = cp.empty(n_perm * kc, dtype=cp.float64)  # Intentional float64: statistic vector (best split gains, compared with the host's)
        d_ncand = cp.empty(n_perm * kc, dtype=cp.int32)
        d_idx = cp.empty(n_perm * kc, dtype=cp.int32)
        if n_classes_total == 2 and _use_bitset_accept() and _bitset_accept(cp, order, tie, y_raw_perm, n, n_perm, kc, min_split_size, alpha, h_full, xlogx_table(n), d_gain, d_ncand, d_idx):
            pass
        elif n_classes_total == 2:
            ybits = _pack_bits_cached(y_raw_perm)
            _kernel, _full_table = _root_split_kernel(cp)
            threads = block_size_for(_kernel, work_items=n_perm * kc)
            _kernel(
                ((n_perm * kc + threads - 1) // threads,),
                (threads,),
                (
                    cp.asarray(order),
                    cp.asarray(tie),
                    cp.asarray(np.ascontiguousarray(ybits.T).view(np.uint8)),  # (words, permutations): a warp of consecutive permutations reads consecutive words
                    cp.asarray(xlogx_table(n) if _full_table else tab),
                    cp.asarray(counts.astype(np.int64)),
                    np.int32(n),
                    np.int32(n_perm),
                    np.int32(kc),
                    np.int32(min_split_size),
                    # Intentional float64: full-sample statistic handed to the device null kernel (must match the host gain)
                    np.float64(h_full),
                    np.int32(ybits.shape[1]),
                    d_gain,
                    d_ncand,
                    d_idx,
                ),
            )
        else:
            multi_kernel = _get_kernel(cp)
            threads = block_size_for(multi_kernel, work_items=n_perm * kc)
            multi_kernel(
                ((n_perm * kc + threads - 1) // threads,),
                (threads,),
                (
                    cp.asarray(order),
                    cp.asarray(tie),
                    cp.asarray(np.ascontiguousarray(np.asarray(y_raw_perm, dtype=np.int8).T)),  # (rows, permutations): coalesced across a warp
                    cp.asarray(tab),
                    cp.asarray(counts.astype(np.int64)),
                    np.int32(n_classes_total),
                    np.int32(n),
                    np.int32(n_perm),
                    np.int32(kc),
                    np.int32(min_split_size),
                    # Intentional float64: full-sample statistic handed to the device null kernel (must match the host gain)
                    np.float64(h_full),
                    d_gain,
                    d_ncand,
                    d_idx,
                ),
            )
        gain = np.ascontiguousarray(cp.asnumpy(d_gain).reshape(kc, n_perm).T)
        # The candidate counts and split positions are at most ``n``: below 32768 rows they cross to the host as int16 and are widened there, which gives the same arrays.
        ncand = np.ascontiguousarray(_int_to_host(cp, d_ncand, n).reshape(kc, n_perm).T)
        bidx = np.ascontiguousarray(_int_to_host(cp, d_idx, n).reshape(kc, n_perm).T)
        if h_full > 0.0:
            accepted_c = _accepted(gain, ncand, bidx, n, n_classes_total, float(alpha))
    return accepted_c, order, tie
