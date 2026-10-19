"""Stages of ``honest_null_columns_device``."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
from mlframe.feature_selection.filters._honest_null_device import XLOGX_TABLE_SIZE
from mlframe.feature_selection.filters._honest_null_device import _accepted
from mlframe.feature_selection.filters._honest_null_device import _entropy
from mlframe.feature_selection.filters._honest_null_device import _get_kernel
from mlframe.feature_selection.filters._honest_null_device import _get_kernel_k2
from mlframe.feature_selection.filters._honest_null_device import _pack_bits
from mlframe.feature_selection.filters._honest_null_device import _tie_flags
from mlframe.feature_selection.filters._mdlp_honest_njit import _presort_columns
# --- end imports ---


def _continuous_accept_block(kc, raw_cont, y_raw_perm, n_classes_total, n, xlogx_table, cp, n_perm, min_split_size, alpha, accepted_c, order, tie):
    """Stage of ``honest_null_columns_device``."""
    if kc:
        order = _presort_columns(raw_cont)
        tie = _tie_flags(raw_cont, order)
        counts = np.bincount(np.asarray(y_raw_perm[0], dtype=np.int64), minlength=n_classes_total)
        h_full = _entropy(counts, n)
        tab = xlogx_table(XLOGX_TABLE_SIZE)
        d_gain = cp.empty(n_perm * kc, dtype=cp.float64)  # Intentional float64: statistic vector (best split gains, compared with the host's)
        d_ncand = cp.empty(n_perm * kc, dtype=cp.int32)
        d_idx = cp.empty(n_perm * kc, dtype=cp.int32)
        if n_classes_total == 2:
            ybits = _pack_bits(y_raw_perm)
            _get_kernel_k2(cp)(
                ((n_perm * kc + 127) // 128,),
                (128,),
                (
                    cp.asarray(order),
                    cp.asarray(tie),
                    cp.asarray(np.ascontiguousarray(ybits.T).view(np.uint8)),  # (words, permutations): a warp of consecutive permutations reads consecutive words
                    cp.asarray(tab),
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
            _get_kernel(cp)(
                ((n_perm * kc + 127) // 128,),
                (128,),
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
        ncand = np.ascontiguousarray(cp.asnumpy(d_ncand).reshape(kc, n_perm).T)
        bidx = np.ascontiguousarray(cp.asnumpy(d_idx).reshape(kc, n_perm).T)
        if h_full > 0.0:
            accepted_c = _accepted(gain, ncand, bidx, n, n_classes_total, float(alpha))
    return accepted_c, order, tie
