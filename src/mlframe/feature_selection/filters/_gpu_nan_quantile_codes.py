"""Quantile codes of a resident candidate matrix whose columns may hold NaN, equal to the host binner's.

The host binner (``discretize_2d_quantile_batch``) takes the percentile edges of a column over its non-NaN rows (``np.nanpercentile``, linear method: virtual index ``(m - 1) * q`` with ``m`` the number of non-NaN values) and codes a value as
``searchsorted(edges[1:-1], x, side="right")``, a NaN going to the rightmost bin. The device binner of the finite path takes its edges over every row, so a candidate holding NaN (a missing external factor) used to send the whole ext-val
batch to the host (host materialise, host percentile, upload).

Here each column block is transposed to rows, sorted (NaN sorts last), the interior edges are read at ``(m - 1) * q`` of each column's own ``m`` with numpy's lerp (the ``t >= 0.5`` form), and a code is the number of interior edges ``<= x``; a NaN
gets the top code. Columns with fewer than two non-NaN values or any infinite value are not handled (``None``: the caller keeps the host path), as an infinite value makes the interpolated edge undefined.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_BYTES_PER_COLUMN_ROW = 40  # float64 block, its sorted copy, the sort workspace and the comparison temporaries
_BUDGET_FRACTION = 0.2
_COMPARE_COLUMNS = 8


def _block_columns(cp: Any, n_rows: int, n_cols: int) -> int:
    """Columns per block that the memory pool can still hold at ``_BUDGET_FRACTION`` (at least 4)."""
    try:
        pool = cp.get_default_memory_pool()
        free_device = int(cp.cuda.runtime.memGetInfo()[0]) + int(pool.free_bytes())
        limit = int(pool.get_limit())
        room = free_device if limit <= 0 else min(free_device, max(0, limit - int(pool.used_bytes())))
    except Exception as exc:
        logger.debug("nan quantile codes: free memory unknown (%s)", exc)
        return 16
    return int(max(4, min(n_cols, (_BUDGET_FRACTION * room) // (_BYTES_PER_COLUMN_ROW * max(1, n_rows)))))


def nan_aware_quantile_codes_gpu(cand: Any, nbins: int) -> Any:
    """``(n, K)`` int32 codes (resident) of the columns of the resident float64 matrix ``cand`` ``(n, K)``, NaN-aware as the host binner; ``None`` when a column has fewer than two non-NaN values or an infinite one."""
    import cupy as cp

    n, k = int(cand.shape[0]), int(cand.shape[1])
    if n < 2 or k == 0 or int(nbins) < 2:
        return None
    qs = cp.asarray(np.linspace(0.0, 100.0, int(nbins) + 1)[1:-1] / 100.0)  # the interior quantiles, as np.percentile divides q by 100
    n_edges = int(nbins) - 1
    codes = cp.empty((n, k), dtype=cp.int32)
    block = _block_columns(cp, n, k)
    for c0 in range(0, k, block):
        c1 = min(k, c0 + block)
        xb = cp.ascontiguousarray(cand[:, c0:c1].T)  # (kb, n)
        nan = cp.isnan(xb)
        valid = n - nan.sum(axis=1)
        if bool(cp.any(cp.isinf(xb))) or int(valid.min()) < 2:
            return None
        xs = cp.sort(xb, axis=1)
        pos = (valid - 1)[:, None].astype(cp.float64) * qs[None, :]  # (kb, n_edges)
        lo = cp.floor(pos).astype(cp.int64)
        gamma = pos - lo
        a = cp.take_along_axis(xs, lo, axis=1)
        b = cp.take_along_axis(xs, lo + 1, axis=1)
        del xs
        diff = b - a
        edges = cp.where(gamma >= 0.5, b - diff * (1.0 - gamma), a + diff * gamma)  # numpy's lerp
        for s0 in range(0, c1 - c0, _COMPARE_COLUMNS):
            s1 = min(c1 - c0, s0 + _COMPARE_COLUMNS)
            x = xb[s0:s1]
            cnt = (edges[s0:s1, :, None] <= x[:, None, :]).sum(axis=1)
            codes[:, c0 + s0 : c0 + s1] = cp.where(nan[s0:s1], n_edges, cnt).T.astype(cp.int32)
    return codes


__all__ = ["nan_aware_quantile_codes_gpu"]
