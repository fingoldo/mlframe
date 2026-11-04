"""Equi-frequency bin codes of a BLOCK of resident columns in one pass: the batched twin of ``_mi_greedy_cmi_fe_binning._sync_free_qbin_codes``.

The per-column binner launches about fifteen small kernels and a full sort for every column (233 ms for 100 columns x 100000 rows, a plain ``cp.sort`` of the whole block takes 77 ms). Here one ``cp.sort(axis=0)`` orders every column,
the quantile edges are taken for all columns together with the same lerp the per-column binner uses (so each edge is the same number), the duplicate / min / max edges are marked ``+inf`` as there, and a code is the number of kept edges that are
``<= x``, which is exactly what ``searchsorted(side="right")`` on the sorted kept edges returns. Same edges, same comparison, so the codes are bit-identical to the per-column binner for every column shape.

``resident_qbin_codes_block`` is the content-cached entry point: columns whose codes are already cached are served from the cache, the rest are binned together.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from mlframe.feature_selection.filters._device_sort_safe import device_argsort, device_sort  # noqa: F401  (thrust aborts the process on a short pool)

logger = logging.getLogger(__name__)

#: Columns compared with the edge table at once; the (rows, edges, columns) boolean temporary is the peak.
_COMPARE_COLUMNS = 16
_SORT_BYTES_PER_ROW_COLUMN = 24


def sync_free_qbin_codes_block(cp, xd, nbins: int):
    """Resident int64 codes ``(k, n)`` of the rows of the resident float matrix ``xd`` ``(k, n)`` (one column per row, contiguous); row ``j`` equals ``_sync_free_qbin_codes(cp, xd[j], nbins)``."""
    k, n = int(xd.shape[0]), int(xd.shape[1])
    qs = cp.linspace(0.0, 100.0, int(nbins) + 1)
    pos = qs / 100.0 * (n - 1)
    lo = cp.floor(pos).astype(cp.int64)
    hi = cp.minimum(lo + 1, n - 1)
    frac = (pos - lo)[None, :]
    picked = _order_statistics_rows(cp, xd, n, int(nbins))
    if picked is not None:
        xlo, xhi = picked
    else:
        xs = device_sort(xd, axis=1)
        xlo = xs[:, lo]
        xhi = xs[:, hi]
        del xs
    diff = xhi - xlo
    e = cp.where(frac >= 0.5, xhi - diff * (1 - frac), xlo + diff * frac)  # (k, nbins + 1), ascending per row
    dup = cp.zeros(e.shape, dtype=bool)
    dup[:, 1:] = e[:, 1:] == e[:, :-1]
    emin = e[:, 0:1]
    emax = e[:, -1:]
    has_interior = cp.any((e > emin) & (e < emax), axis=1, keepdims=True)
    excl = (e == emin) | dup | (has_interior & (e == emax))
    kept = cp.where(excl, cp.inf, e)
    out = cp.empty((k, n), dtype=cp.int64)
    if kept.dtype == cp.float64 and xd.dtype in (cp.float32, cp.float64):
        # one thread per value counts the kept edges <= x (the comparison in double precision, as the broadcast below does after its cast): the same integer,
        # without the (columns, edges, rows) boolean temporary and its reduction
        _count_kernel(cp)(cp.ascontiguousarray(xd), cp.ascontiguousarray(kept), np.int64(n), np.int32(kept.shape[1]), out, size=k * n)  # Intentional float64: the kept edges are double
        return out
    for c0 in range(0, k, _COMPARE_COLUMNS):
        c1 = min(k, c0 + _COMPARE_COLUMNS)
        x = xd[c0:c1].astype(kept.dtype, copy=False)
        out[c0:c1] = (kept[c0:c1, :, None] <= x[:, None, :]).sum(axis=1)
    return out


def _order_statistics_rows(cp, xd, n: int, nbins: int):
    """``(xs[:, lo], xs[:, hi])`` of the row-sorted ``xd`` ``(k, n)`` for the quantile positions of ``linspace(0, 100, nbins + 1)``, selected by the radix kernels
    (the values a sort would put there, in the input dtype) instead of sorting every row; ``None`` when the select does not apply (the caller sorts).
    The ranks are formed on the host with the device expression's formula, so no value is read back; ``MLFRAME_QBIN_BLOCK_RADIX=0`` keeps the sort."""
    import os

    if os.environ.get("MLFRAME_QBIN_BLOCK_RADIX", "1").strip() == "0" or xd.dtype not in (cp.float32, cp.float64) or not xd.flags.c_contiguous or n < 1:
        return None
    try:
        from ._gpu_resident_select_kernels import _RADIX_SELECT_MAXR, _RADIX_STATIC_SHARED_BYTES, _get_radix_select_f32_dispatch, _get_radix_select_kernel, _resolve_radix_threads

        pos = np.linspace(0.0, 100.0, nbins + 1) / 100.0 * (n - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, n - 1)
        uniq = np.unique(np.concatenate([lo, hi]))
        R = int(uniq.size)
        shmem = R * 256 * 4
        limit = int(cp.cuda.Device().attributes.get("MaxSharedMemoryPerBlock", 48 * 1024))
        if R > _RADIX_SELECT_MAXR or shmem > limit - _RADIX_STATIC_SHARED_BYTES:
            return None
        k = int(xd.shape[0])
        osv = cp.empty((R, k), dtype=xd.dtype)
        is_f32 = xd.dtype == cp.float32
        ker = _get_radix_select_f32_dispatch(n) if is_f32 else _get_radix_select_kernel(False)
        ker((k,), (_resolve_radix_threads(n),), (xd, np.int64(n), np.int32(k), cp.asarray(uniq, dtype=cp.int64), np.int32(R), osv), shared_mem=shmem)
        row = {int(r): i for i, r in enumerate(uniq)}
        lo_rows = cp.asarray([row[int(v)] for v in lo], dtype=cp.int64)
        hi_rows = cp.asarray([row[int(v)] for v in hi], dtype=cp.int64)
        return osv[lo_rows].T, osv[hi_rows].T
    except Exception as exc:
        logger.debug("radix order statistics for the qbin block not used (%s: %s); sorting", type(exc).__name__, exc)
        return None


_COUNT_KERNELS: dict = {}


def _count_kernel(cp):
    """Elementwise kernel: ``out[j, i]`` = number of ``kept[j, :]`` entries ``<= (double) x[j, i]``."""
    kern = _COUNT_KERNELS.get("k")
    if kern is None:
        kern = cp.ElementwiseKernel(
            "raw X x, raw float64 kept, int64 n, int32 ne",
            "raw int64 out",
            """
            const long long j = i / n;
            const double v = (double)x[i];
            long long c = 0;
            for (int t = 0; t < ne; ++t) c += (kept[j * ne + t] <= v) ? 1 : 0;
            out[i] = c;
            """,
            "qbin_count_kept_edges",
        )
        _COUNT_KERNELS["k"] = kern
    return kern


def _block_columns(cp, n_rows: int, n_cols: int) -> int:
    """Columns per sort launch that the memory pool can still hold (at least 1)."""
    try:
        pool = cp.get_default_memory_pool()
        free_device = int(cp.cuda.runtime.memGetInfo()[0]) + int(pool.free_bytes())
        limit = int(pool.get_limit())
        room = free_device if limit <= 0 else min(free_device, max(0, limit - int(pool.used_bytes())))
    except Exception as exc:
        logger.debug("qbin block: free memory unknown (%s); small blocks", exc)
        return 8
    return int(max(1, min(n_cols, (0.25 * room) // (_SORT_BYTES_PER_ROW_COLUMN * max(1, n_rows)))))


def resident_qbin_codes_block(columns: Sequence[np.ndarray], nbins: int, dtype) -> "list | None":
    """Resident int64 codes of each host column of ``columns``, same values and same code cache as ``resident_qbin_codes`` column by column; ``None`` when the device path fails (the caller keeps its per-column loop)."""
    import cupy as cp

    from . import _fe_resident_operands as ro
    from ._mi_greedy_cmi_fe_binning import _sync_free_qbin_codes

    hosts = [np.ascontiguousarray(np.asarray(c)) for c in columns]
    if not hosts or ro._disabled() or len({h.shape for h in hosts}) != 1 or hosts[0].ndim != 1:
        return None
    n = int(hosts[0].shape[0])
    sigs = [(h.shape, h.dtype.str, ro._content_hash_memoized(h), int(nbins), str(dtype)) for h in hosts]
    codes: list = [None] * len(hosts)
    with ro._FE_RESIDENT_OPERANDS_LOCK:
        for i, sig in enumerate(sigs):
            dev = ro._FE_RESIDENT_QBIN_CODES.get(sig)
            if dev is not None:
                ro._FE_RESIDENT_QBIN_CODES.move_to_end(sig)
                codes[i] = dev.astype(cp.int64)
    miss = [i for i, c in enumerate(codes) if c is None]
    if not miss:
        return codes
    try:
        step = _block_columns(cp, n, len(miss))
        for s0 in range(0, len(miss), step):
            part = miss[s0 : s0 + step]
            if len(part) == 1:
                block = _sync_free_qbin_codes(cp, cp.asarray(hosts[part[0]], dtype=dtype), int(nbins))[None, :]
            else:
                xd = cp.asarray(np.stack([hosts[i] for i in part], axis=0), dtype=dtype)
                block = sync_free_qbin_codes_block(cp, xd, int(nbins))
                del xd
            narrow = block.astype(cp.int16) if int(nbins) <= 32767 else block
            with ro._FE_RESIDENT_OPERANDS_LOCK:
                for j, i in enumerate(part):
                    codes[i] = block[j].copy()
                    ro._FE_RESIDENT_QBIN_CODES[sigs[i]] = narrow[j].copy()
                    if len(ro._FE_RESIDENT_QBIN_CODES) > ro._MAX_QBIN_CODE_ENTRIES:
                        ro._FE_RESIDENT_QBIN_CODES.popitem(last=False)
            del block, narrow
    except Exception:
        logger.debug("batched device quantile binning failed; per-column path", exc_info=True)
        return None
    return codes


def group_mi_block_gpu(columns: Sequence[np.ndarray], y_codes: np.ndarray, nbins: int) -> "np.ndarray | None":
    """``MI(qbin(column); y_codes)`` of every host column of ``columns`` (finite, not constant) in one device pass, as ``_cheap_mi_with_y`` computes it per column: numpy's linear-interpolated interior edges, ``searchsorted(side="right")`` codes, plug-in MI in nats.

    One ``cp.sort(axis=1)`` of the block gives every column's edges and one fused kernel bins, histograms and scores all columns. ``None`` when the device path fails or the (bins x y classes) tile does not fit, so the caller keeps its per-column loop."""
    try:
        import cupy as cp

        from ._fe_batched_mi import binned_mi_from_values_gpu
        from ._fe_gpu_batch._devices import crit_float_dtype

        y = np.ascontiguousarray(y_codes, dtype=np.int64)
        n = int(y.size)
        k = len(columns)
        if k == 0 or n < 2 or int(nbins) < 2:
            return None
        xd = cp.asarray(np.stack([np.ascontiguousarray(c) for c in columns], axis=0), dtype=crit_float_dtype())  # (k, n), one contiguous row per column
        xs = device_sort(xd, axis=1)
        pos = cp.asarray(np.linspace(0.0, 1.0, int(nbins) + 1)[1:-1]) * (n - 1)
        lo = cp.floor(pos).astype(cp.int64)
        hi = cp.minimum(lo + 1, n - 1)
        xlo = xs[:, lo]
        edges = cp.ascontiguousarray((xlo + (xs[:, hi] - xlo) * (pos - lo)[None, :]).T)  # (nbins - 1, k)
        del xs
        mi = binned_mi_from_values_gpu(None, edges, cp.asarray(y), int(nbins), int(y.max()) + 1, x_cm=xd)
        return None if mi is None else np.asarray(mi, dtype=np.float64)  # Intentional float64: MI statistic vector
    except Exception:
        logger.debug("batched device group MI failed; per-column path", exc_info=True)
        return None


__all__ = ["group_mi_block_gpu", "resident_qbin_codes_block", "sync_free_qbin_codes_block"]
