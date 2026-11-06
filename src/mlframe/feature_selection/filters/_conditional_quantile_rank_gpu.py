"""Device twin of ``conditional_quantile_rank_fe`` for the fit-on-the-rows-ranked case (``x_i_fit`` and ``x_j_bins_fit`` not given).

The host loops over the bins, sorts each bin's finite values and takes ``searchsorted(side="right")`` of the bin's rows against them. On the device the same
numbers come from one ``lexsort`` over (bin, value): the right-hand rank of a row is the end of its tie run minus the start of its bin, divided by the bin's
finite count. Rows with a non-finite value sit in a sentinel bin and come back NaN, as on the host.

A device fault returns ``None`` and the caller takes the host path.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

#: Rows from which the device twin is the default choice (below it the host loop over a few bins is faster than the launches).
DEVICE_MIN_ROWS = 50_000


def use_device_cqr(n_rows: int) -> bool:
    """Whether the rank runs on the device: ``MLFRAME_CQR_DEVICE=0|1`` forces a path, else from ``DEVICE_MIN_ROWS`` rows when a device is usable."""
    forced = os.environ.get("MLFRAME_CQR_DEVICE", "").strip()
    if forced == "0":
        return False
    from mlframe.feature_selection.filters._gpu_policy import cuda_available_for_run, gpu_globally_disabled

    if gpu_globally_disabled() or not cuda_available_for_run():
        return False
    return forced == "1" or int(n_rows) >= DEVICE_MIN_ROWS


def _rank_device(x_i: np.ndarray, x_j_bins: np.ndarray, with_sorted_bins: bool) -> "tuple[np.ndarray, list[np.ndarray]] | None":
    """The within-bin percentile rank of ``x_i`` per ``x_j_bins`` value, double in ``[0, 1]`` (NaN for a non-finite row); ``None`` on a device fault.

    The result is ``(ranks, sorted_bins)``: ``sorted_bins[c]`` holds the sorted finite values of bin id ``c`` for ``c`` in ``0 .. x_j_bins.max()``, the reference a
    recipe stores (empty for a bin without finite values; the list is empty unless ``with_sorted_bins``)."""
    try:
        import cupy as cp

        from mlframe.feature_selection.filters._device_sort_safe import sort_room

        xv = cp.asarray(x_i)
        n = int(xv.shape[0])
        if n == 0:
            return np.empty(0, dtype=np.float64), []  # Intentional float64: the rank is a ratio of integer counts, exact in double, and the CPU twin returns float64
        finite = cp.isfinite(xv)
        codes = cp.asarray(x_j_bins).astype(cp.int64)
        cmin = codes.min()
        b = cp.where(finite, codes - cmin + 1, 0)
        v = cp.where(finite, xv, 0)
        with sort_room(v):
            order = cp.lexsort(cp.stack([v, b]))
        sv = v[order]
        sb = b[order]
        new_run = cp.empty(n, dtype=cp.bool_)
        new_run[0] = True
        new_run[1:] = (sb[1:] != sb[:-1]) | (sv[1:] != sv[:-1])
        run_id = cp.cumsum(new_run) - 1
        run_ends = cp.concatenate([cp.flatnonzero(new_run)[1:], cp.asarray([n])])
        end_of_pos = run_ends[run_id]
        counts = cp.bincount(b)
        starts = cp.cumsum(counts) - counts
        rank_sorted = (end_of_pos - starts[sb]).astype(cp.float64) / counts[sb].astype(cp.float64)  # Intentional float64: the rank is a ratio of integer counts, exact in double, and the CPU twin returns float64
        out = cp.empty(n, dtype=cp.float64)
        out[order] = rank_sorted
        out = cp.where(finite, cp.clip(out, 0.0, 1.0), cp.nan)
        ranks = cp.asnumpy(out)
        if not with_sorted_bins:
            return ranks, []
        host_sorted = cp.asnumpy(sv[int(counts[0]) :])
        per_bin = np.split(host_sorted, np.cumsum(cp.asnumpy(counts[1:]))[:-1]) if counts.size > 1 else []
        first = int(cmin)
        top = int(codes.max())
        sorted_bins = [per_bin[c - first] if 0 <= c - first < len(per_bin) else host_sorted[:0] for c in range(top + 1)]
        return ranks, sorted_bins
    except Exception as exc:
        logger.warning("device conditional quantile rank failed (%s: %s); host path", type(exc).__name__, exc)
        return None


def conditional_quantile_rank_device(x_i: np.ndarray, x_j_bins: np.ndarray) -> "np.ndarray | None":
    """The within-bin percentile rank of ``x_i`` per ``x_j_bins`` value, double in ``[0, 1]`` (NaN for a non-finite row); ``None`` on a device fault."""
    out = _rank_device(x_i, x_j_bins, False)
    return None if out is None else out[0]


def conditional_quantile_rank_device_with_bins(x_i: np.ndarray, x_j_bins: np.ndarray) -> "tuple[np.ndarray, list[np.ndarray]] | None":
    """:func:`conditional_quantile_rank_device` plus the per-bin sorted finite values a recipe stores; ``None`` on a device fault."""
    return _rank_device(x_i, x_j_bins, True)
