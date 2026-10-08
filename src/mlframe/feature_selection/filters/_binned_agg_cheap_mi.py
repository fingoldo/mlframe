"""Cheap quantile-binned MI helpers of the binned numeric-aggregate FE: edge computation, per-column / batched MI against the label codes.

Carved out of ``_binned_numeric_agg_fe``, which re-exports every name so the historical import path keeps working.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numba import njit, prange

logger = logging.getLogger("mlframe.feature_selection.filters._binned_numeric_agg_fe")

# Columns at or above this many rows take the device quantile (a host np.quantile there costs 5-10x a device sort plus the upload it overlaps with).
_DEVICE_QUANTILE_MIN_N = 200_000


def quantile_edges(x: np.ndarray, nbins: int) -> np.ndarray:
    """Inner quantile cut points (unique-deduped); code = searchsorted(edges, v, side='right')."""
    qs = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    x = np.asarray(x, dtype=np.float64)
    if x.size >= _DEVICE_QUANTILE_MIN_N:
        # Large column under the strict-resident path: one device sort + the order statistics numpy would read, bit-identical to np.quantile (~115 ms
        # per 1M-row column on the host).
        try:
            from ._gpu_strict_fe import fe_gpu_strict_resident_enabled

            if fe_gpu_strict_resident_enabled():
                from ._device_quantile import device_quantile

                dev = device_quantile(x, qs)
                if dev is not None:
                    return np.unique(dev)
        except Exception as e:
            logger.debug("device quantile edges failed, using the host np.quantile: %s", e)
    return np.unique(np.quantile(x, qs))


def _cheap_mi_with_y(col: np.ndarray, y_codes: np.ndarray, nbins: int = 10) -> float:
    """Cheap MI(qbin(col); y_codes) via a bincount joint histogram - the relevance proxy for GROUP pre-selection.
    A group column only helps if its quantile cells separate y; this scores exactly that in O(n)."""
    cv = np.asarray(col, dtype=np.float64)
    # min != max on an all-finite column == unique().size >= 2 without the full-n unique SORT.
    if not np.isfinite(cv).all() or float(np.min(cv)) == float(np.max(cv)):
        return 0.0
    # DEVICE route (STRICT-resident): the full-n np.quantile edges + searchsorted codes + bincount joint all
    # run on the GPU; only the MI scalar returns. Same 'linear' percentile edges / joint counts -> the group
    # pre-selection RANKING is unchanged (near-ULP). Any cupy fault -> the exact host path below.
    try:
        import os as _os
        from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
        # No size gate under STRICT (user contract: 100% residency, no KTC-style size dispatch).
        if _os.environ.get("MLFRAME_FE_CHEAP_MI_GPU", "1").strip().lower() in ("1", "true", "on", "yes") and fe_gpu_strict_resident_enabled():
            import cupy as cp
            from ._resident_bincount import resident_bincount
            from ._fe_resident_operands import resident_operand
            cvd = cp.asarray(cv)
            _n = int(cvd.size)
            # Sync-free interior percentile edges (manual sort + linear interp, no cp.quantile host read) and NO
            # cp.unique: this only computes an MI, which is invariant to empty bins, so duplicate/relabeled bins
            # do not change the result. searchsorted against the nbins-1 interior edges yields codes in
            # [0, nbins-1] -> na = nbins is a known width (no int(max) sync). resident_bincount avoids the
            # cp.bincount int(max) sync. Only the y cardinality (nb) and the final MI scalar cross the bus.
            _qs = cp.asarray(np.linspace(0.0, 1.0, nbins + 1)[1:-1])
            _xs = cp.sort(cvd.ravel())
            _pos = _qs * (_n - 1)
            _lo = cp.floor(_pos).astype(cp.int64)
            _hi = cp.minimum(_lo + 1, _n - 1)
            _frac = _pos - _lo
            # numpy's form, exact on a tied edge; a*(1-f) + b*f rounds it one ULP off and shifts the tied mass to the next bin.
            e_d = _xs[_lo] + (_xs[_hi] - _xs[_lo]) * _frac
            xc_d = cp.searchsorted(e_d, cvd, side="right").astype(cp.int64)
            # y_codes is the SAME target re-used by every gcands candidate in this loop (fit-constant) AND
            # by the downstream survivor-stage device gate (_binned_numeric_agg_resident.local_mi_gate_binagg_
            # resident, role "y_mi_classif") - route through the content-keyed resident cache under the SAME
            # role string so a repeat/cross-call upload of identical y-code bytes shares one device buffer
            # instead of re-uploading. Content-keyed, so this is safe even when the two sites' y-encodings
            # differ (classification: both paths reduce to the same np.unique(..., return_inverse=True) codes
            # and always dedupe; continuous y: this site's quantile_edges+searchsorted vs the resident gate's
            # _quantile_bin can legitimately produce different edge/code bytes on tied/degenerate data, in
            # which case this is simply a cache miss - never a correctness issue).
            yc_d = resident_operand(np.ascontiguousarray(y_codes), "y_mi_classif", dtype=np.int64)
            na = int(nbins)
            nb = int(yc_d.max()) + 1
            joint = resident_bincount(cp, xc_d * nb + yc_d, na * nb, dtype=cp.float64).reshape(na, nb)
            pj = joint / cv.size
            pa = pj.sum(axis=1, keepdims=True)
            pb = pj.sum(axis=0, keepdims=True)
            term = cp.where(pj > 0, pj * cp.log(pj / (pa * pb)), 0.0)
            return float(cp.nansum(term))
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed: %s", e)
        pass
    edges = quantile_edges(cv, nbins)
    if edges.size == 0:
        return 0.0
    xc = np.searchsorted(edges, cv, side="right")
    return float(compute_mi_from_codes(xc, y_codes))


def _cheap_mi_group_selection(X: pd.DataFrame, gcands: Sequence[str], y_codes: np.ndarray, nbins: int = 10) -> dict:
    """GROUP MI(qbin(g); y) pre-selection over ``gcands`` -- batched-parallel CPU path by default.

    Under ``fe_gpu_strict_resident_enabled()`` (the diagnostic FULL-GPU-coverage mode), stays on
    ``_cheap_mi_with_y``'s existing per-column loop unchanged, so that mode's every-kernel-on-device
    contract and its GPU/host selection-equivalence guarantees are untouched. Otherwise batches every
    finite/non-constant candidate through ONE ``_cheap_mi_batch_njit`` prange call instead of walking them
    one at a time in a serial Python loop -- bit-identical values, just computed concurrently across cores.
    """
    try:
        from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
        if fe_gpu_strict_resident_enabled():
            return {g: _cheap_mi_with_y(X[g].to_numpy(), y_codes, nbins) for g in gcands}
    except Exception as e:
        logger.debug("_cheap_mi_group_selection: GPU-strict-resident probe failed, using the batched CPU path: %s", e)
    out: dict = {}
    valid_names: list = []
    valid_cols: list = []
    for g in gcands:
        cv = np.asarray(X[g].to_numpy(), dtype=np.float64)
        if not np.isfinite(cv).all() or float(np.min(cv)) == float(np.max(cv)):
            out[g] = 0.0
            continue
        valid_names.append(g)
        valid_cols.append(cv)
    if valid_names:
        mat = np.ascontiguousarray(np.column_stack(valid_cols))
        y_codes_i64 = np.ascontiguousarray(y_codes, dtype=np.int64)
        mis = _cheap_mi_batch_njit(mat, y_codes_i64, int(nbins))
        for g, mi in zip(valid_names, mis):
            out[g] = float(mi)
    return out


@njit(cache=True, fastmath=True)
def _cheap_mi_edge_dedup_njit(cv: np.ndarray, y_codes: np.ndarray, nbins: int) -> float:
    """Bit-identical njit port of ``_cheap_mi_with_y``'s host fallback (``quantile_edges`` dedup +
    ``searchsorted`` + ``compute_mi_from_codes``), one column, for use inside ``_cheap_mi_batch_njit``'s
    ``prange`` over columns. Interior quantile edges via the SAME linear-interpolation order-statistic
    convention ``np.quantile`` uses (order statistics found via ``np.partition``, matching
    ``_fe_edge_mi._edge_bin_codes``'s technique), then adjacent-deduped (the edges are non-decreasing by
    quantile-function monotonicity, so a single adjacent-dedup pass is equivalent to ``np.unique`` on this
    array) -- unlike ``_fe_edge_mi.plugin_mi_classif_batch_edge_njit`` (the OTHER existing batched-edge-MI
    kernel), which deliberately does NOT dedup so it bit-matches the GPU orth-family kernel; that would
    change the effective bin count on tied columns and is NOT a safe drop-in for this call site."""
    n = cv.shape[0]
    nq = nbins - 1
    if nq <= 0 or n == 0:
        return 0.0
    los = np.empty(nq, dtype=np.int64)
    fracs = np.empty(nq, dtype=np.float64)
    kths = np.empty(2 * nq, dtype=np.int64)
    m = 0
    for k in range(nq):
        q = (k + 1) / nbins
        pos = q * (n - 1)
        lo = int(np.floor(pos))
        hi = lo + 1 if lo < n - 1 else lo
        los[k] = lo
        fracs[k] = pos - lo
        kths[m] = lo
        kths[m + 1] = hi
        m += 2
    part = np.partition(cv, kths[:m])
    raw_edges = np.empty(nq, dtype=np.float64)
    for k in range(nq):
        lo = los[k]
        hi = lo + 1 if lo < n - 1 else lo
        raw_edges[k] = part[lo] + (part[hi] - part[lo]) * fracs[k]
    edges = np.empty(nq, dtype=np.float64)
    ne = 0
    for k in range(nq):
        if ne == 0 or raw_edges[k] != edges[ne - 1]:
            edges[ne] = raw_edges[k]
            ne += 1
    if ne == 0:
        return 0.0
    na = ne + 1
    y_min = y_codes[0]
    y_max = y_codes[0]
    for i in range(1, y_codes.shape[0]):
        if y_codes[i] < y_min:
            y_min = y_codes[i]
        if y_codes[i] > y_max:
            y_max = y_codes[i]
    nb = (y_max - y_min) + 1
    hist_xy = np.zeros((na, nb), dtype=np.int64)
    hist_x = np.zeros(na, dtype=np.int64)
    hist_y = np.zeros(nb, dtype=np.int64)
    for i in range(n):
        v = cv[i]
        lo = 0
        hi = ne
        while lo < hi:
            mid = (lo + hi) // 2
            if v < edges[mid]:
                hi = mid
            else:
                lo = mid + 1
        b = lo
        c = y_codes[i] - y_min
        hist_xy[b, c] += 1
        hist_x[b] += 1
        hist_y[c] += 1
    log_n = np.log(n)
    mi = 0.0
    for b in range(na):
        if hist_x[b] == 0:
            continue
        log_hx = np.log(hist_x[b])
        for c in range(nb):
            n_xy = hist_xy[b, c]
            if n_xy == 0 or hist_y[c] == 0:
                continue
            mi += (n_xy / n) * (np.log(n_xy) + log_n - log_hx - np.log(hist_y[c]))
    if mi < 0.0:
        mi = 0.0
    return mi


@njit(cache=True, fastmath=True, parallel=True)
def _cheap_mi_batch_njit(X_cols: np.ndarray, y_codes: np.ndarray, nbins: int) -> np.ndarray:
    """``prange``-parallel batch of :func:`_cheap_mi_edge_dedup_njit` over every column of ``X_cols``.

    ``binned_numeric_agg_with_recipes``'s GROUP pre-selection called ``_cheap_mi_with_y`` once per
    candidate column in a serial Python dict comprehension -- each call independently sorting/binning the
    full ``n``-row column single-threaded (cProfile at n=2M: 228 calls / 47.1s tottime, ~207ms/call, one
    core at a time). The candidate columns are independent (no cross-column dependency), so this fuses them
    into ONE parallel-njit call spread across all cores instead of walking them one at a time -- bit-
    identical to the sequential loop (same per-column algorithm, see ``_cheap_mi_edge_dedup_njit``'s
    docstring), just computed concurrently."""
    k = X_cols.shape[1]
    out = np.zeros(k, dtype=np.float64)
    for j in prange(k):
        out[j] = _cheap_mi_edge_dedup_njit(np.ascontiguousarray(X_cols[:, j]), y_codes, nbins)
    return out


def compute_mi_from_codes(a: np.ndarray, b: np.ndarray) -> float:
    """Plug-in MI of two integer-code arrays via a 2-D bincount joint histogram (nats)."""
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    n = a.size
    joint = np.bincount(a * nb + b, minlength=na * nb).astype(np.float64).reshape(na, nb)
    pj = joint / n
    pa = pj.sum(axis=1, keepdims=True)
    pb = pj.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = pj * np.log(pj / (pa * pb))
    return float(np.nansum(np.where(pj > 0, term, 0.0)))
