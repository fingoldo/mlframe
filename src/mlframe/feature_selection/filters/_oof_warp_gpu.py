"""Device twin of the scan of ``_oof_warp_fe.warp_candidates``: the held-out MI of every column's cross-fitted warp against the raw column, for a block of columns in one pass.

The host fits and scores one column at a time on the column pool (102 columns x 100000 rows: 0.72 s on eight threads, about 3 thread-seconds inside a fit). Every step of it is a per-column
sort, a binning or a count, so a block of columns is done with whole-block device operations:

1. one row-wise sort of the block gives each column's finite count, its number of distinct values (the nominal-like skip), the 19 warp edges and the bulk range;
2. ``oof_warp_codes`` bins every value against its column's edges; three weighted ``bincount`` passes give the per-(column, fold, bin) sums of ``x``, of the target and the counts;
3. the fold tables (bin mean of ``x``, shrunk bin mean of the target, on the other folds) are formed on the device and ``oof_warp_interp`` interpolates every row on the table of its own fold,
   skipping the empty bins (the host compresses them out first: the same table);
4. two half-block sorts give the held-out edges of the warp and of the raw column; the pointwise MI of both codes and the paired standard error, over all scored rows and over the bulk rows.

Only the five statistics per (target, column) and the usable flags come back. The accepted candidates (a handful) are then fitted by the host service, so an emitted column and its recipe are
the host's to the bit; the device decides which columns are worth that fit. Sums are accumulated with atomic adds in double precision, so a table value can differ from the host's in the last
bits (the statistics agree with the host's to 1e-12 relative in the sweep; the edges are numpy's to the bit).

Any device fault returns ``None`` and the caller scans on the host. The host/device choice is the kernel-tuning entry ``fe_oof_warp_scan_device`` (fallback: the host).
"""

from __future__ import annotations

import threading

import logging
from typing import Any, Optional, Sequence

import numpy as np

from ._ktc_fallback import host_device_sweep, rows_fallback
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

DEVICE_FALLBACK_ROWS = 10**9  # the tuner decides; until a sweep says the device wins the host pool runs
_BUDGET_FRACTION = 0.33
_BYTES_PER_ELEMENT = 96  # peak device bytes per (column, row) of a block: the block, its sorted copy, codes, keys, weights, the warp and the pointwise terms

_KERNELS: dict = {}
_KERNELS_LOCK = threading.RLock()  # serialises the get-or-build of the cache above (one build per key even with concurrent fits)


def _codes_kernel(cp: Any) -> Any:
    """``code`` = number of the column's edges ``<= v`` (``-1`` for a non-finite ``v``); ``edges`` holds ``ne`` ascending edges per column, the column being ``i / width``."""
    with _KERNELS_LOCK:
        ker = _KERNELS.get("codes")
        if ker is None:
            ker = _KERNELS["codes"] = cp.ElementwiseKernel(
                "T v, raw float64 edges, int32 ne, int64 width",  # Intentional float64: CUDA signature text: the parameter is a double in the kernel source
                "int32 code",
                """
            if (!isfinite(v)) {
                code = -1;
            } else {
                const double x = (double)v;
                const long long base = (i / width) * ne;
                int lo = 0, hi = ne;
                while (lo < hi) {
                    const int mid = (lo + hi) >> 1;
                    if (edges[base + mid] <= x) lo = mid + 1; else hi = mid;
                }
                code = lo;
            }
            """,
                "oof_warp_codes",
            )
        return ker


def _interp_kernel(cp: Any) -> Any:
    """The warp value of ``v`` on the table of its column (``i / width``) and fold ``f``: linear between the centres of the bins that hold rows, the end values outside, the fold mean for a
    non-finite ``v`` or an empty table (the arithmetic of ``_oof_warp_service._interp``)."""
    with _KERNELS_LOCK:
        ker = _KERNELS.get("interp")
        if ker is None:
            ker = _KERNELS["interp"] = cp.ElementwiseKernel(
                "T v, int32 f, raw float64 cx, raw float64 cy, raw float64 cnt, raw float64 mu, int32 nb, int32 nf, int64 width",  # Intentional float64: CUDA signature text: the parameter is a double in the kernel source
                "float64 o",
                """
            const long long cf = (i / width) * nf + f;
            const long long tb = cf * nb;
            const double fill = mu[cf];
            if (!isfinite(v)) {
                o = fill;
            } else {
                const double x = (double)v;
                int first = -1, last = -1, lo = -1, hi = -1;
                for (int b = 0; b < nb; ++b) {
                    if (cnt[tb + b] > 0.0) {
                        if (first < 0) first = b;
                        last = b;
                        if (cx[tb + b] <= x) lo = b; else if (hi < 0) hi = b;
                    }
                }
                if (first < 0) {
                    o = fill;
                } else if (x <= cx[tb + first]) {
                    o = cy[tb + first];
                } else if (x >= cx[tb + last] || hi < 0 || lo < 0) {
                    o = cy[tb + last];
                } else {
                    const double d = cx[tb + hi] - cx[tb + lo];
                    o = d <= 0.0 ? cy[tb + lo] : cy[tb + lo] + (cy[tb + hi] - cy[tb + lo]) * (x - cx[tb + lo]) / d;
                }
            }
            """,
                "oof_warp_interp",
            )
        return ker


def _row_quantiles(cp: Any, sorted_rows: Any, counts: Any, probs: np.ndarray) -> Any:
    """``np.quantile(row[:count], probs)`` (linear) of every row of ``sorted_rows``, whose first ``counts[r]`` entries are the row's ascending sample; double, ``(rows, len(probs))``."""
    q = cp.asarray(probs, dtype=cp.float64)[None, :]  # Intentional float64: quantile positions and edges, as numpy computes them
    cnt = cp.maximum(counts, 1).astype(cp.float64)[:, None]
    virt = (cnt - 1.0) * q  # numpy's virtual index of the linear method, in the same operations: a position that is a whole number must come out as one on both sides
    lo = cp.floor(virt)
    gamma = virt - lo
    last = (cp.maximum(counts, 1) - 1)[:, None]
    lo_i = cp.clip(lo.astype(cp.int64), 0, last)
    hi_i = cp.minimum(lo_i + 1, last)
    a = cp.take_along_axis(sorted_rows, lo_i, axis=1).astype(cp.float64)  # Intentional float64: statistic / reference vector in double precision (compared exactly with its host twin)
    b = cp.take_along_axis(sorted_rows, hi_i, axis=1).astype(cp.float64)
    diff = b - a
    return cp.where(gamma >= 0.5, b - diff * (1.0 - gamma), a + diff * gamma)


def _pointwise(cp: Any, codes: Any, y: Any, n_codes: int, ky: int, mask: Any = None) -> Any:
    """Per-row pointwise MI of every row of ``codes`` (``(columns, rows)``, in ``[0, n_codes)``) with ``y`` (``(rows,)``); with ``mask`` the counts are those of the masked rows and the other
    rows get 0."""
    n_cols = codes.shape[0]
    dump = n_cols * n_codes * ky
    key = (cp.arange(n_cols, dtype=cp.int32)[:, None] * n_codes + codes) * ky + y[None, :]
    if mask is not None:
        key = cp.where(mask, key, dump)
    joint = cp.bincount(key.ravel(), minlength=dump + 1)[:dump].reshape(n_cols, n_codes, ky).astype(cp.float64)  # Intentional float64: joint counts and their log ratios
    px = joint.sum(axis=2, keepdims=True)
    py = joint.sum(axis=1, keepdims=True)
    n = joint.sum(axis=(1, 2), keepdims=True)
    den = px * py
    table = cp.where(joint > 0, cp.log(cp.where(joint > 0, joint * n, 1.0) / cp.where(den > 0, den, 1.0)), 0.0)
    return cp.concatenate([table.ravel(), cp.zeros(1, dtype=cp.float64)])[key]  # Intentional float64: double-precision accumulator / output buffer: sums and statistics keep full precision


def _block_stats(
    cp: Any, xr: Any, targets: "Sequence[Any]", y_odd: Any, ky: int, fold: Any, n_folds: int, n_bins: int, shrink: float, mi_bins: int, min_rows: int, few_cap: int, bulk_trim: float
) -> "tuple[Any, Any]":
    """``(usable, stats)`` of one device block ``xr`` (``(columns, rows)``): ``stats[t, c]`` = (raw MI, warp MI, standard error of the gain, bulk gain, its standard error) for target ``t``."""
    from ._device_sort_safe import device_sort

    n_cols, m = xr.shape
    finite = cp.isfinite(xr)
    nfin = finite.sum(axis=1)
    xs = device_sort(cp.where(finite, xr, cp.inf), axis=1)
    pos = cp.arange(1, m)
    distinct = ((xs[:, 1:] != xs[:, :-1]) & (pos[None, :] < nfin[:, None])).sum(axis=1) + (nfin > 0)
    usable = (nfin >= min_rows) & (distinct > few_cap)
    fit_edges = cp.ascontiguousarray(_row_quantiles(cp, xs, nfin, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    bulk_range = _row_quantiles(cp, xs, nfin, np.array([bulk_trim, 1.0 - bulk_trim])).astype(xr.dtype)  # the host compares the range in the column's type
    col_min = xs[:, :1]
    del xs, pos, distinct
    held_probs = np.linspace(0.0, 1.0, mi_bins + 1)[1:-1]
    n_edges = n_bins - 1
    odd = slice(1, None, 2)
    even = slice(0, None, 2)
    codes_of = _codes_kernel(cp)
    # the raw column's held-out codes: edges of the even rows (non-finite values take the column's smallest finite one), codes of the odd rows
    feat = cp.where(finite, xr, col_min)
    feat_even = cp.ascontiguousarray(feat[:, even])
    raw_edges = cp.ascontiguousarray(_row_quantiles(cp, device_sort(feat_even, axis=1), cp.full(n_cols, feat_even.shape[1], dtype=cp.int64), held_probs))
    feat_odd = cp.ascontiguousarray(feat[:, odd])
    cr = codes_of(feat_odd, raw_edges, np.int32(mi_bins - 1), np.int64(feat_odd.shape[1]))
    del feat, feat_even, feat_odd
    x_odd = xr[:, odd]
    bulk = finite[:, odd] & (x_odd >= bulk_range[:, :1]) & (x_odd <= bulk_range[:, 1:])
    n_bulk = bulk.sum(axis=1)
    use_bulk = n_bulk >= (min_rows // 2)
    n_bulk_f = cp.maximum(n_bulk, 1).astype(cp.float64)  # Intentional float64: statistic / reference vector in double precision (compared exactly with its host twin)
    mo = x_odd.shape[1]
    pm_r = _pointwise(cp, cr, y_odd, mi_bins, ky)
    pm_rb = _pointwise(cp, cr, y_odd, mi_bins, ky, bulk)
    mi_r = pm_r.mean(axis=1)
    cells = n_cols * n_folds * n_bins
    cell0 = (cp.arange(n_cols, dtype=cp.int32)[:, None] * n_folds + fold[None, :]) * n_bins
    out = []
    for target in targets:
        t_fin = cp.isfinite(target)
        b = codes_of(xr, fit_edges, np.int32(n_edges), np.int64(m))
        key = cp.where((b >= 0) & t_fin[None, :], cell0 + b, cells).ravel()
        del b
        sx = cp.bincount(key, weights=cp.where(finite, xr, 0).ravel(), minlength=cells + 1)[:cells].reshape(n_cols, n_folds, n_bins)
        sy = cp.bincount(key, weights=cp.broadcast_to(cp.where(t_fin, target, 0.0), (n_cols, m)).ravel(), minlength=cells + 1)[:cells].reshape(n_cols, n_folds, n_bins)
        cn = cp.bincount(key, minlength=cells + 1)[:cells].reshape(n_cols, n_folds, n_bins).astype(cp.float64)  # Intentional float64: exact counts in the table arithmetic
        del key
        fy = sy.sum(axis=2)
        fc = cn.sum(axis=2)
        ty = fy.sum(axis=1, keepdims=True)
        tc = fc.sum(axis=1, keepdims=True)
        mu_all = cp.where(tc > 0, ty / cp.where(tc > 0, tc, 1.0), 0.0)
        c_tr = tc - fc
        mu_f = cp.ascontiguousarray(cp.where(c_tr > 0, (ty - fy) / cp.where(c_tr > 0, c_tr, 1.0), mu_all))
        cnt_f = cp.ascontiguousarray(cn.sum(axis=1, keepdims=True) - cn)
        safe = cp.where(cnt_f > 0, cnt_f, 1.0)
        cx_f = cp.ascontiguousarray((sx.sum(axis=1, keepdims=True) - sx) / safe)
        cy_f = cp.ascontiguousarray((sy.sum(axis=1, keepdims=True) - sy + shrink * mu_f[:, :, None]) / (cnt_f + shrink))
        oof = _interp_kernel(cp)(xr, fold[None, :], cx_f, cy_f, cnt_f, mu_f, np.int32(n_bins), np.int32(n_folds), np.int64(m))
        oof_even = cp.ascontiguousarray(oof[:, even])
        warp_edges = cp.ascontiguousarray(_row_quantiles(cp, device_sort(oof_even, axis=1), cp.full(n_cols, oof_even.shape[1], dtype=cp.int64), held_probs))
        oof_odd = cp.ascontiguousarray(oof[:, odd])
        cw = codes_of(oof_odd, warp_edges, np.int32(mi_bins - 1), np.int64(mo))
        del oof, oof_even, oof_odd
        d = _pointwise(cp, cw, y_odd, mi_bins, ky) - pm_r
        mi_w = mi_r + d.mean(axis=1)
        se = d.std(axis=1) / np.sqrt(mo)
        db = _pointwise(cp, cw, y_odd, mi_bins, ky, bulk) - pm_rb
        mean_b = db.sum(axis=1) / n_bulk_f
        centred = cp.where(bulk, db - mean_b[:, None], 0.0)  # the differences are 0 outside the bulk: centre first, then square
        var_b = (centred * centred).sum(axis=1) / n_bulk_f
        bulk_gain = cp.where(use_bulk, mean_b, mi_w - mi_r)
        bulk_se = cp.where(use_bulk, cp.sqrt(var_b) / cp.sqrt(n_bulk_f), se)
        out.append(cp.stack([mi_r, mi_w, se, bulk_gain, bulk_se], axis=1))
        del d, db, cw
    return usable, cp.stack(out)


def warp_scores_device(
    columns: "Sequence[np.ndarray]",
    names: "Optional[Sequence[Any]]",
    rows: "Optional[np.ndarray]",
    targets: "Sequence[np.ndarray]",
    y_codes: np.ndarray,
    ky: int,
    fold: np.ndarray,
    *,
    n_folds: int,
    n_bins: int,
    shrink: float,
    mi_bins: int,
    min_rows: int,
    few_cap: int,
    bulk_trim: float,
) -> "Optional[tuple[np.ndarray, np.ndarray]]":
    """``(usable, stats)`` of the warp scan of ``columns`` (full host columns of one float dtype; ``rows`` picks the scan rows, ``None`` for all), or ``None`` when the host should scan.

    ``targets`` are the warp targets over the scan rows, ``y_codes`` the class codes of the scan rows (the MI is taken on the odd ones) and ``fold`` their fold ids. ``usable[c]`` is the host's
    skip rule (enough finite rows, more than ``few_cap`` distinct values); ``stats[t, c]`` = (raw MI, warp MI, standard error of the gain, bulk gain, its standard error), double."""
    if not columns or any(c.dtype != columns[0].dtype or c.dtype.kind != "f" for c in columns):
        return None
    try:
        import cupy as cp

        from ._device_blocks import run_column_blocks
        from ._device_sort_safe import device_room
        from ._fe_resident_operands import resident_operand

        m = len(y_codes)
        rows_d = None if rows is None else cp.asarray(rows)
        y_odd = resident_operand(np.ascontiguousarray(np.asarray(y_codes)[1::2], dtype=np.int32), "oof_warp_y_odd", dtype=np.int32)
        fold_d = resident_operand(np.ascontiguousarray(fold, dtype=np.int32), "oof_warp_fold", dtype=np.int32)
        targets_d = [cp.asarray(np.ascontiguousarray(t, dtype=np.float64)) for t in targets]  # Intentional float64: the warp target, as the host service takes it
        per_block = max(1, int(device_room(cp) * _BUDGET_FRACTION) // (m * _BYTES_PER_ELEMENT * max(1, len(targets) // 2 + 1)))

        def block_work(lo: int, hi: int) -> tuple:
            """The flags and statistics of columns ``lo .. hi``."""
            xr = cp.stack(
                [resident_operand(np.ascontiguousarray(columns[j]), ("xbasis_op", names[j] if names is not None else j), dtype=columns[j].dtype) for j in range(lo, hi)]
            )
            if rows_d is not None:
                xr = cp.ascontiguousarray(xr[:, rows_d])
            return _block_stats(cp, xr, targets_d, y_odd, int(ky), fold_d, int(n_folds), int(n_bins), float(shrink), int(mi_bins), int(min_rows), int(few_cap), float(bulk_trim))

        # column blocks sized to the room of the calling thread's pool; a block that still runs out of device memory is retried at half the width
        pieces = run_column_blocks(cp, len(columns), per_block, block_work)
        return cp.asnumpy(cp.concatenate([u for u, _s in pieces])), cp.asnumpy(cp.concatenate([st for _u, st in pieces], axis=1))
    except Exception as exc:
        log_throttle(logger, "oof_warp_gpu.failed", logging.WARNING, "device warp scan failed (%s: %s); the host pool scans the columns", type(exc).__name__, exc)
        return None


def _sweep_inputs(dims: dict) -> tuple:
    """A 40-column float32 frame of ``dims["n_samples"]`` rows (five columns carry a non-monotone effect on the target, one has missing values) and its target."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    xs = rng.standard_normal((n, 40)).astype(np.float32)
    y = (np.sin(3.0 * xs[:, 0]) + np.abs(xs[:, 1]) + xs[:, 2] ** 2 + np.cos(5.0 * xs[:, 3]) + 0.5 * xs[:, 4] + rng.standard_normal(n) > 1.5).astype(np.int64)
    xs[rng.random(n) < 0.02, 5] = np.nan
    return (xs, y)


def _scan(xs: np.ndarray, y: np.ndarray, device: bool) -> np.ndarray:
    """The scan's statistics per column, from the host pool or from the device."""
    import pandas as pd

    from ._oof_warp_fe import warp_candidates

    frame = pd.DataFrame(xs, columns=[f"c{j}" for j in range(xs.shape[1])], copy=False)
    cands = warp_candidates(frame, y, list(frame.columns), device=device)
    if device and not all(c.get("scored_on") == "device" for c in cands):
        raise RuntimeError("device warp scan unavailable")
    return np.array([[c["mi_raw"], c["mi_warp"], c["se"], c["bulk_gain"], c["bulk_se"]] for c in cands], dtype=np.float64)  # Intentional float64: statistics compared by the sweep


def _host(xs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The host pool's statistics, the reference of the sweep."""
    return _scan(xs, y, False)


def _device(xs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The device scan's statistics; an unavailable device raises so the sweep does not record a host fallback as a device time."""
    return _scan(xs, y, True)


_run_sweep = host_device_sweep(__name__, _host, _device, repeats=3, rtol=1e-9)


_SWEEP_ROWS = [5_000, 30_000, 100_000]
_SALT = 1


_fallback_choice = rows_fallback(__name__)


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="fe_oof_warp_scan_device",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_oof_warp_scan_device",
    )
except Exception as e:
    logger.warning("kernel_tuner spec build for fe_oof_warp_scan_device failed: %s", e)
    _SPEC = None


from ._ktc_fallback import ktc_device_chooser as _ktc_device_chooser

use_device = _ktc_device_chooser(__name__, "fe_oof_warp_scan_device", env="MLFRAME_OOF_WARP_DEVICE")


__all__ = ["warp_scores_device", "use_device"]
