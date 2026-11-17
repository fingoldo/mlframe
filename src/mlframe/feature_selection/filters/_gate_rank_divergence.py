"""Gate rank of the conditional gate: the conditional divergence of the probe operands' MI across a ``c > median(c)`` split, for every candidate gate column in one pass.

``_conditional_gate_fe._rank_and_prune`` ranks each candidate gate column ``c`` by ``sum_a |MI(a; y | c > med) - MI(a; y | c <= med)|`` over a few probe operands ``a``. The loop it used
called the batched plug-in MI twice per gate column on the two row subsets, which re-sorts every probe column of every subset: 2 p calls, each a sort, a gather and a histogram.

The binning inside a subset needs no new sort. The rank binner orders a column by a stable sort, gives the ``m`` sorted rows of the subset the bins of their positions (the first ``m % nbins`` bins
one row larger) and lets a run of equal values take the bin of its first position. The sorted order of a subset is the global stable order restricted to its rows, so the position of a row inside the
subset is the number of subset rows before it in the global order. The probes are sorted once; for each gate column one walk over each probe's global order keeps a running position and a run start for
the rows above and for the rows below the median, adds the row to a ``(bin, class)`` histogram of its side and reads both plug-in MIs from the two histograms. The values are those of the rank
binner (same bins, same tie rule); only the summation order of the plug-in differs, at round-off.

``gate_divergence`` is the numba form: ``prange`` over chunks of 16 gate columns, each chunk walking a probe's global order once for all its columns (a row of the chunk is one cache line).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
from numba import njit, prange

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)


@njit(cache=True, nogil=True)
def _bin_size(b: int, m: int, nbins: int) -> int:
    """Rows in bin ``b`` of ``m`` rows split into ``nbins`` equi-frequency bins: the first ``m % nbins`` bins hold one row more."""
    return m // nbins + (1 if b < m % nbins else 0)


@njit(cache=True, nogil=True)
def _plug_in_mi(counts: np.ndarray, m: int) -> float:
    """Plug-in MI (nats) of a ``(bins, classes)`` count table of ``m`` rows."""
    nb, k = counts.shape
    mi = 0.0
    col = np.zeros(k, dtype=np.int64)
    for b in range(nb):
        for c in range(k):
            col[c] += counts[b, c]
    for b in range(nb):
        row = 0
        for c in range(k):
            row += counts[b, c]
        if row == 0:
            continue
        for c in range(k):
            v = counts[b, c]
            if v > 0:
                mi += (v / m) * np.log(v * m / (row * col[c]))
    return mi


_CHUNK = 16  # gate columns walked together: one row of a chunk is a 64-byte cache line of the float32 matrix


@njit(cache=True, nogil=True)
def _column_median(mat: np.ndarray, g: int) -> float:
    """Median of column ``g`` (a contiguous copy, then ``np.median``)."""
    col = np.empty(mat.shape[0], dtype=mat.dtype)
    for i in range(mat.shape[0]):
        col[i] = mat[i, g]
    return float(np.median(col))


@njit(cache=True, nogil=True)
def _upper_counts(mat: np.ndarray, g0: int, w: int, meds: np.ndarray, nbins: int) -> tuple:
    """``(m_hi, ok)`` of the gate columns ``g0 .. g0 + w``: the rows above each column's median (one pass over the rows, the chunk's columns of a row read together) and whether both halves hold at least ``nbins`` rows."""
    n = mat.shape[0]
    m_hi = np.zeros(w, dtype=np.int64)
    for i in range(n):
        for j in range(w):
            if mat[i, g0 + j] > meds[g0 + j]:
                m_hi[j] += 1
    ok = np.zeros(w, dtype=np.bool_)
    for j in range(w):
        ok[j] = m_hi[j] >= nbins and (n - m_hi[j]) >= nbins
    return m_hi, ok


@njit(cache=True, nogil=True)
def _finish_chunk(hi: np.ndarray, lo: np.ndarray, m_hi: np.ndarray, n: int, ok: np.ndarray, total: np.ndarray) -> None:
    """Add the divergence of the walk just finished (one probe) to ``total`` for every usable gate column of the chunk."""
    for j in range(ok.shape[0]):
        if ok[j]:
            total[j] += abs(_plug_in_mi(hi[j], m_hi[j]) - _plug_in_mi(lo[j], n - m_hi[j]))


@njit(cache=True, nogil=True)
def _walk_chunk(mat: np.ndarray, g0: int, g1: int, meds: np.ndarray, order: np.ndarray, probe_idx: np.ndarray, y: np.ndarray, n_classes: int, nbins: int, out: np.ndarray) -> None:
    """Divergence of the gate columns ``g0:g1``: one walk per probe over its global order, every row updating the running positions and histograms of all the chunk's columns."""
    n = mat.shape[0]
    w = g1 - g0
    m_hi, ok = _upper_counts(mat, g0, w, meds, nbins)
    hi = np.zeros((w, nbins, n_classes), dtype=np.int64)
    lo = np.zeros((w, nbins, n_classes), dtype=np.int64)
    r_hi = np.zeros(w, dtype=np.int64)
    r_lo = np.zeros(w, dtype=np.int64)
    gs_hi = np.zeros(w, dtype=np.int64)
    gs_lo = np.zeros(w, dtype=np.int64)
    prev_hi = np.zeros(w, dtype=np.float64)
    prev_lo = np.zeros(w, dtype=np.float64)
    bin_hi = np.zeros(w, dtype=np.int64)  # bin of the run start, advanced as the run start moves up the positions (no division per row)
    bin_lo = np.zeros(w, dtype=np.int64)
    end_hi = np.zeros(w, dtype=np.int64)  # first position past the current bin
    end_lo = np.zeros(w, dtype=np.int64)
    total = np.zeros(w, dtype=np.float64)
    for a in range(order.shape[0]):
        hi[:, :, :] = 0
        lo[:, :, :] = 0
        r_hi[:] = 0
        r_lo[:] = 0
        gs_hi[:] = 0
        gs_lo[:] = 0
        bin_hi[:] = 0
        bin_lo[:] = 0
        for j in range(w):
            end_hi[j] = _bin_size(0, m_hi[j], nbins)
            end_lo[j] = _bin_size(0, n - m_hi[j], nbins)
        pc = probe_idx[a]
        for t in range(n):
            i = order[a, t]
            v = mat[i, pc]
            yi = y[i]
            for j in range(w):
                if not ok[j]:
                    continue
                if mat[i, g0 + j] > meds[g0 + j]:
                    if r_hi[j] > 0 and v != prev_hi[j]:
                        gs_hi[j] = r_hi[j]
                        while gs_hi[j] >= end_hi[j]:
                            bin_hi[j] += 1
                            end_hi[j] += _bin_size(bin_hi[j], m_hi[j], nbins)
                    prev_hi[j] = v
                    hi[j, bin_hi[j], yi] += 1
                    r_hi[j] += 1
                else:
                    if r_lo[j] > 0 and v != prev_lo[j]:
                        gs_lo[j] = r_lo[j]
                        while gs_lo[j] >= end_lo[j]:
                            bin_lo[j] += 1
                            end_lo[j] += _bin_size(bin_lo[j], n - m_hi[j], nbins)
                    prev_lo[j] = v
                    lo[j, bin_lo[j], yi] += 1
                    r_lo[j] += 1
        _finish_chunk(hi, lo, m_hi, n, ok, total)
    for j in range(w):
        out[g0 + j] = total[j] if ok[j] else 0.0


@njit(cache=True, parallel=True, nogil=True)
def _divergence_njit(mat: np.ndarray, order: np.ndarray, probe_idx: np.ndarray, y: np.ndarray, n_classes: int, nbins: int) -> np.ndarray:
    """Divergence of every column of ``mat`` ``(n, p)`` as a gate: medians per column, then the chunked walks in parallel."""
    p = mat.shape[1]
    meds = np.empty(p, dtype=mat.dtype)
    for g in prange(p):
        meds[g] = _column_median(mat, g)
    out = np.zeros(p, dtype=np.float64)  # Intentional float64: statistic vector, ranked by argsort
    n_chunks = (p + _CHUNK - 1) // _CHUNK
    for c in prange(n_chunks):
        g0 = c * _CHUNK
        g1 = min(p, g0 + _CHUNK)
        _walk_chunk(mat, g0, g1, meds, order, probe_idx, y, n_classes, nbins, out)
    return out


def probe_orders(probes: np.ndarray) -> np.ndarray:
    """``(n_probe, n)`` stable argsort of each probe column of ``probes`` ``(n, n_probe)`` (the global orders the subset walks restrict)."""
    return np.ascontiguousarray(np.stack([np.argsort(probes[:, a], kind="stable") for a in range(probes.shape[1])]).astype(np.int64))


def gate_divergence(mat: np.ndarray, y: np.ndarray, probe_idx: "list[int]", nbins: int) -> Optional[np.ndarray]:
    """Divergence of every column of the finite ``mat`` ``(n, p)`` as a gate, against the probe columns ``probe_idx`` and the class codes ``y``; ``None`` when ``mat`` is not finite
    (the caller keeps the per-column path, whose partial-NaN handling this walk does not reproduce)."""
    from mlframe._numba_parallel_guard import parallel_kernel_entry

    mat = np.ascontiguousarray(mat)
    if not np.isfinite(mat).all():
        return None
    order = probe_orders(mat[:, probe_idx])
    classes, y_dense = np.unique(np.asarray(y), return_inverse=True)
    with parallel_kernel_entry():
        out: np.ndarray = _divergence_njit(mat, order, np.asarray(probe_idx, dtype=np.int64), np.ascontiguousarray(y_dense, dtype=np.int64), int(classes.size), int(nbins))
    return out


#: Bytes of device temporaries per (row, gate column) of one column block: the gathered mask, two prefix counts and their run-start gathers, the key, its flat copy (about 28 bytes).
_DEVICE_BYTES_PER_CELL = 28
_DEVICE_BLOCK_BYTES = 256 * 1024 * 1024
_KEY_KERNELS: dict = {}


def _key_kernel(cp: Any) -> Any:
    """Fused elementwise pass: the flat histogram key ``(side * nbins + bin) * classes + class`` of every (row, gate column), the bin read from the run-start position inside the row's own half."""
    ker = _KEY_KERNELS.get("key")
    if ker is None:
        ker = _KEY_KERNELS["key"] = cp.ElementwiseKernel(
            "bool h, int32 gh, int32 gl, int64 mh, int32 y, int64 n, int32 nbins, int32 classes",
            "int32 key",
            """
            const long long m = h ? mh : (n - mh);
            const long long g = h ? gh : gl;
            const long long base = (m / nbins) > 0 ? (m / nbins) : 1;
            const long long rem = m % nbins;
            const long long thr = rem * (base + 1);
            const long long b = g < thr ? g / (base + 1) : rem + (g - thr) / base;
            key = (int)((((h ? 0 : 1) * nbins) + b) * classes + y);
            """,
            "gate_divergence_key",
        )
    return ker


def _device_block(cp: Any, mat: Any, meds: Any, y: Any, n_classes: int, probe_idx: "list[int]", nbins: int) -> Any:
    """Divergence of the gate columns of the device block ``mat`` ``(n, w)``: per probe one stable argsort, the run starts, a prefix count of the upper-half mask in that order, the flat keys and one
    bincount for all columns' two histograms, then the plug-in MIs of every (column, half) at once."""
    n, w = mat.shape
    hi_all = mat > meds[None, :]
    m_hi = hi_all.sum(axis=0).astype(cp.int64)
    ok = (m_hi >= nbins) & ((n - m_hi) >= nbins)
    out = cp.zeros(w, dtype=cp.float64)  # Intentional float64: statistic vector, ranked by argsort
    cells = 2 * nbins * n_classes
    offsets = (cp.arange(w, dtype=cp.int32) * np.int32(cells))[None, :]
    key_ker = _key_kernel(cp)
    for a in probe_idx:
        col = mat[:, a]
        order = cp.argsort(col, kind="stable")
        sorted_vals = col[order]
        new_run = cp.ones(n, dtype=cp.bool_)
        new_run[1:] = sorted_vals[1:] != sorted_vals[:-1]
        run_start = cp.nonzero(new_run)[0].astype(cp.int32)[cp.cumsum(new_run, dtype=cp.int32) - 1]  # first position of each row's run of equal values
        hs = hi_all[order]
        pre_hi = cp.cumsum(hs, axis=0, dtype=cp.int32) - hs  # exclusive count of upper-half rows before each position
        gs_hi = pre_hi[run_start]
        gs_lo = run_start[:, None] - gs_hi  # lower-half rows before the run start
        key = key_ker(hs, gs_hi, gs_lo, m_hi[None, :], y[order][:, None], np.int64(n), np.int32(nbins), np.int32(n_classes)) + offsets
        counts = cp.bincount(key.ravel(), minlength=w * cells).reshape(w, 2, nbins, n_classes).astype(cp.float64)  # Intentional float64: plug-in sums
        m_side = counts.sum(axis=(2, 3))
        row = counts.sum(axis=3, keepdims=True)
        colsum = counts.sum(axis=2, keepdims=True)
        safe = counts > 0
        ratio = cp.where(safe, counts * m_side[:, :, None, None] / cp.where(safe, row * colsum, 1.0), 1.0)
        mi = (cp.where(safe, counts, 0.0) * cp.log(ratio)).sum(axis=(2, 3)) / cp.maximum(m_side, 1.0)
        out += cp.abs(mi[:, 0] - mi[:, 1])
    return cp.where(ok, out, 0.0)


def gate_divergence_device(cp: Any, mat: Any, y: Any, probe_idx: "list[int]", nbins: int, medians: Any = None) -> Optional[np.ndarray]:
    """Device form of :func:`gate_divergence` for the resident finite matrix ``mat`` ``(n, p)`` and class codes ``y`` (a device vector); the columns go through in column blocks of
    ``_DEVICE_BLOCK_BYTES`` of temporaries. ``medians`` (device, ``(p,)``) when the caller already has them. A host ``(p,)`` float64 vector, or ``None`` on any device fault."""
    try:
        n, p = int(mat.shape[0]), int(mat.shape[1])
        classes, y_dense = cp.unique(y, return_inverse=True)
        y32 = y_dense.astype(cp.int32).ravel()
        n_classes = int(classes.size)
        width = max(1, _DEVICE_BLOCK_BYTES // max(1, n * _DEVICE_BYTES_PER_CELL))
        meds = medians if medians is not None else cp.median(mat, axis=0)
        parts = []
        for lo in range(0, p, width):
            hi = min(p, lo + width)
            # the probe columns must be inside every block: they are gathered from the full matrix by index, so each block carries them as its first columns
            block = mat[:, lo:hi]
            probe_cols = mat[:, list(probe_idx)]
            full = cp.concatenate([probe_cols, block], axis=1)
            meds_full = cp.concatenate([meds[list(probe_idx)], meds[lo:hi]])
            res = _device_block(cp, full, meds_full, y32, n_classes, list(range(len(probe_idx))), nbins)
            parts.append(res[len(probe_idx) :])
        return np.asarray(cp.asnumpy(cp.concatenate(parts)), dtype=np.float64)  # Intentional float64: statistic vector
    except Exception as exc:
        log_throttle(logger, "gate_rank_divergence_device_gate_divergence_unavailable", logging.WARNING, "device gate divergence unavailable (%s: %s)", type(exc).__name__, exc)
        return None


# ----------------------------------------------------------------------------------------------------------------------
# Host or device walk (kernel-tuning entry ``gate_rank_divergence_backend``)
# ----------------------------------------------------------------------------------------------------------------------
_SWEEP_ROWS = [5_000, 20_000, 100_000, 300_000]
_SWEEP_COLS = 100
_SALT = 1
DEVICE_FALLBACK_ROWS = 10_000_000  # the host walk is as fast or faster at every measured size on the measured card: the device only where a sweep says so


def _sweep_inputs(dims: dict) -> tuple:
    """``(mat, y)``: ``_SWEEP_COLS`` float32 columns of ``dims["n_samples"]`` rows and a 3-class target the first columns drive."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    mat = rng.normal(size=(n, _SWEEP_COLS)).astype(np.float32)
    y = ((mat[:, 0] > 0).astype(np.int64) + (mat[:, 1] > 0.3) + 2 * (mat[:, 2] < -0.5)) % 3
    return (mat, y.astype(np.int64))


def _sweep_host(mat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The host walk on the sweep inputs."""
    got = gate_divergence(mat, y, [0, 1, 2], 10)
    if got is None:
        raise RuntimeError("host gate walk declined")
    return got


def _sweep_device(mat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The device walk on the sweep inputs, host in and host out (the upload is part of the cost: a non-STRICT caller holds the matrix on the host)."""
    import cupy as cp

    got = gate_divergence_device(cp, cp.asarray(mat), cp.asarray(y), [0, 1, 2], 10)
    if got is None:
        raise RuntimeError("device gate walk declined")
    return got


try:
    from ._ktc_fallback import host_device_sweep, rows_fallback

    _fallback_choice = rows_fallback(__name__)
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="gate_rank_divergence_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-9, atol=1e-9),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="gate_rank_divergence_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the host walk decides
    logger.warning("kernel_tuner spec build for gate_rank_divergence_backend failed: %s", e)
    _SPEC = None


def gate_divergence_on_device(n_rows: int) -> bool:
    """Whether a host-resident gate matrix of ``n_rows`` rows is walked on the device: ``MLFRAME_GATE_DIVERGENCE_DEVICE=0|1`` forces a side, else the per-host kernel-tuning cache, else the row mark.

    STRICT-GPU keeps the resident matrix on the device and does not consult this."""
    forced = os.environ.get("MLFRAME_GATE_DIVERGENCE_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("gate_rank_divergence_backend: KTC choose() failed, using the row-count fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


__all__: Any = ["gate_divergence", "gate_divergence_device", "gate_divergence_on_device", "probe_orders"]
