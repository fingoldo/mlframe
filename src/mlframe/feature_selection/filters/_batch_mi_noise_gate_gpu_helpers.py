"""Device-side helpers of the cupy / numba.cuda noise-gate paths in ``batch_mi_noise_gate_gpu`` (carved out to keep that module under the 1k-LOC ceiling and
its public functions within the function-length gate)."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from mlframe.utils.log_throttle import log_throttle
from ._batch_mi_noise_gate_kernels import _cupy_bincount_known_size, _nb_cuda, _reduce_counts_on_device

logger = logging.getLogger("mlframe.feature_selection.filters.batch_mi_noise_gate_gpu")

_INT32_MAX = 2_147_483_647


def tiled_permutation_counts(cp: Any, d_base: Any, d_y_all: Any, n: int, K: int, total_size: int, P1: int, idx_dtype: Any) -> np.ndarray:
    """Integer joint-histogram counts of every y-vector, computed on the device in tiles over the permutation axis and downloaded one tile at a time."""
    # ---- 4GB tiling over the permutation axis. Each tile of ``rows`` y-vectors
    # materialises a (rows, n) int64 index array + a (rows, total_size) int64 count
    # array on device. Budget the bigger of the two (~ rows * max(n, total_size) * 8B)
    # against queried free memory (use a fraction to leave headroom for d_base etc.).
    try:
        free_b, _tot_b = cp.cuda.runtime.memGetInfo()
    except Exception as e:
        log_throttle(
            logger,
            "noise_gate.memgetinfo",
            logging.WARNING,
            "noise gate: cp.cuda.runtime.memGetInfo() failed (%s), tiling against a conservative 512MiB default",
            e,
        )
        free_b = 512 * 1024 * 1024  # conservative default
    budget = int(free_b * 0.35)
    # Per y-row device cost of a tile: the flat index array (n*K int64) + the
    # bincount output slot (total_size int64) + the y-codes (n int64). The n*K
    # index array dominates; keep the divisor honest so the 1050 Ti never OOMs.
    bytes_per_row = 8 * (n * K + total_size + n)
    rows_per_tile = max(1, budget // max(1, bytes_per_row))
    if rows_per_tile > P1:
        rows_per_tile = P1
    if rows_per_tile < P1:
        logger.info(
            "batch_mi_noise_gate cupy: tiling %d y-vectors into tiles of %d (free=%dMB, n=%d, total_size=%d)",
            P1,
            rows_per_tile,
            free_b // (1024 * 1024),
            n,
            total_size,
        )

    # Output integer counts for every y-vector, kept compact (P1, total_size) and
    # downloaded with ONE D2H PER TILE (O(num_tiles) D2H total, == 1 when it all fits).
    counts_all = np.empty((P1, total_size), dtype=np.int64)
    d_base_3d = d_base.reshape(1, n, K)  # broadcast base over the y-rows of a tile

    start = 0
    while start < P1:
        stop = min(start + rows_per_tile, P1)
        rows = stop - start
        # (rows, n) y-codes for this tile.
        d_y_tile = d_y_all[start:stop].reshape(rows, n, 1)  # (rows, n, 1) int64
        # per-(row, r, k) global index, then offset each row into its own
        # ``total_size`` slot so a SINGLE bincount fills all rows of the tile at once:
        #   slot = row*total_size + (base[r,k] + y[row,r])
        d_idx = (d_base_3d + d_y_tile).reshape(rows, n * K)  # (rows, n*K) idx_dtype
        # The row-spanning flat index reaches ``rows*total_size``; widen to int64 ONLY when that exceeds
        # int32 (else stay int32, halving the largest buffer too). idx_dtype already covers d_idx itself.
        flat_dtype = cp.int32 if (int(rows) * total_size) <= _INT32_MAX else cp.int64
        d_row_off = (cp.arange(rows, dtype=flat_dtype) * total_size).reshape(rows, 1)
        d_flat = (d_idx.astype(flat_dtype, copy=False) + d_row_off).reshape(-1)  # (rows*n*K,)
        # OPT-D: known-size bincount (skip cupy.bincount's two host-sync validations);
        # byte-identical counts (same kernel). ``rows*total_size`` is the exact size and
        # ``d_flat`` is non-negative by construction (offsets + non-negative codes).
        tile_counts = _cupy_bincount_known_size(d_flat, rows * total_size)
        # ONE D2H for the whole tile.
        # bench-attempt-rejected (2026-06-07): NARROW this D2H int64 -> int32 (the counts are
        # bounded by n << 2^31 so a cast is byte-identical) to halve the transferred bytes (Q2b).
        # Net LOSS: the on-device ``astype(int32)`` kernel launch + sync costs MORE than the saved
        # bytes on the scene tile shapes (0.42-0.99x; only the largest 1200col x 4perm tile wins
        # 1.43x). Crucially the ~16% "asnumpy" the scene sampler attributes here is NOT the
        # transfer - it is the main thread BLOCKED in asnumpy's implicit sync waiting on the
        # preceding async index-build + bincount GPU kernels (the real, irreducible cost on the
        # PCIe-bound 1050 Ti). Narrowing the transfer cannot shrink GPU compute, and the cast
        # adds more of it. (proto profiling/bench_cupy_d2h_narrow.py)
        counts_all[start:stop, :] = cp.asnumpy(tile_counts).reshape(rows, total_size)
        del d_y_tile, d_idx, d_row_off, d_flat, tile_counts
        start = stop
    return counts_all


def resident_codes_on_device(d_disc_resident: Any, disc_2d: np.ndarray, n: int, K: int) -> Any:
    """Device codes for the histogram kernels: the producer's resident array when it matches (no H2D), else a fresh upload in the native narrow dtype."""
    # H2D the codes in their NATIVE narrow dtype (int8/int16 - the bins are < 256/65536) instead of
    # up-casting to int32: nvprof showed the resident gate's wall is dominated by this (n, K) codes H2D,
    # not the (microsecond) kernels, so a 4x-smaller int8 transfer is the real lever. The hist kernel
    # reads disc_2d[r, k] as an index either way (numba compiles a per-dtype variant); counts unchanged.
    # RESIDENT-CODES HANDOFF (gated): when the producer kept the codes on-device, consume that array IN
    # PLACE (CUDA Array Interface) instead of re-uploading - skips the (n, K) codes H2D entirely. Requires
    # a matching shape + a narrow (itemsize<=2) dtype so the hist kernel reads it as an index exactly like
    # the host path; any mismatch (or None) falls back to the H2D so it can never change the counts.
    _disc_dt = disc_2d.dtype if disc_2d.dtype.itemsize <= 2 else np.int32
    d_disc = None
    if d_disc_resident is not None:
        try:
            _rshape = tuple(int(s) for s in d_disc_resident.shape)
            _rdt = np.dtype(d_disc_resident.dtype)
            if _rshape == (n, K) and _rdt.itemsize <= 2 and _rdt == np.dtype(_disc_dt):
                d_disc = d_disc_resident  # resident device codes -> NO H2D (the round-trip we eliminate)
        # best-effort: a fresh upload of the same codes gives the same result, only with an extra transfer
        except Exception as e:
            logger.debug("resident device-codes reuse check failed, falling back to a fresh H2D upload: %s", e)
            d_disc = None
    if d_disc is None:
        # disc_2d is the discretized FE-CANDIDATE matrix - a different chunk of engineered columns on
        # (almost) every dispatch, so it is NOT fit-constant like y/freqs_y; content-hashing it here would
        # pay an O(n*K) hash on every call for a near-guaranteed miss. Left as a plain per-call upload
        # (the resident-codes handoff above is the real lever for this operand).
        d_disc = _nb_cuda.to_device(np.ascontiguousarray(disc_2d, dtype=_disc_dt))
    return d_disc


def column_major_codes(d_disc: Any, use_shared: bool, hist_kernel_shared_cm: Any) -> Any:
    """Column-major (K, n) coalesced copy of the codes for the shared-memory histogram kernel; None when not applicable or on failure (row-major kernel is used)."""
    # COLUMN-MAJOR coalesced disc: the shared-mem hist kernel is LOAD-bound (CUDA-event
    # decomposition: per-row shared atomics ~3% of the kernel, ~12 GB/s << ~96 GB/s read peak), throttled
    # by the stride-K ``disc_2d[r, k]`` read (~1/8 coalesced). Reading the SAME values from a (K, n) C-order
    # buffer is fully coalesced -> 12.55x kernel-only (147 GB/s), 5.59x NET with a fresh transpose folded in
    # at n=100k/K=583/P=26 (this corrects the "shared-atomic-bound" rejection). The transpose is
    # paid ONCE (all P y-vectors share the one (K, n) disc). Counts are layout-invariant -> BIT-IDENTICAL
    # (verified maxdiff 0). Gated ON; ``MLFRAME_FE_GPU_HISTGATE_CM=0`` forces the row-major kernel; any
    # transpose/compile failure falls back to it too, so CPU / no-CUDA is byte-unchanged.
    _cm_on = os.environ.get("MLFRAME_FE_GPU_HISTGATE_CM", "1").strip().lower() in ("1", "true", "on", "yes")
    _d_disc_cm = None
    if use_shared and _cm_on and hist_kernel_shared_cm is not None:
        try:
            import cupy as _cp_cm
            from ._gpu_resident_select import transpose_codes_to_cm as _transpose_codes_to_cm

            # COALESCED tiled int-codes transpose: cp.ascontiguousarray(disc.T) was an
            # uncoalesced strided copy - nsys charged it cupy_copy__int8_int8 46 calls / 1776ms / 29.5% of
            # GPU-kernel time @F2 100k. The tiled kernel reads (n,K) + writes (K,n) coalesced, BIT-IDENTICAL
            # (same values, same (K,n) C-order layout -> counts unchanged); falls back to ascontiguousarray.
            _d_disc_cm = _transpose_codes_to_cm(_cp_cm.asarray(d_disc))  # (K, n) C-order, coalesced disc load
        except Exception:
            log_throttle(
                logger, "noise_gate.cm_transpose", logging.WARNING, "noise gate: the column-major transpose failed, using the row-major layout", exc_info=True
            )
            _d_disc_cm = None
    return _d_disc_cm


def launch_hist_and_reduce(
    use_shared: bool, d_disc_cm: Any, d_disc: Any, d_off: Any, d_nb: Any, d_y: Any, d_counts: Any, n: int, K: int, K_y: int, total_size: int, P: int,
    threads_per_block: int, sh_bytes: int, hist_kernel_shared_cm: Any, hist_kernel_shared: Any, hist_kernel_batched: Any, mi_kernel: Any, su_kernel: Any,
    use_su: bool, d_freq: Any, d_ref: Any, d_out: Any, freqs_y: np.ndarray,
) -> None:
    """Launch the batched histogram kernel (column-major shared, row-major shared, or global-atomic) and reduce the counts to MI / SU on the device into ``d_out``."""
    from mlframe.utils.warning_filters import silenced_once as _silenced_once

    try:
        from numba.core.errors import NumbaPerformanceWarning as _NbPerfWarn
    except ImportError:
        _NbPerfWarn = Warning
    with _silenced_once(_NbPerfWarn, r"numba\.cuda"):
        if use_shared and d_disc_cm is not None:
            hist_kernel_shared_cm[(K, P), threads_per_block, 0, sh_bytes](  # numba cuda kernel[grid, block] launch syntax, not real indexing
                d_disc_cm,
                d_off,
                d_nb,
                d_y,
                d_counts,
                n,
                K_y,
                total_size,
            )
        elif use_shared:
            hist_kernel_shared[(K, P), threads_per_block, 0, sh_bytes](  # numba cuda kernel[grid, block] launch syntax, not real indexing
                d_disc,
                d_off,
                d_nb,
                d_y,
                d_counts,
                n,
                K_y,
                total_size,
            )
        else:
            hist_kernel_batched[(K, P), threads_per_block](d_disc, d_off, d_y, d_counts, n, K_y, total_size)  # numba cuda kernel[grid, block] launch syntax, not real indexing
        _tot = K * P
        _blocks = (_tot + threads_per_block - 1) // threads_per_block
        _reduce_counts_on_device(mi_kernel, su_kernel if use_su else None, _blocks, threads_per_block, d_counts, d_off, d_nb, d_freq, n, K_y, d_ref, total_size, P, d_out, freqs_y)
