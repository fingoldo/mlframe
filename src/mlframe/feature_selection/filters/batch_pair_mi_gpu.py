"""GPU variants of :func:`batch_pair_mi_prange` + a size-aware dispatcher.

Three backends are exposed:

* ``batch_pair_mi_njit_prange`` -- re-export of the CPU njit kernel from
  ``info_theory``. The reference implementation; numerical baseline.
* ``batch_pair_mi_cuda`` -- ``numba.cuda`` JIT kernel. One CUDA block per pair,
  threads inside the block share a joint-class histogram via shared memory
  before a single thread runs the MI reduction.
* ``batch_pair_mi_cupy`` -- pure CuPy implementation. One vectorised sweep
  per pair using ``cupy.bincount`` for the joint histogram + a manual MI
  reduction. Trades GPU-occupancy for code simplicity; benefits more on
  high-bin combinations where CuPy's elementwise kernels saturate the SMs.

The dispatcher (:func:`dispatch_batch_pair_mi`) picks the fastest backend
given the input shape and the available hardware. Crossover thresholds
were measured on a single benchmark machine (CPU: 4 physical cores, GPU:
GTX 1050 Ti compute_capability=6.1, 4GB VRAM); callers can override via the
``force_backend`` knob to lock a specific implementation.

Numerical equivalence vs the ``merge_vars + compute_mi_from_classes`` legacy
path is verified by ``tests/feature_selection/test_batch_pair_mi_prange.py``
(CPU) and ``test_batch_pair_mi_gpu.py`` (GPU variants when CUDA is
available; auto-skip otherwise).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# Re-export CPU baseline for callers who want a single import point.
from .info_theory import batch_pair_mi_prange as batch_pair_mi_njit_prange

# Optional GPU deps. The dispatcher gracefully falls back to the CPU kernel
# when either is missing.
try:
    from numba import cuda as _nb_cuda
    _CUDA_AVAIL = bool(getattr(_nb_cuda, "is_available", lambda: False)())
except Exception:
    _nb_cuda = None
    _CUDA_AVAIL = False

try:
    import cupy as _cp
    _CUPY_AVAIL = True
except Exception:
    _cp = None
    _CUPY_AVAIL = False


# ---------------------------------------------------------------------------
# numba.cuda variant
# ---------------------------------------------------------------------------

# Per-pair CUDA kernel: one block per pair, threads in the block populate a
# shared-memory joint-class histogram, then one thread reduces it to MI.
#
# Compute capability 6.x exposes a 48 KB per-block static-shared-memory budget
# (96 KB physical, but 48 KB is the default carve-out without opt-in). Sizing
# the histogram as int64[MAX_JOINT_BINS][MAX_Y_BINS] + freqs_x[MAX_JOINT_BINS]
# the budget is:
#     MAX_JOINT_BINS * MAX_Y_BINS * 8  +  MAX_JOINT_BINS * 8
# With (128, 16) -> 128*16*8 + 128*8 = 16384 + 1024 = 17 KB  (fits).
# With (256, 32) -> 256*32*8 + 256*8 = 65536 + 2048 = 64 KB  (overflows; the
# first revision tripped CUDA_ERROR_INVALID_SOURCE / "uses too much shared data
# (0x10800 bytes, 0xc000 max)" on a GTX 1050 Ti).
# MAX_JOINT_BINS=128 covers nbins[a]*nbins[b] up to 128 (e.g. 11x11 = 121),
# which is generous for the mlframe MI-screen axis where nbins is typically
# 5-10. MAX_Y_BINS=16 supports up to 16-class multiclass targets. Inputs
# exceeding either bound are rejected by the dispatcher and routed to the
# CPU njit kernel where the histogram lives in main memory.

MAX_JOINT_BINS_CUDA = 128  # covers nbins<=11 per column (11x11 = 121)
MAX_Y_BINS_CUDA = 16       # multiclass up to 16 levels


def _cuda_kernel_factory():
    """Build the CUDA kernel lazily so importing this module on a CPU-only
    host doesn't trigger numba.cuda's CUDA driver lookup (which can raise on
    machines without the CUDA toolkit installed even when numba is present).
    """
    if not _CUDA_AVAIL:
        return None

    @_nb_cuda.jit
    def _kernel(
        factors_data,   # (n_samples, n_features) int32
        pair_a,         # (n_pairs,) int64
        pair_b,         # (n_pairs,) int64
        nbins,          # (n_features,) int32
        classes_y,      # (n_samples,) int32
        freqs_y,        # (n_classes_y,) float64
        mi_out,         # (n_pairs,) float64
        n_samples,
        n_classes_y,
    ):
        # One block per pair. Threads within the block stride over n_samples
        # to populate a shared-memory joint-class histogram. Then thread 0
        # reduces to MI.
        p = _nb_cuda.blockIdx.x
        if p >= pair_a.shape[0]:
            return

        a = pair_a[p]
        b = pair_b[p]
        nb_a = nbins[a]
        nb_b = nbins[b]
        joint_card = nb_a * nb_b

        # Shared-memory joint histogram + freqs_x. Sized at compile time to
        # the worst-case bound; actual usage may be smaller.
        # Layout: joint_counts[joint_card * n_classes_y] then freqs_x[joint_card].
        sm_hist = _nb_cuda.shared.array(
            shape=(MAX_JOINT_BINS_CUDA, MAX_Y_BINS_CUDA), dtype=np.int64,
        )
        sm_fx = _nb_cuda.shared.array(shape=(MAX_JOINT_BINS_CUDA,), dtype=np.int64)

        tid = _nb_cuda.threadIdx.x
        nthreads = _nb_cuda.blockDim.x

        # Zero the shared-memory histogram.
        for i in range(tid, joint_card, nthreads):
            sm_fx[i] = 0
            for j in range(n_classes_y):
                sm_hist[i, j] = 0
        _nb_cuda.syncthreads()

        # Populate histogram: each thread strides over n_samples.
        for i in range(tid, n_samples, nthreads):
            va = factors_data[i, a]
            vb = factors_data[i, b]
            cls_x = va * nb_b + vb
            cls_y = classes_y[i]
            _nb_cuda.atomic.add(sm_hist, (cls_x, cls_y), 1)
            _nb_cuda.atomic.add(sm_fx, cls_x, 1)
        _nb_cuda.syncthreads()

        # Reduce on thread 0. MI = sum_{i,j} jf * log(jf / (px * py)).
        if tid == 0:
            total = 0.0
            inv_n = 1.0 / n_samples
            for i in range(joint_card):
                fx = sm_fx[i]
                if fx == 0:
                    continue
                prob_x = fx * inv_n
                for j in range(n_classes_y):
                    jc = sm_hist[i, j]
                    if jc == 0:
                        continue
                    jf = jc * inv_n
                    prob_y = freqs_y[j]
                    if prob_y > 0.0:
                        total += jf * math.log(jf / (prob_x * prob_y))
            mi_out[p] = total

    return _kernel


_CUDA_KERNEL: Any = None  # lazy-bound on first call


def batch_pair_mi_cuda(
    factors_data: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    threads_per_block: int = 128,
) -> np.ndarray:
    """numba.cuda batch pair-MI kernel. Mirrors :func:`batch_pair_mi_njit_prange`.

    Raises ``RuntimeError`` if CUDA is not available. The dispatcher routes
    around this; direct callers should gate on :data:`_CUDA_AVAIL`.

    Shared-memory budget caps (sized for cc 6.x 48 KB per-block budget):
      * Max joint cardinality (``nbins[a] * nbins[b]``): ``MAX_JOINT_BINS_CUDA = 128``
      * Max target cardinality (``n_classes_y``): ``MAX_Y_BINS_CUDA = 16``
    Callers exceeding either limit must fall back to the CPU kernel.

    Preconditions enforced host-side (raise ``ValueError`` on violation):
      * ``factors_data >= 0`` everywhere -- negative codes would underflow
        the joint-index arithmetic ``va * nb_b + vb`` and write to undefined
        shared-mem cells (numba.cuda has no array-bounds checks in release
        mode, so this is silent corruption rather than a crash).
      * ``classes_y[i] < n_classes_y`` for every i -- out-of-range class ids
        would write past the shared histogram into ``sm_fx``.
      * ``nbins[a] >= 1`` for every column referenced by ``pair_a`` /
        ``pair_b`` -- a zero-cardinality column collapses ``joint_card`` to
        zero, returning a meaningless MI=0.
    """
    global _CUDA_KERNEL
    if not _CUDA_AVAIL:
        raise RuntimeError("numba.cuda is not available on this host")
    if _CUDA_KERNEL is None:
        _CUDA_KERNEL = _cuda_kernel_factory()
        if _CUDA_KERNEL is None:
            raise RuntimeError("numba.cuda kernel factory failed to build")

    n_pairs = int(pair_a.shape[0])
    if n_pairs == 0:
        # Critic-flagged P0: ``max(...)`` over the empty pair zip raises
        # ``ValueError`` before any device work; short-circuit cleanly.
        return np.empty(0, dtype=np.float64)

    # Shape guard
    n_classes_y = int(freqs_y.shape[0])
    if n_classes_y > MAX_Y_BINS_CUDA:
        raise ValueError(
            f"n_classes_y={n_classes_y} exceeds CUDA shared-memory budget "
            f"MAX_Y_BINS_CUDA={MAX_Y_BINS_CUDA}; use the CPU kernel instead",
        )
    # Joint-card + min-cardinality guard: check the largest and smallest pair.
    max_joint = 0
    min_nb = None
    for a, b in zip(pair_a, pair_b):
        nb_a = int(nbins[a])
        nb_b = int(nbins[b])
        if nb_a < 1 or nb_b < 1:
            raise ValueError(
                f"degenerate pair ({int(a)}, {int(b)}): nbins=({nb_a}, {nb_b}); "
                f"at least one column has zero cardinality (skip the pair host-side)",
            )
        if min_nb is None or min(nb_a, nb_b) < min_nb:
            min_nb = min(nb_a, nb_b)
        if nb_a * nb_b > max_joint:
            max_joint = nb_a * nb_b
    if max_joint > MAX_JOINT_BINS_CUDA:
        raise ValueError(
            f"max joint cardinality nbins[a]*nbins[b]={max_joint} exceeds "
            f"CUDA shared-memory budget MAX_JOINT_BINS_CUDA={MAX_JOINT_BINS_CUDA}; "
            f"use the CPU kernel instead",
        )

    # Critic-flagged P0: out-of-range classes_y or negative factors_data writes
    # past shared memory. Validate on host (cheap: one min/max sweep) so the
    # device kernel can stay branch-free in the hot loop.
    if classes_y.size > 0:
        cy_max = int(classes_y.max())
        cy_min = int(classes_y.min())
        if cy_max >= n_classes_y or cy_min < 0:
            raise ValueError(
                f"classes_y values must be in [0, n_classes_y={n_classes_y}); "
                f"got [min={cy_min}, max={cy_max}]",
            )
    if factors_data.size > 0:
        fd_min = int(factors_data.min())
        if fd_min < 0:
            raise ValueError(
                f"factors_data must be non-negative; got min={fd_min} "
                f"(merge_vars output should be >= 0 by contract)",
            )

    n_samples = int(factors_data.shape[0])

    # Move inputs to device. ``ascontiguousarray`` guards against non-C-layout
    # numpy arrays that the harness occasionally produces (e.g. .T views).
    d_data = _nb_cuda.to_device(np.ascontiguousarray(factors_data, dtype=np.int32))
    d_pa = _nb_cuda.to_device(np.ascontiguousarray(pair_a, dtype=np.int64))
    d_pb = _nb_cuda.to_device(np.ascontiguousarray(pair_b, dtype=np.int64))
    d_nb = _nb_cuda.to_device(np.ascontiguousarray(nbins, dtype=np.int32))
    d_cy = _nb_cuda.to_device(np.ascontiguousarray(classes_y, dtype=np.int32))
    d_fy = _nb_cuda.to_device(np.ascontiguousarray(freqs_y, dtype=np.float64))
    d_out = _nb_cuda.device_array(n_pairs, dtype=np.float64)

    _CUDA_KERNEL[n_pairs, threads_per_block](
        d_data, d_pa, d_pb, d_nb, d_cy, d_fy, d_out, n_samples, n_classes_y,
    )
    return d_out.copy_to_host()


# ---------------------------------------------------------------------------
# cupy variant
# ---------------------------------------------------------------------------


def batch_pair_mi_cupy(
    factors_data: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
) -> np.ndarray:
    """CuPy batch pair-MI kernel. Mirrors :func:`batch_pair_mi_njit_prange`.

    Implementation note: CuPy doesn't expose a 2D-histogram primitive, so the
    pair joint code (``cls_x * n_classes_y + cls_y``) is collapsed to a 1D
    bincount per pair. The pair loop runs in Python; each iteration is one
    CuPy kernel launch + one ``bincount`` + a small MI reduction. For low
    pair counts (n_pairs <~ 50) this is dispatch-dominated; the CUDA kernel
    is generally faster. For large pair counts the per-pair work amortises.
    """
    if not _CUPY_AVAIL:
        raise RuntimeError("cupy is not available on this host")
    cp = _cp  # local alias

    if pair_a.shape[0] == 0:
        return np.empty(0, dtype=np.float64)

    d_data = cp.asarray(factors_data, dtype=cp.int32)
    d_classes_y = cp.asarray(classes_y, dtype=cp.int32)
    d_freqs_y = cp.asarray(freqs_y, dtype=cp.float64)
    nb_arr = np.asarray(nbins, dtype=np.int32)
    pa_arr = np.asarray(pair_a, dtype=np.int64)
    pb_arr = np.asarray(pair_b, dtype=np.int64)

    n_samples = int(factors_data.shape[0])
    n_pairs = int(pa_arr.shape[0])
    n_classes_y = int(d_freqs_y.shape[0])
    inv_n = 1.0 / n_samples
    out_host = np.empty(n_pairs, dtype=np.float64)

    for p in range(n_pairs):
        a = int(pa_arr[p])
        b = int(pb_arr[p])
        nb_a = int(nb_arr[a])
        nb_b = int(nb_arr[b])
        joint_card = nb_a * nb_b

        va = d_data[:, a]
        vb = d_data[:, b]
        cls_x = va * nb_b + vb  # 1-D joint code
        joint_idx = cls_x * n_classes_y + d_classes_y  # 1-D flat index
        joint_counts_flat = cp.bincount(
            joint_idx, minlength=joint_card * n_classes_y,
        )[: joint_card * n_classes_y]
        joint_counts = joint_counts_flat.reshape(joint_card, n_classes_y).astype(cp.float64)
        joint_freqs = joint_counts * inv_n
        fx = joint_freqs.sum(axis=1)
        # MI reduction: sum where joint_freqs > 0
        # prob_x[i] = fx[i], prob_y[j] = freqs_y[j]; jf = joint_freqs[i, j]
        # Vectorised: log(jf / (fx[:, None] * freqs_y[None, :])) where jf > 0
        denom = fx[:, None] * d_freqs_y[None, :]
        # Guard zeros to avoid log(0). cupy's where + log are safe under masked.
        mask = (joint_freqs > 0) & (denom > 0)
        ratio = cp.where(mask, joint_freqs / cp.where(denom > 0, denom, 1.0), 1.0)
        log_term = cp.where(mask, cp.log(ratio), 0.0)
        mi = (joint_freqs * log_term).sum()
        out_host[p] = float(mi.get())

    return out_host


# ---------------------------------------------------------------------------
# Size-aware dispatcher
# ---------------------------------------------------------------------------


# Crossover thresholds derived from bench_batch_pair_mi_prange.py on a GTX 1050 Ti
# (cc 6.1, 768 CUDA cores, 4 GB VRAM) vs an i7 4-physical-core CPU. Measured points:
#
#   | n_rows  x  n_pairs | layer2_prange | cuda    | cuda/cpu_njit |
#   |--------------------|---------------|---------|----------------|
#   |  200 000  x   28   |   8.56 x      |  8.00x  |  0.93x (CPU)   |
#   |  500 000  x  120   |  13.94 x      | 29.88x  |  2.14x (CUDA)  |
#   | 1 000 000  x   66   |  11.47 x      | 20.19x  |  1.76x (CUDA)  |
#
# CUDA pulls ahead of the CPU njit kernel around n_rows ~= 400k (with enough
# pairs to amortise the per-block fixed cost). Below that, the CPU prange
# kernel keeps the GTX-1050-Ti grid under-occupied (28 pairs => 28 blocks of 128
# threads = 3584 active threads on a card that wants 6-10x more for full
# occupancy), and the H2D / D2H transfer overhead dominates.
#
# CuPy never beat numba.cuda in any of the measured points (always 2-5x SLOWER)
# because each pair dispatches one Python-side bincount kernel; the per-launch
# overhead defeats the per-pair work. It only becomes competitive on very
# large pair counts (>200) where the launch cost amortises -- those thresholds
# stay defensive.
#
# Callers can override the heuristic via ``dispatch_batch_pair_mi(..., force_backend=)``.
CUDA_MIN_ROWS = 400_000
CUDA_MIN_PAIRS = 16
CUPY_MIN_ROWS = 5_000_000
CUPY_MIN_PAIRS = 200


def dispatch_batch_pair_mi(
    factors_data: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    force_backend: str | None = None,
) -> tuple[np.ndarray, str]:
    """Pick the fastest backend by (n_samples, n_pairs) heuristic and run it.

    Returns ``(mi_array, backend_name)`` so callers can log which path fired.
    ``force_backend in {"njit", "cuda", "cupy"}`` overrides the heuristic.
    Falls back to "njit" when the requested GPU backend is unavailable.
    """
    n_samples = int(factors_data.shape[0])
    n_pairs = int(pair_a.shape[0])

    # Explicit override
    if force_backend is not None:
        force_backend = force_backend.lower()
        if force_backend == "cuda" and _CUDA_AVAIL:
            return batch_pair_mi_cuda(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "cuda"
        if force_backend == "cupy" and _CUPY_AVAIL:
            return batch_pair_mi_cupy(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "cupy"
        return batch_pair_mi_njit_prange(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "njit"

    # Heuristic
    if (
        _CUPY_AVAIL
        and n_samples >= CUPY_MIN_ROWS
        and n_pairs >= CUPY_MIN_PAIRS
    ):
        try:
            return batch_pair_mi_cupy(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "cupy"
        except Exception:
            pass  # fall through

    if (
        _CUDA_AVAIL
        and n_samples >= CUDA_MIN_ROWS
        and n_pairs >= CUDA_MIN_PAIRS
    ):
        try:
            return batch_pair_mi_cuda(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "cuda"
        except (ValueError, RuntimeError):
            # Shape guard tripped or runtime fault -> fall back to CPU.
            pass

    return batch_pair_mi_njit_prange(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "njit"


__all__ = [
    "batch_pair_mi_njit_prange",
    "batch_pair_mi_cuda",
    "batch_pair_mi_cupy",
    "dispatch_batch_pair_mi",
    "_CUDA_AVAIL",
    "_CUPY_AVAIL",
]
