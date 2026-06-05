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
#
# ``_CUDA_AVAIL`` consults the central pyutilz probe (``is_cuda_available``)
# rather than re-running ``numba.cuda.is_available()`` inline. The numba
# module itself is still imported here because the ``@_nb_cuda.jit``
# decorator at ``_cuda_kernel_factory`` needs a binding; the probe is just
# how we DECIDE whether to take the GPU path.
try:
    from numba import cuda as _nb_cuda
except Exception:
    _nb_cuda = None

try:
    from pyutilz.core.pythonlib import is_cuda_available as _pyutilz_is_cuda_available
    _CUDA_AVAIL = _pyutilz_is_cuda_available()
except Exception:
    # Fallback to inline numba probe if pyutilz is not importable for some reason.
    try:
        _CUDA_AVAIL = bool(getattr(_nb_cuda, "is_available", lambda: False)()) if _nb_cuda is not None else False
    except Exception:
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
# The shared-memory footprint per block is:
#     MAX_JOINT_BINS_CUDA * MAX_Y_BINS_CUDA * 8  +  MAX_JOINT_BINS_CUDA * 8
# i.e. (joint_card * n_classes_y * int64) for the histogram, plus joint_card *
# int64 for the marginal freqs.
#
# We derive the safe caps from the live device's per-block shared-memory
# budget via ``pyutilz.system.gpu_dispatch.get_shared_mem_budget_per_block``.
# That probe returns the correct per-block ceiling for every shipped compute
# capability (cc 6.x = 48 KB, cc 7.0 Volta = 96 KB opt-in, cc 8.0 A100 = 163 KB
# opt-in, cc 9.0 Hopper = 227 KB opt-in; cf. pyutilz commit 8371ce1 for the
# full table). The cap is locked to MAX_Y_BINS_CUDA=16 (sufficient for 16-class
# multiclass targets) and MAX_JOINT_BINS_CUDA is the largest power-of-2 that
# fits the remaining budget.
#
# Fallback: if pyutilz is unavailable OR the probe fails, we use the cc 6.x
# safe defaults (128, 16) -> 17 KB.

MAX_Y_BINS_CUDA = 16  # supports up to 16-class multiclass targets


def _derive_max_joint_bins(max_y_bins: int) -> int:
    """Pick MAX_JOINT_BINS_CUDA from the live device's per-block shared-memory
    budget (via pyutilz). Solves
        joint * max_y_bins * 8 + joint * 8 <= budget
    for ``joint``, then rounds DOWN to the nearest power of 2 for kernel-launch
    safety. Caps at 1024 (kernel design guarantees correctness only up to
    that joint cardinality). Falls back to 128 (cc 6.x safe) on probe failure.
    """
    try:
        from pyutilz.system.gpu_dispatch import (
            gpu_capability_summary,
            get_shared_mem_budget_per_block,
        )
        summary = gpu_capability_summary(0) if _CUDA_AVAIL else None
        if summary is None:
            return 128
        budget = get_shared_mem_budget_per_block(
            summary["cc_major"], summary["cc_minor"], allow_opt_in=False,
        )
        # Solve joint * (max_y_bins + 1) * 8 <= budget.
        raw = budget // ((max_y_bins + 1) * 8)
        if raw < 16:
            return 16
        # Round down to nearest power of 2 to keep CUDA shared-mem alignment
        # predictable.
        joint = 1
        while joint * 2 <= raw and joint < 1024:
            joint *= 2
        return joint
    except Exception:
        return 128


MAX_JOINT_BINS_CUDA = _derive_max_joint_bins(MAX_Y_BINS_CUDA)


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
    # Wave 47 (2026-05-20): empty factors_data divides by zero in inv_n; return zeros.
    if n_samples == 0:
        return np.zeros(n_pairs, dtype=np.float64)
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
# Wave 23 P1 (2026-05-20): the 4 hardcoded thresholds below are
# "measured on GTX 1050 Ti cc 6.1" defaults; per
# `feedback_use_kernel_tuning_cache_for_gpu` they should be lookup-
# driven via ``pyutilz.performance.kernel_tuning.cache`` so that consumer
# Ampere GPUs (~5-10x lower cuda crossover) and high-VRAM cards don't
# leave 2-4x on the table.
#
# These remain as the source-code fallback; the live dispatch path
# below now consults the cache first and uses these only when the
# cache has no entry for the live HW yet (first-run / no-sweep state).
CUDA_MIN_ROWS = 400_000
CUDA_MIN_PAIRS = 16
CUPY_MIN_ROWS = 5_000_000
CUPY_MIN_PAIRS = 200

# Kernel-tuning-cache integration (get_or_tune + @kernel_tuner) ---------------
#
# Wave 23 P1 (2026-05-20) flagged the four hardcoded crossover thresholds above
# as HW-overfit (GTX 1050 Ti cc 6.1). Per ``feedback_use_kernel_tuning_cache_for_gpu``
# the live dispatch now consults the per-host cache via the shared ``get_or_tune``
# orchestrator and a measured backend-crossover sweep. The thresholds above remain
# the source-code fallback, applied verbatim when the cache has no entry for the
# live HW yet (mirrors ``signal/dtw.py``).
#
# 2-D axes: ``n_samples`` (dominant; primary sweep axis) and ``n_pairs`` (held at a
# representative value for the sweep -- 64, above CUDA_MIN_PAIRS=16 and below
# CUPY_MIN_PAIRS=200, matching the measured 28-120-pair bench band -- and threaded
# through as an extra region key so the cached regions stay keyed on both axes).
_BPMI_SWEEP_N_PAIRS_GRID = [16, 64, 256]  # full n_pairs axis (was a single fixed 64)
_BPMI_SWEEP_N_SAMPLES = [200_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
_BPMI_SWEEP_N_CLASSES_Y = 4
_BPMI_SWEEP_N_BINS = 8  # joint card = 8*8 = 64 <= MAX_JOINT_BINS_CUDA fallback (128)
_BPMI_SALT = 2  # serial-njit variant added + full 2-D (n_samples x n_pairs) grid

# Serial CPU variant: recompile the SAME prange body WITHOUT ``parallel`` -> numba
# treats prange as range. The tuner now picks njit_serial (small n: no thread-spawn
# overhead) vs njit_parallel per region, not just CPU-vs-GPU.
from numba import njit as _njit

# getattr fallback: under NUMBA_DISABLE_JIT=1 the kernel is a plain function (no
# .py_func) and njit is a pass-through, so serial == the same callable.
batch_pair_mi_njit_serial = _njit(nogil=True, cache=True)(
    getattr(batch_pair_mi_njit_prange, "py_func", batch_pair_mi_njit_prange)
)


def _make_batch_pair_mi_inputs(dims: dict):
    """Synthetic (factors_data, pair_a, pair_b, nbins, classes_y, freqs_y) at
    ``dims['n_samples']`` rows x ``dims['n_pairs']`` pairs. Bins are capped so the
    per-pair joint cardinality stays inside the CUDA shared-mem budget (so the cuda
    variant is exercised, not guard-rejected)."""
    rng = np.random.default_rng(0)
    n_samples = int(dims["n_samples"])
    n_pairs = int(dims["n_pairs"])
    nbins_val = _BPMI_SWEEP_N_BINS
    n_features = 32  # enough columns for up to 256 distinct (a, b) pairs
    factors_data = rng.integers(0, nbins_val, size=(n_samples, n_features)).astype(np.int32)
    nbins = np.full(n_features, nbins_val, dtype=np.int32)
    pair_a = rng.integers(0, n_features, size=n_pairs).astype(np.int64)
    pair_b = (pair_a + 1 + rng.integers(0, n_features - 1, size=n_pairs)) % n_features
    pair_b = pair_b.astype(np.int64)
    classes_y = rng.integers(0, _BPMI_SWEEP_N_CLASSES_Y, size=n_samples).astype(np.int32)
    freqs_y = np.bincount(classes_y, minlength=_BPMI_SWEEP_N_CLASSES_Y).astype(np.float64) / max(1, n_samples)
    return (factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)


def _run_batch_pair_mi_sweep() -> list:
    """Full (n_samples x n_pairs) grid sweep -> backend_choice regions: njit_serial /
    njit_parallel / cuda / cupy, fastest EQUIVALENT per cell. Both n_samples and
    n_pairs are swept (not a fixed-n_pairs 1-D crossover). Inputs are host-resident
    (the FS pipeline feeds the host dataframe) so there is no residency axis. GPU
    variants only when available; cupy/cuda fp reductions agree with njit to a
    loosened rtol (last-bit log() reassociation)."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {
        "njit_serial": lambda *a: batch_pair_mi_njit_serial(*a),
        "njit_parallel": lambda *a: batch_pair_mi_njit_prange(*a),
    }
    if _CUDA_AVAIL:
        variants["cuda"] = lambda *a: batch_pair_mi_cuda(*a)
    if _CUPY_AVAIL:
        variants["cupy"] = lambda *a: batch_pair_mi_cupy(*a)
    return sweep_backend_grid(
        variants,
        {"n_samples": _BPMI_SWEEP_N_SAMPLES, "n_pairs": _BPMI_SWEEP_N_PAIRS_GRID},
        _make_batch_pair_mi_inputs,
        reference="njit_serial",
        repeats=5, equiv_rtol=1e-3, equiv_atol=1e-3,
    )


def _batch_pair_mi_code_version():
    """code_version over the CPU bodies (serial + parallel share one source) + the
    available GPU bodies; re-tunes on a kernel edit."""
    try:
        from pyutilz.performance.kernel_tuning.code_versioning import compute_code_version

        fns = [batch_pair_mi_njit_serial, batch_pair_mi_njit_prange]
        if _CUDA_AVAIL:
            fns.append(batch_pair_mi_cuda)
        if _CUPY_AVAIL:
            fns.append(batch_pair_mi_cupy)
        return compute_code_version(*fns, salt=_BPMI_SALT)
    except Exception:
        return None


def _batch_pair_mi_fallback_choice(n_samples: int, n_pairs: int) -> str:
    """Pre-sweep heuristic: the old CUDA_/CUPY_MIN_* GPU thresholds + availability;
    on CPU, parallel njit above a modest row count (below it the thread-spawn
    overhead loses to serial)."""
    if _CUPY_AVAIL and n_samples >= CUPY_MIN_ROWS and n_pairs >= CUPY_MIN_PAIRS:
        return "cupy"
    if _CUDA_AVAIL and n_samples >= CUDA_MIN_ROWS and n_pairs >= CUDA_MIN_PAIRS:
        return "cuda"
    return "njit_parallel" if n_samples >= 100_000 else "njit_serial"


def _batch_pair_mi_backend_choice(n_samples: int, n_pairs: int) -> str:
    """Per-host backend (njit/cuda/cupy) for this (n_samples, n_pairs) via the shared
    get_or_tune orchestrator; measurement-backed threshold fallback. ``n_samples`` is the
    swept (primary) axis; ``n_pairs`` is passed in the dims so the cache key is complete."""
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        result = KernelTuningCache().get_or_tune(
            "batch_pair_mi",
            dims={"n_samples": int(n_samples), "n_pairs": int(n_pairs)},
            tuner=_run_batch_pair_mi_sweep,
            axes=["n_samples", "n_pairs"],
            fallback={"backend_choice": _batch_pair_mi_fallback_choice(n_samples, n_pairs)},
            code_version=_batch_pair_mi_code_version(),
        )
        bc = result if isinstance(result, str) else str((result or {}).get("backend_choice", ""))
        if bc == "njit":  # legacy region from before the serial/parallel split
            bc = "njit_parallel"
        if bc in ("njit_serial", "njit_parallel", "cuda", "cupy"):
            return bc
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("batch_pair_mi get_or_tune failed: %s", e)
    return _batch_pair_mi_fallback_choice(n_samples, n_pairs)


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

    # Per-host backend (njit/cuda/cupy) from the kernel_tuning_cache via the shared
    # get_or_tune orchestrator; measurement-backed fallback = the old CUDA_/CUPY_MIN_*
    # thresholds. Guarded by live availability (the tuning host had the backend; a
    # reader may not) -- preserves the original cupy-then-cuda-then-njit preference order.
    choice = _batch_pair_mi_backend_choice(n_samples, n_pairs)

    if choice == "cupy" and _CUPY_AVAIL:
        try:
            return batch_pair_mi_cupy(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "cupy"
        except Exception:
            pass  # fall through

    if choice == "cuda" and _CUDA_AVAIL:
        try:
            return batch_pair_mi_cuda(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "cuda"
        except (ValueError, RuntimeError):
            # Shape guard tripped or runtime fault -> fall back to CPU.
            pass

    # CPU: serial vs parallel njit per the tuned choice (tag stays "njit").
    if choice == "njit_serial":
        return batch_pair_mi_njit_serial(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "njit"
    return batch_pair_mi_njit_prange(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "njit"


# Register with the @kernel_tuner registry so retune_all / mlframe-tune-kernels
# discover + batch-tune batch_pair_mi. GPU-capable (cuda/cupy backends).
from pyutilz.performance.kernel_tuning.registry import kernel_tuner

kernel_tuner(
    kernel_name="batch_pair_mi",
    variant_fns=(batch_pair_mi_njit_serial, batch_pair_mi_njit_prange),  # CPU bodies; GPU covered by salt
    tuner=_run_batch_pair_mi_sweep,
    axes={"n_samples": list(_BPMI_SWEEP_N_SAMPLES), "n_pairs": list(_BPMI_SWEEP_N_PAIRS_GRID)},
    fallback={"backend_choice": "njit_serial"},
    gpu_capable=True,
    salt=_BPMI_SALT,
    cli_label="batch_pair_mi",
)


__all__ = [
    "batch_pair_mi_njit_prange",
    "batch_pair_mi_cuda",
    "batch_pair_mi_cupy",
    "dispatch_batch_pair_mi",
    "_CUDA_AVAIL",
    "_CUPY_AVAIL",
]
