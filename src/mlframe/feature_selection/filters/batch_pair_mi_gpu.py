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

import logging

import numpy as np

# Re-export CPU baseline for callers who want a single import point.
from .info_theory import batch_pair_mi_prange as batch_pair_mi_njit_prange

logger = logging.getLogger(__name__)

# Optional GPU deps. The dispatcher gracefully falls back to the CPU kernel
# when either is missing.
#
# The numba.cuda kernel machinery (shared-memory-budget derivation, kernel
# factories, ``batch_pair_mi_cuda``/``batch_pair_mi_cuda_row_chunked``) lives
# in the sibling ``_batch_pair_mi_cuda_kernels`` module (carved out to keep
# this file under the repo's 1000-LOC gate). ``_CUDA_AVAIL`` is re-exported
# from there as the single source of truth.
from ._batch_pair_mi_cuda_kernels import (
    MAX_JOINT_BINS_CUDA,  # noqa: F401 -- re-exported facade name, imported directly by tests/benchmarks
    MAX_Y_BINS_CUDA,  # noqa: F401 -- re-exported facade name, imported directly by tests/benchmarks
    _CUDA_AVAIL,
    _choose_pair_subchunk_rows,  # noqa: F401 -- re-exported facade name, imported directly by tests/benchmarks
    _choose_row_chunk_rows,  # noqa: F401 -- re-exported facade name, imported directly by tests/benchmarks
    _hist_kernel_shared_fits_budget,  # noqa: F401 -- re-exported facade name, imported directly by tests/benchmarks
    _new_zeroed_device_array,  # noqa: F401 -- re-exported facade name, imported directly by tests/benchmarks
    batch_pair_mi_cuda,
    batch_pair_mi_cuda_row_chunked,
)

try:
    import cupy as _cp
    _CUPY_AVAIL = True
except Exception:
    _cp = None
    _CUPY_AVAIL = False


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
    # Stage each pair's scalar MI into a RESIDENT (n_pairs,) device buffer and D2H it ONCE at the end, instead
    # of a blocking ``float(mi.get())`` per pair (n_pairs serialising syncs that drain the queue between pairs).
    # Bit-identical: same per-pair scalar written to out_dev[p]; only the transfer is batched into one .get().
    out_dev = cp.empty(n_pairs, dtype=cp.float64)

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
        out_dev[p] = (joint_freqs * log_term).sum()

    return np.asarray(cp.asnumpy(out_dev))


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
# Grid floor reaches below the GPU crossover (measured ~85-100k rows on a laptop RTX 500 Ada) so the cache learns the CPU-favorable
# low-n region instead of extrapolating the lowest measured cell down to n=0 (which mis-routed 50-75k-row calls to a slower GPU launch).
_BPMI_SWEEP_N_SAMPLES = [50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
_BPMI_SWEEP_N_CLASSES_Y = 4
_BPMI_SWEEP_N_BINS = 8  # joint card = 8*8 = 64 <= MAX_JOINT_BINS_CUDA fallback (128)
_BPMI_SALT = 3  # serial-njit variant + full 2-D (n_samples x n_pairs) grid + grid floor lowered below the GPU crossover

# Serial CPU variant: recompile the SAME prange body WITHOUT ``parallel`` -> numba
# treats prange as range. The tuner now picks njit_serial (small n: no thread-spawn
# overhead) vs njit_parallel per region, not just CPU-vs-GPU.
from numba import njit as _njit

# getattr fallback: under NUMBA_DISABLE_JIT=1 the kernel is a plain function (no
# .py_func) and njit is a pass-through, so serial == the same callable.
batch_pair_mi_njit_serial = _njit(nogil=True, cache=True)(getattr(batch_pair_mi_njit_prange, "py_func", batch_pair_mi_njit_prange))


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
    return sweep_backend_grid(  # type: ignore[no-any-return]  # pyutilz helper returns the declared list of results
        variants,
        {"n_samples": _BPMI_SWEEP_N_SAMPLES, "n_pairs": _BPMI_SWEEP_N_PAIRS_GRID},
        _make_batch_pair_mi_inputs,
        reference="njit_serial",
        repeats=5, equiv_rtol=1e-3, equiv_atol=1e-3,
    )


def _batch_pair_mi_fallback_choice(n_samples: int, n_pairs: int) -> str:
    """Pre-sweep heuristic (the spec's dynamic fallback callable): the old
    CUDA_/CUPY_MIN_* GPU thresholds + availability; on CPU, parallel njit above a
    modest row count (below it the thread-spawn overhead loses to serial)."""
    if _CUPY_AVAIL and n_samples >= CUPY_MIN_ROWS and n_pairs >= CUPY_MIN_PAIRS:
        return "cupy"
    if _CUDA_AVAIL and n_samples >= CUDA_MIN_ROWS and n_pairs >= CUDA_MIN_PAIRS:
        return "cuda"
    return "njit_parallel" if n_samples >= 100_000 else "njit_serial"


def _batch_pair_mi_backend_choice(n_samples: int, n_pairs: int) -> str:
    """Per-host backend (njit_serial/njit_parallel/cuda/cupy) for this (n_samples,
    n_pairs) via the spec's choose(); maps a legacy 'njit' region (pre serial/parallel
    split) to njit_parallel."""
    bc = _BPMI_SPEC.choose(n_samples=int(n_samples), n_pairs=int(n_pairs))
    return "njit_parallel" if bc == "njit" else bc


def _required_gpu_bytes(factors_data: np.ndarray, pair_a: np.ndarray, nbins: np.ndarray, classes_y: np.ndarray, freqs_y: np.ndarray) -> int:
    """Estimated device-resident bytes for one ``batch_pair_mi_cuda``/``batch_pair_mi_cupy`` call.

    ``factors_data`` is uploaded WHOLESALE (every column, not just the ones referenced by this pair
    chunk) and always as int32 regardless of its host dtype -- that upload dominates the footprint at
    production scale (n=2.4M rows already needs ~4GB as int32, i.e. the entire VRAM budget of a 4GB
    card) and is invariant across chunks, so this must be checked BEFORE every cuda/cupy attempt, not
    just once.
    """
    n_pairs = int(pair_a.shape[0])
    return int(factors_data.size * 4 + n_pairs * (8 + 8 + 8) + nbins.size * 4 + classes_y.size * 4 + freqs_y.size * 8)


def _gpu_upload_fits(required_bytes: int, *, n_samples: int = 0, n_cols: int = 0, n_pairs: int = 0, context: str = "batch_pair_mi") -> bool:
    """Pre-flight VRAM check before launching a cuda/cupy pair-MI kernel -- mirrors the ``_should_use_cuda``
    guard pattern already used by ``_cmi_cuda.py`` / ``gpu.py`` / ``hermite_fe`` / ``friend_graph_gpu.py`` /
    ``batch_mi_noise_gate_gpu.py`` / ``_permutation_null_pair_resident.py`` (this module was the one
    remaining GPU dispatch site without the guard). Two layers: a relative cap (<=50% of currently free
    VRAM, capped at 1.5 GB) as a cheap first pass, then the shared ABSOLUTE cushion floor from
    ``_fe_gpu_vram.fe_gpu_has_vram_cushion`` (>=1 GB free after the allocation, env-overridable via
    ``MLFRAME_FE_GPU_MIN_FREE_MB``) so every GPU dispatch site in this package agrees on one definition of
    "safe to launch".

    Why this matters (2026-07-10 wellbore 3M-row production crash): on a small-VRAM WDDM host, uploading an
    oversized array does NOT reliably raise a catchable CUDA OOM -- WDDM can transparently over-subscribe
    device memory via host-paging, so the upload "succeeds" and the kernel launch then grinds through
    PCIe-paged memory for minutes before the OS kills the process with NO Python exception, no traceback,
    and no Windows Event Log trace (silent ``EXIT_CODE=1``, confirmed via a real 2.4M-row/423-column
    production run). A pre-flight check is the only way to avoid entering that state at all; catching an
    exception afterward is too late because there may be none to catch.

    A REJECTION is always logged at WARNING with the full sizing context (rows/cols/pairs/dtype, requested
    GB, free/total VRAM) so a production run is diagnosable from the log alone -- never a silent fallback.

    Permissive (``True``) whenever cupy/memGetInfo is unavailable, matching every sibling guard's fail-open
    contract for non-GPU / probe-failure hosts.
    """
    cap = 1536 * 1024 * 1024  # 1.5 GB conservative cap (shared small card), same default as _cmi_cuda._should_use_cuda
    try:
        import cupy as cp

        free_b, total_b = cp.cuda.runtime.memGetInfo()
        cap = min(cap, int(free_b * 0.5))
    except Exception as e:
        logger.debug("%s._gpu_upload_fits: memGetInfo failed (%s); permissive", context, e)
        return True
    if required_bytes > cap:
        try:
            pool = cp.get_default_memory_pool()
            if int(pool.free_bytes()) > 0:
                pool.free_all_blocks()
                free_b, total_b = cp.cuda.runtime.memGetInfo()
                cap = min(1536 * 1024 * 1024, int(free_b * 0.5))
        except Exception as e:
            logger.debug("%s._gpu_upload_fits: pool flush/re-probe failed (%s); keeping prior cap", context, e)
    if required_bytes > cap:
        logger.warning(
            "%s: GPU upload REJECTED -- requested %.2fGB (n_samples=%d, n_cols=%d, n_pairs=%d, dtype=int32) "
            "exceeds the safe relative cap %.2fGB (50%% of %.2fGB free / %.2fGB total VRAM) -- routing to a "
            "row-chunked GPU path or CPU njit instead of risking a silent VRAM-oversubscription crash",
            context, required_bytes / 1024**3, n_samples, n_cols, n_pairs, cap / 1024**3, free_b / 1024**3, total_b / 1024**3,
        )
        return False
    try:
        from mlframe.feature_selection.filters._fe_gpu_vram import fe_gpu_has_vram_cushion

        if not fe_gpu_has_vram_cushion(required_bytes):
            logger.warning(
                "%s: GPU upload REJECTED -- requested %.2fGB (n_samples=%d, n_cols=%d, n_pairs=%d, dtype=int32) "
                "would breach the absolute VRAM cushion floor (free=%.2fGB, total=%.2fGB) -- routing to a "
                "row-chunked GPU path or CPU njit instead of risking a silent VRAM-oversubscription crash",
                context, required_bytes / 1024**3, n_samples, n_cols, n_pairs, free_b / 1024**3, total_b / 1024**3,
            )
            return False
    except Exception as e:
        logger.debug("%s._gpu_upload_fits: cushion probe failed (%s); permissive", context, e)
    return True


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

    When the FULL upload would not safely fit in free VRAM (see :func:`_gpu_upload_fits`) but CUDA IS
    available, routes to :func:`batch_pair_mi_cuda_row_chunked` -- a row-chunked GPU path that still gets
    the GPU speed win by uploading ``factors_data`` in VRAM-sized row-blocks and accumulating the joint
    histogram across them (bit-identical result; see that function's docstring). Only fully drops to the
    CPU njit kernel when even that fails (no CUDA, or a genuine runtime/driver fault) -- a slower CORRECT
    result is still preferred over ever risking the silent-crash upload, but "slower" no longer means
    "no GPU at all" whenever CUDA is present.
    """
    n_samples = int(factors_data.shape[0])
    n_cols = int(factors_data.shape[1]) if factors_data.ndim == 2 else 0
    n_pairs = int(pair_a.shape[0])
    _req_bytes = _required_gpu_bytes(factors_data, pair_a, nbins, classes_y, freqs_y)
    _vram_ok = _gpu_upload_fits(_req_bytes, n_samples=n_samples, n_cols=n_cols, n_pairs=n_pairs)

    def _try_cuda_row_chunked(reason: str) -> tuple[np.ndarray, str] | None:
        """Attempt the row-chunked CUDA kernel; returns None (falls through to CPU) on any failure."""
        if not _CUDA_AVAIL:
            return None
        try:
            mi = batch_pair_mi_cuda_row_chunked(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)
            logger.info("batch_pair_mi: %s -- completed via row-chunked CUDA (GPU speed preserved, VRAM-safe)", reason)
            return mi, "cuda_row_chunked"
        except Exception as e:
            logger.warning("batch_pair_mi: row-chunked CUDA also failed (%s: %s) -- falling back to CPU njit", type(e).__name__, e)
            return None

    # Explicit override
    if force_backend is not None:
        force_backend = force_backend.lower()
        if force_backend == "cuda" and _CUDA_AVAIL:
            if _vram_ok:
                try:
                    return batch_pair_mi_cuda(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "cuda"
                except Exception as e:
                    logger.warning("batch_pair_mi: forced CUDA backend failed (%s: %s) -- trying row-chunked CUDA", type(e).__name__, e)
                    _result = _try_cuda_row_chunked("forced CUDA backend failed on the full-upload path")
                    if _result is not None:
                        return _result
            else:
                _result = _try_cuda_row_chunked("forced CUDA backend requested but full upload does not fit VRAM")
                if _result is not None:
                    return _result
        elif force_backend == "cupy" and _CUPY_AVAIL and _vram_ok:
            try:
                return batch_pair_mi_cupy(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "cupy"
            except Exception as e:
                logger.warning("batch_pair_mi: forced cupy backend failed (%s: %s) -- falling back to CPU njit", type(e).__name__, e)
        return batch_pair_mi_njit_prange(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "njit"

    # Per-host backend (njit/cuda/cupy) from the kernel_tuning_cache via the shared
    # get_or_tune orchestrator; measurement-backed fallback = the old CUDA_/CUPY_MIN_*
    # thresholds. Guarded by live availability (the tuning host had the backend; a
    # reader may not) -- preserves the original cupy-then-cuda-then-njit preference order.
    choice = _batch_pair_mi_backend_choice(n_samples, n_pairs)

    if choice == "cupy" and _CUPY_AVAIL and _vram_ok:
        try:
            return batch_pair_mi_cupy(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "cupy"
        except Exception:  # nosec B110 - optional/best-effort path, rationale documented
            pass  # fall through

    if choice == "cuda" and _CUDA_AVAIL:
        if _vram_ok:
            try:
                return batch_pair_mi_cuda(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "cuda"
            except Exception:
                # Shape guard tripped or a runtime/driver fault -> try row-chunked CUDA, then CPU. Broadened
                # from ``(ValueError, RuntimeError)`` (2026-07-10): numba's ``CudaAPIError``/``CudaDriverError``
                # derive directly from ``Exception``, not ``RuntimeError``, so a genuine CUDA driver fault
                # used to skip this handler and propagate to the caller uncaught.
                _result = _try_cuda_row_chunked("full-upload CUDA kernel raised")
                if _result is not None:
                    return _result
        else:
            _result = _try_cuda_row_chunked("size-heuristic picked CUDA but full upload does not fit VRAM")
            if _result is not None:
                return _result

    # CPU: serial vs parallel njit per the tuned choice (tag stays "njit").
    if choice == "njit_serial":
        return batch_pair_mi_njit_serial(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "njit"
    return batch_pair_mi_njit_prange(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "njit"


def _free_ram_bytes_for_chunking() -> int:
    """Best-effort free physical RAM in bytes; conservative fallback if psutil is missing. Mirrors
    ``_mrmr_sis_screen._free_ram_bytes`` -- duplicated (not imported) to avoid a cross-package import
    cycle (``_mrmr_sis_screen`` itself lazily imports sibling FE modules that can reach this file)."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return 2 * 1024**3  # 2 GB conservative fallback


def _fallback_pair_chunk_size(free_bytes: int) -> int:
    """Measurement-backed default pairs-per-chunk from free RAM alone.

    Per chunk we hold, at most simultaneously: two int64 id arrays (``pair_a``/``pair_b``) and one
    float64 output array, i.e. ``chunk_pairs * (8 + 8 + 8)`` bytes, plus per-backend transient overhead
    (the CUDA/CuPy paths additionally stage a same-sized device-resident buffer). Budget conservatively
    at ~1/16 of free RAM for the transient so this never competes meaningfully with the caller's own
    ``data``/``cached_MIs`` state. Clamped to a sane [50_000, 20_000_000] pairs.
    """
    budget = max(1, free_bytes // 16)
    chunk = int(budget // 48)  # 3 arrays * 8 bytes * ~2x safety headroom
    return int(np.clip(chunk, 50_000, 20_000_000))


def _choose_pair_chunk_size(free_bytes: int) -> int:
    """Look the pairs-per-chunk up in the kernel_tuning_cache keyed on a free-RAM bucket; fall back to
    the measured analytic default. Mirrors ``_mrmr_sis_screen._choose_chunk_width``'s pattern -- this is
    a MEMORY-SAFETY bound (never hardcoded), not a throughput choice; the throughput-critical decision
    (which backend: njit/cuda/cupy) remains fully delegated to :func:`dispatch_batch_pair_mi` per chunk,
    which is already kernel_tuning_cache-driven via ``_batch_pair_mi_backend_choice``."""
    gb = max(1, int(free_bytes // (1024**3)))
    ram_bucket = int(gb.bit_length())
    fb = _fallback_pair_chunk_size(free_bytes)
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        ktc = KernelTuningCache.load_or_create()
        hit = ktc.lookup("mrmr_batch_pair_mi_chunk_size", ram_bucket=ram_bucket)
        if hit and "chunk_pairs" in hit:
            return int(np.clip(int(hit["chunk_pairs"]), 50_000, 20_000_000))
        try:
            ktc.update(
                "mrmr_batch_pair_mi_chunk_size",
                axes=["ram_bucket"],
                regions=[{"ram_bucket": ram_bucket, "chunk_pairs": fb}],
            )
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            import logging

            logging.getLogger(__name__).debug("suppressed in batch_pair_mi_gpu.py (chunk-size cache update): %s", e)
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        import logging

        logging.getLogger(__name__).debug("suppressed in batch_pair_mi_gpu.py (chunk-size cache lookup): %s", e)
    return fb


def _iter_upper_triangle_pair_chunks(k: int, chunk_pairs: int):
    """Yield ``(a_pos, b_pos)`` int64 POSITION arrays (0-based positions into a length-``k`` id list) for
    successive row-blocks of the upper-triangle pair space (``a_pos < b_pos``), each containing at most
    ``chunk_pairs`` pairs.

    Never materialises the full ``C(k, 2)`` pair list: each row ``i`` contributes ``k - 1 - i`` pairs via
    a plain ``np.arange``, so the per-chunk cost is ``O(chunk_pairs)`` and the total cost across the whole
    generator is ``O(k + total_pairs)`` -- the same asymptotic work an exhaustive pairwise scan requires
    regardless of implementation, with peak memory bounded by ``chunk_pairs`` instead of ``C(k, 2)``.
    """
    if k < 2 or chunk_pairs < 1:
        return
    i = 0
    while i < k - 1:
        rows: list[int] = []
        cum = 0
        while i < k - 1 and cum < chunk_pairs:
            rows.append(i)
            cum += k - 1 - i
            i += 1
        row_ids = np.asarray(rows, dtype=np.int64)
        counts = (k - 1 - row_ids).astype(np.int64)
        a_pos = np.repeat(row_ids, counts)
        b_pos = np.concatenate([np.arange(r + 1, k, dtype=np.int64) for r in rows])
        yield a_pos, b_pos


def dispatch_batch_pair_mi_chunked(
    factors_data: np.ndarray,
    ids: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    force_backend: str | None = None,
    max_pairs_per_chunk: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Full-upper-triangle batch pair-MI over ``ids`` (all-pairs among ``ids``), processed in RAM-bounded
    row-block chunks so peak memory never scales with ``C(len(ids), 2)``.

    Replaces the previous pattern of building the full ``pair_a``/``pair_b``/output arrays via
    ``np.triu_indices`` up front (``O(k^2)`` memory -- infeasible past a few thousand columns; at
    k=100_000, ``C(k,2)`` ~= 5e9 pairs would need ~120 GB just for the index/output arrays). Each chunk is
    still dispatched through the existing :func:`dispatch_batch_pair_mi` (so backend selection stays fully
    kernel_tuning_cache-driven, per-chunk); only the ENUMERATION of which pairs to compute is chunked.

    Returns ``(pair_a_ids, pair_b_ids, mi_values, backend_counts)`` where ``pair_a_ids``/``pair_b_ids`` are
    the actual column ids (not positions) and ``backend_counts`` maps backend name -> number of chunks
    that ran on it (for logging; a mixed-backend run is possible if a GPU chunk fails mid-sweep and the
    per-chunk call falls through to CPU).
    """
    ids_arr = np.asarray(ids, dtype=np.int64)
    k = int(ids_arr.shape[0])
    if k < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64), {}

    chunk_pairs = int(max_pairs_per_chunk) if max_pairs_per_chunk else _choose_pair_chunk_size(_free_ram_bytes_for_chunking())
    chunk_pairs = max(1, chunk_pairs)

    a_out: list[np.ndarray] = []
    b_out: list[np.ndarray] = []
    mi_out: list[np.ndarray] = []
    backend_counts: dict[str, int] = {}

    for a_pos, b_pos in _iter_upper_triangle_pair_chunks(k, chunk_pairs):
        pair_a = ids_arr[a_pos]
        pair_b = ids_arr[b_pos]
        mi_chunk, backend_used = dispatch_batch_pair_mi(
            factors_data=factors_data,
            pair_a=pair_a,
            pair_b=pair_b,
            nbins=nbins,
            classes_y=classes_y,
            freqs_y=freqs_y,
            force_backend=force_backend,
        )
        a_out.append(pair_a)
        b_out.append(pair_b)
        mi_out.append(mi_chunk)
        backend_counts[backend_used] = backend_counts.get(backend_used, 0) + 1

    if not a_out:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64), {}

    return (
        np.concatenate(a_out),
        np.concatenate(b_out),
        np.concatenate(mi_out),
        backend_counts,
    )


# Register with the @kernel_tuner registry so retune_all / mlframe-tune-kernels
# discover + batch-tune batch_pair_mi. GPU-capable (cuda/cupy backends).
from pyutilz.performance.kernel_tuning.registry import kernel_tuner

_BPMI_SPEC = kernel_tuner(
    kernel_name="batch_pair_mi",
    variant_fns=(batch_pair_mi_njit_serial, batch_pair_mi_njit_prange),  # CPU bodies; GPU covered by salt
    tuner=_run_batch_pair_mi_sweep,
    axes={"n_samples": list(_BPMI_SWEEP_N_SAMPLES), "n_pairs": list(_BPMI_SWEEP_N_PAIRS_GRID)},
    fallback=_batch_pair_mi_fallback_choice,  # callable (n_samples, n_pairs) -> str
    gpu_capable=True,
    salt=_BPMI_SALT,
    cli_label="batch_pair_mi",
)


__all__ = [
    "batch_pair_mi_njit_prange",
    "batch_pair_mi_cuda",
    "batch_pair_mi_cupy",
    "dispatch_batch_pair_mi",
    "dispatch_batch_pair_mi_chunked",
    "_CUDA_AVAIL",
    "_CUPY_AVAIL",
]
