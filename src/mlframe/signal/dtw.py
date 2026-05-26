"""Dynamic Time Warping with CPU + GPU backends and auto-dispatch.

Three backends:
  * ``dtw_cpu`` -- thin wrapper around ``dtaidistance.dtw.warping_path``.
    Battle-tested C kernel; the dispatcher's CPU fallback.
  * ``dtw_cuda`` -- numba.cuda host-driven diagonal sweep. Works on
    any Pascal+ GPU (CC >= 6.0). Per-diagonal kernel launch overhead
    makes it ~3x slower than ``dtw_cupy`` but doesn't depend on
    cupy / CUDA-side libs beyond numba.cuda.
  * ``dtw_cupy`` -- cupy ``RawKernel`` host-driven diagonal sweep.
    Lowest per-call overhead; ~25-100x faster than dtaidistance on
    sequences >= 2K rows (bench 2026-05-24 on GTX 1050 Ti).

The dispatcher (``dtw_dispatch``) picks GPU when available AND the
sequence is large enough that GPU transfer overhead is amortised
(default crossover 2000 rows, tunable).

Returns ``(distance, warping_path)`` where path is a list of
``(i, j)`` index pairs from cost[n,m] back to cost[0,0] (path[0]
maps the first horizontal row to the first matched typewell row).

All backends use a Sakoe-Chiba band of half-width ``window`` and
squared-Euclidean cost. The CPU backend additionally supports
``psi`` relaxation; the GPU backends currently do not (we get away
with this because the calling project anchors the start via a
preceding z-norm cross-correlation -- making the path nearly-
forced at the boundaries anyway).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = None  # type: ignore
    _HAS_CUPY = False

try:
    from numba import cuda as _nb_cuda  # type: ignore
    _HAS_NB_CUDA = bool(_nb_cuda.is_available())
except Exception:
    _nb_cuda = None  # type: ignore
    _HAS_NB_CUDA = False


# ---------------------------------------------------------------------
# CPU backend
# ---------------------------------------------------------------------


def dtw_cpu(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window: int = 200,
    psi: int = 0,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Thin wrapper around ``dtaidistance.dtw.warping_path``. Returns
    ``(distance, path)``.

    ``psi`` relaxes ``psi`` cells at each boundary; when 0 the path
    must start at (0, 0) and end at (n-1, m-1). A typical caller
    uses ``psi=20`` for soft start/end alignment.
    """
    from dtaidistance import dtw
    x64 = np.ascontiguousarray(x, dtype=np.float64)
    y64 = np.ascontiguousarray(y, dtype=np.float64)
    if psi > 0:
        path = dtw.warping_path(x64, y64, window=window, psi=psi)
    else:
        path = dtw.warping_path(x64, y64, window=window)
    distance = float(dtw.distance(x64, y64, window=window))
    return distance, list(path)


# ---------------------------------------------------------------------
# numba.cuda backend (host-driven diagonal sweep)
# ---------------------------------------------------------------------


if _HAS_NB_CUDA:

    @_nb_cuda.jit
    def _numba_cuda_diagonal_step(x, y, cost, n, m, w, k):
        """Fill one diagonal k of the DTW cost matrix in parallel."""
        tid = _nb_cuda.grid(1)
        i_min = max(1, k - m)
        i_max = min(n, k - 1)
        n_cells = i_max - i_min + 1
        if tid >= n_cells:
            return
        i = i_min + tid
        j = k - i
        if abs(i - j) > w:
            cost[i, j] = 1e18
            return
        d = x[i - 1] - y[j - 1]
        d2 = d * d
        c_diag = cost[i - 1, j - 1]
        c_up = cost[i - 1, j]
        c_left = cost[i, j - 1]
        best = c_diag if c_diag <= c_up else c_up
        if c_left < best:
            best = c_left
        cost[i, j] = d2 + best


def dtw_cuda(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window: int = 200,
) -> Tuple[float, List[Tuple[int, int]]]:
    """numba.cuda banded DTW. Host-driven diagonal sweep; one kernel
    launch per diagonal (n + m total launches). Returns
    ``(distance, path)``.

    Per-launch overhead is ~50 microseconds on CC 6.x; for very short
    sequences this can be slower than the CPU path. ``dtw_dispatch``
    handles that via a size threshold.
    """
    if not _HAS_NB_CUDA:
        raise ImportError(
            "dtw_cuda requires numba.cuda + a CUDA-capable GPU."
        )
    n = len(x)
    m = len(y)
    x_d = _nb_cuda.to_device(np.ascontiguousarray(x, dtype=np.float32))
    y_d = _nb_cuda.to_device(np.ascontiguousarray(y, dtype=np.float32))
    cost_h = np.full((n + 1, m + 1), np.float32(1e18), dtype=np.float32)
    cost_h[0, 0] = np.float32(0.0)
    cost_d = _nb_cuda.to_device(cost_h)
    threads = 256
    for k in range(1, n + m + 1):
        i_min = max(1, k - m)
        i_max = min(n, k - 1)
        n_cells = max(0, i_max - i_min + 1)
        if n_cells == 0:
            continue
        blocks = (n_cells + threads - 1) // threads
        _numba_cuda_diagonal_step[blocks, threads](
            x_d, y_d, cost_d, n, m, window, k,
        )
    _nb_cuda.synchronize()
    cost = cost_d.copy_to_host()
    distance = float(np.sqrt(cost[n, m]))
    path = _backtrace_path(cost, n, m)
    return distance, path


# ---------------------------------------------------------------------
# cupy backend (RawKernel diagonal sweep)
# ---------------------------------------------------------------------


_CUPY_KERNEL_SRC = r"""
extern "C" __global__
void dtw_diagonal(const float* __restrict__ x, const float* __restrict__ y,
                  float* cost, int n, int m, int w, int k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i_min = max(1, k - m);
    int i_max = min(n, k - 1);
    int n_cells = i_max - i_min + 1;
    if (tid >= n_cells) return;
    int i = i_min + tid;
    int j = k - i;
    int stride = m + 1;
    if (abs(i - j) > w) {
        cost[i * stride + j] = 1e18f;
        return;
    }
    float d = x[i - 1] - y[j - 1];
    float d2 = d * d;
    float c_diag = cost[(i - 1) * stride + (j - 1)];
    float c_up   = cost[(i - 1) * stride + j];
    float c_left = cost[i * stride + (j - 1)];
    float best = (c_diag <= c_up) ? c_diag : c_up;
    if (c_left < best) best = c_left;
    cost[i * stride + j] = d2 + best;
}
"""


_CUPY_KERNEL_CACHE: list = [None]


def _get_cupy_kernel():
    if not _HAS_CUPY:
        raise ImportError("dtw_cupy requires cupy + a CUDA-capable GPU.")
    if _CUPY_KERNEL_CACHE[0] is None:
        _CUPY_KERNEL_CACHE[0] = cp.RawKernel(
            _CUPY_KERNEL_SRC, "dtw_diagonal",
        )
    return _CUPY_KERNEL_CACHE[0]


def dtw_cupy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window: int = 200,
) -> Tuple[float, List[Tuple[int, int]]]:
    """cupy RawKernel banded DTW. Host-driven diagonal sweep with one
    kernel launch per diagonal. Returns ``(distance, path)``.

    Per-launch overhead is lower than numba.cuda (no per-call JIT
    dispatch), giving ~3x speedup over ``dtw_cuda`` on Pascal-class
    hardware. Bench 2026-05-24 on GTX 1050 Ti: 6.5K x 3K window=200
    = 240 ms (cupy) vs 1.0 s (numba.cuda) vs 22 s (dtaidistance).
    """
    if not _HAS_CUPY:
        raise ImportError("dtw_cupy requires cupy + a CUDA-capable GPU.")
    kernel = _get_cupy_kernel()
    n = len(x)
    m = len(y)
    x_d = cp.asarray(x, dtype=cp.float32)
    y_d = cp.asarray(y, dtype=cp.float32)
    cost_d = cp.full((n + 1, m + 1), cp.float32(1e18))
    cost_d[0, 0] = cp.float32(0.0)
    threads = 256
    for k in range(1, n + m + 1):
        i_min = max(1, k - m)
        i_max = min(n, k - 1)
        n_cells = max(0, i_max - i_min + 1)
        if n_cells == 0:
            continue
        blocks = (n_cells + threads - 1) // threads
        kernel(
            (blocks,), (threads,),
            (x_d, y_d, cost_d, np.int32(n), np.int32(m),
             np.int32(window), np.int32(k)),
        )
    cp.cuda.Stream.null.synchronize()
    cost = cp.asnumpy(cost_d)
    distance = float(np.sqrt(cost[n, m]))
    path = _backtrace_path(cost, n, m)
    return distance, path


# ---------------------------------------------------------------------
# CPU backtrace (used by both GPU backends)
# ---------------------------------------------------------------------


def _backtrace_path(
    cost: np.ndarray, n: int, m: int,
) -> List[Tuple[int, int]]:
    """O(n+m) backtrace from cost[n,m] to cost[0,0]. Ties resolved
    diagonal-then-up-then-left (standard DTW convention)."""
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        c_diag = cost[i - 1, j - 1]
        c_up = cost[i - 1, j]
        c_left = cost[i, j - 1]
        if c_diag <= c_up and c_diag <= c_left:
            i -= 1
            j -= 1
        elif c_up <= c_left:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------


import os as _os


# Source-code default crossover: below this product n*m the CPU
# dtaidistance C kernel beats GPU. Calibrated 2026-05-24 on
# GTX 1050 Ti (Pascal CC 6.1, 4 GB VRAM); cell counts (warping path
# included for both backends):
#   10K   (n=100,  m=100)  -> GPU wins 13.8x (cupy 4.6 ms vs CPU 63 ms)
#   40K   (n=200,  m=200)  -> GPU wins 5.6x  (cupy 9 ms vs CPU 52 ms)
#   250K  (n=500,  m=500)  -> GPU wins 6.6x
#   1.3M  (n=2.6K, m=500)  -> GPU wins 70x   (cupy 67 ms vs CPU 4.7 s)
#   13.5M (n=2.6K, m=5.2K) -> GPU wins 288x  (cupy 179 ms vs CPU 51 s)
# Cross-over found at ~5K cells (CPU per-call overhead ~50 ms vs cupy
# per-call ~4 ms). Default to 50K cells = ~5x guaranteed margin.
# kernel_tuning_cache is consulted FIRST; this constant is the
# fallback when no cache entry exists for the current HW.
_DEFAULT_GPU_MIN_CELLS = 50_000

# Per-backend force-overrides via env (testing / debugging). One of
# "cpu" / "cuda" / "cupy"; "" means auto-dispatch.
_ENV_BACKEND = "MLFRAME_DTW_BACKEND"


def set_dtw_dispatch_threshold(n_cells: int) -> None:
    """Override the source-code-default GPU-vs-CPU crossover threshold.

    Useful for tests; production callers should rely on the
    ``kernel_tuning_cache`` lookup instead (which holds HW-specific
    crossovers calibrated via ``mlframe.signal._dtw_autotune``).
    """
    global _DEFAULT_GPU_MIN_CELLS
    _DEFAULT_GPU_MIN_CELLS = int(n_cells)


_KERNEL_TUNING_CACHE = None  # populated lazily by _lookup_dtw_threshold
_KERNEL_TUNING_CACHE_LOAD_FAILED = False  # sticky so we don't re-attempt
_LOOKUP_DTW_THRESHOLD_MEMO: dict[int, int] = {}


def _lookup_dtw_threshold(n_cells: int) -> int:
    """Consult ``kernel_tuning_cache`` for the per-HW crossover.

    Mirror of the polyeval / joint_hist_batched dispatcher pattern:
    cache lookup wins, source-code default is the fallback. Cache key
    schema: ``dtw_dispatch`` kernel, region keyed by ``n_cells``;
    region returns ``{"gpu_min_cells": int}``.

    Two-tier memoization (2026-05-25):
    * Per-call cache lookup was 270 ms hot / ~6 s cold per dispatch.
      Refined-typewell calls ``dtw_dispatch`` many times per well, so
      we now load the ``KernelTuningCache`` ONCE at module level and
      memoize the threshold per (n_cells) key.
    * Sticky ``_KERNEL_TUNING_CACHE_LOAD_FAILED`` so a missing
      pyutilz.kernel_tuning_cache module doesn't re-raise on every
      call.
    """
    global _KERNEL_TUNING_CACHE, _KERNEL_TUNING_CACHE_LOAD_FAILED
    memo = _LOOKUP_DTW_THRESHOLD_MEMO.get(int(n_cells))
    if memo is not None:
        return memo
    if _KERNEL_TUNING_CACHE is None and not _KERNEL_TUNING_CACHE_LOAD_FAILED:
        try:
            from pyutilz.system.kernel_tuning_cache import KernelTuningCache
            _KERNEL_TUNING_CACHE = KernelTuningCache.load_or_create()
        except Exception:
            _KERNEL_TUNING_CACHE_LOAD_FAILED = True
    if _KERNEL_TUNING_CACHE is not None:
        try:
            _entry = _KERNEL_TUNING_CACHE.lookup("dtw_dispatch", n_cells=int(n_cells))
            if _entry and "gpu_min_cells" in _entry:
                result = int(_entry["gpu_min_cells"])
                _LOOKUP_DTW_THRESHOLD_MEMO[int(n_cells)] = result
                return result
        except Exception:
            pass
    _LOOKUP_DTW_THRESHOLD_MEMO[int(n_cells)] = _DEFAULT_GPU_MIN_CELLS
    return _DEFAULT_GPU_MIN_CELLS


def dtw_dispatch(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window: int = 200,
    psi: int = 0,
    backend: Optional[str] = None,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Auto-pick the fastest DTW backend for this input.

    Selection order (mirrors ``polyeval_dispatch``):
      1. Env override: ``MLFRAME_DTW_BACKEND=cpu|cuda|cupy`` forces
         the backend (raises ImportError if the forced backend is
         unavailable).
      2. ``backend=...`` kwarg: same forced-selection semantics.
      3. ``psi > 0``: forced CPU (GPU backends do not honour boundary
         relaxation; typical callers use ``psi=20`` for soft
         start/end alignment).
      4. Auto: if cupy is available AND ``len(x) * len(y)`` exceeds
         the per-HW threshold (consulted via ``kernel_tuning_cache``,
         fallback ``_DEFAULT_GPU_MIN_CELLS``), prefer cupy. Else
         numba.cuda under the same gate. Else CPU.

    cupy beats numba.cuda by ~3x on Pascal/Turing because cupy's
    RawKernel has lower per-launch overhead than numba's CUDA
    dispatch; we rank cupy > numba.cuda > cpu and pick the first
    that's available + within its size domain.
    """
    forced = _os.environ.get(_ENV_BACKEND, "") or backend or ""
    if forced == "cpu":
        return dtw_cpu(x, y, window=window, psi=psi)
    if forced == "cuda":
        return dtw_cuda(x, y, window=window)
    if forced == "cupy":
        return dtw_cupy(x, y, window=window)
    if forced:
        raise ValueError(
            f"unknown backend {forced!r}; expected cpu/cuda/cupy"
        )
    if psi > 0:
        return dtw_cpu(x, y, window=window, psi=psi)
    n_cells = len(x) * len(y)
    gpu_min_cells = _lookup_dtw_threshold(n_cells)
    if _HAS_CUPY and n_cells >= gpu_min_cells:
        return dtw_cupy(x, y, window=window)
    if _HAS_NB_CUDA and n_cells >= gpu_min_cells:
        return dtw_cuda(x, y, window=window)
    return dtw_cpu(x, y, window=window, psi=psi)
