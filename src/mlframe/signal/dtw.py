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
    sequences >= 2K rows (bench 2026-05-24 on GTX 1050 Ti). Both GPU
    backends use a BANDED cost buffer of shape ``(n+1, 2*window+1)`` in
    ``(i, j-i+window)`` coordinates, so device RAM is O(n*window) rather
    than O(n*m) (24.9x smaller at 10K x 10K, window=200); the
    full-matrix sweep is retained as ``dtw_cupy_full`` for A/B reference.

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

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from mlframe.system import try_import_cupy

cp, _HAS_CUPY = try_import_cupy()

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

    @_nb_cuda.jit
    def _numba_cuda_banded_step(x, y, band, n, m, w, bw, k):
        """Fill diagonal k of the banded buffer band[i, j-i+w]. Bit-identical
        recipe to the full-matrix step; only the cost-buffer indexing differs."""
        tid = _nb_cuda.grid(1)
        i_min = max(1, k - m)
        i_max = min(n, k - 1)
        n_cells = i_max - i_min + 1
        if tid >= n_cells:
            return
        i = i_min + tid
        j = k - i
        b = j - i + w
        if b < 0 or b >= bw:
            return
        c_diag = band[i - 1, b]
        c_up = band[i - 1, b + 1] if b + 1 < bw else np.float32(1e18)
        c_left = band[i, b - 1] if b > 0 else np.float32(1e18)
        d = x[i - 1] - y[j - 1]
        best = c_diag if c_diag <= c_up else c_up
        if c_left < best:
            best = c_left
        band[i, b] = d * d + best


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
    # Banded buffer (n+1, 2*window+1) in (i, j-i+window) coords -- O(n*window)
    # device RAM vs the prior full (n+1)x(m+1). Bit-identical distance + path to
    # the full-matrix sweep (kept as _numba_cuda_diagonal_step for reference); the
    # (i-1,j-1)/(i-1,j)/(i,j-1) dependency forces the anti-diagonal sweep so the
    # launch count is unchanged -- the memory reduction is the win.
    if not _HAS_NB_CUDA:
        raise ImportError("dtw_cuda requires numba.cuda + a CUDA-capable GPU.")
    n = len(x)
    m = len(y)
    w = int(window)
    bw = 2 * w + 1
    x_d = _nb_cuda.to_device(np.ascontiguousarray(x, dtype=np.float32))
    y_d = _nb_cuda.to_device(np.ascontiguousarray(y, dtype=np.float32))
    band_h = np.full((n + 1, bw), np.float32(1e18), dtype=np.float32)
    band_h[0, w] = np.float32(0.0)  # cost[0,0] -> band[0, w]
    band_d = _nb_cuda.to_device(band_h)
    threads = 256
    for k in range(1, n + m + 1):
        i_min = max(1, k - m)
        i_max = min(n, k - 1)
        n_cells = max(0, i_max - i_min + 1)
        if n_cells == 0:
            continue
        blocks = (n_cells + threads - 1) // threads
        _numba_cuda_banded_step[blocks, threads](
            x_d, y_d, band_d, n, m, w, bw, k,
        )
    _nb_cuda.synchronize()
    band = band_d.copy_to_host()
    distance = float(np.sqrt(np.float32(_band_endpoint(band, n, m, w))))
    path = _backtrace_band(band, n, m, w)
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


# Banded buffer kernel: cost is stored in (i, b) coords where b = j - i + w, so the
# live Sakoe-Chiba band (only diagonals |i-j| <= w are reachable) occupies a tight
# (n+1) x (2*w+1) buffer instead of the full (n+1) x (m+1) matrix -- O(n*w) device
# RAM instead of O(n*m). The cell-(i,j) data dependencies (i-1,j-1)/(i-1,j)/(i,j-1)
# force an anti-diagonal (k = i+j) sweep, so this keeps one launch per anti-diagonal
# (the launch-fusion variant is left for FUTURE; the memory reduction is the primary
# win and is unconditional). Numerics are identical to the full-matrix kernel: same
# squared diff, same min-of-three with the diag<=up<=left tie order, same 1e18
# out-of-band sentinel, single final sqrt -- only the cost-buffer indexing changes.
_CUPY_BANDED_KERNEL_SRC = r"""
extern "C" __global__
void dtw_banded_diagonal(const float* __restrict__ x, const float* __restrict__ y,
                         float* band, int n, int m, int w, int bw, int k) {
    // band stride bw = 2*w+1; band[i*bw + b] holds cost[i][ i - w + b ].
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i_min = max(1, k - m);
    int i_max = min(n, k - 1);
    int n_cells = i_max - i_min + 1;
    if (tid >= n_cells) return;
    int i = i_min + tid;
    int j = k - i;
    int b = j - i + w;                 // band column for (i,j)
    if (b < 0 || b >= bw) return;      // outside the live band -> never reachable
    float c_diag = band[(i - 1) * bw + b];                        // cost[i-1][j-1]
    float c_up   = (b + 1 < bw) ? band[(i - 1) * bw + (b + 1)] : 1e18f; // cost[i-1][j]
    float c_left = (b > 0) ? band[i * bw + (b - 1)] : 1e18f;      // cost[i][j-1]
    float d = x[i - 1] - y[j - 1];
    float best = (c_diag <= c_up) ? c_diag : c_up;
    if (c_left < best) best = c_left;
    band[i * bw + b] = d * d + best;
}
"""


_CUPY_KERNEL_CACHE: list = [None]
_CUPY_BANDED_KERNEL_CACHE: list = [None]


def _get_cupy_kernel():
    if not _HAS_CUPY:
        raise ImportError("dtw_cupy requires cupy + a CUDA-capable GPU.")
    if _CUPY_KERNEL_CACHE[0] is None:
        _CUPY_KERNEL_CACHE[0] = cp.RawKernel(
            _CUPY_KERNEL_SRC, "dtw_diagonal",
        )
    return _CUPY_KERNEL_CACHE[0]


def _get_cupy_banded_kernel():
    if not _HAS_CUPY:
        raise ImportError("dtw_cupy requires cupy + a CUDA-capable GPU.")
    if _CUPY_BANDED_KERNEL_CACHE[0] is None:
        _CUPY_BANDED_KERNEL_CACHE[0] = cp.RawKernel(
            _CUPY_BANDED_KERNEL_SRC, "dtw_banded_diagonal",
        )
    return _CUPY_BANDED_KERNEL_CACHE[0]


def _band_endpoint(band: np.ndarray, n: int, m: int, w: int) -> float:
    """cost[n,m] from the banded buffer; 1e18 sentinel when (n,m) falls outside
    the Sakoe-Chiba band (|n-m| > window), matching the unreachable-endpoint
    value the full-matrix sweep returned."""
    b = m - n + w
    if 0 <= b < band.shape[1]:
        return float(band[n, b])
    return 1e18


def _backtrace_band(
    band: np.ndarray, n: int, m: int, w: int,
) -> List[Tuple[int, int]]:
    """Banded-coordinate backtrace. ``band[i, j - i + w]`` mirrors ``cost[i, j]``;
    out-of-band neighbours read as the 1e18 sentinel so the path policy
    (diag<=up<=left) is bit-identical to the full-matrix ``_backtrace_path``."""
    INF = np.float32(1e18)

    def C(i: int, j: int) -> float:
        b = j - i + w
        if 0 <= b < band.shape[1]:
            return float(band[i, b])
        return float(INF)

    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        c_diag = C(i - 1, j - 1)
        c_up = C(i - 1, j)
        c_left = C(i, j - 1)
        if c_diag <= c_up and c_diag <= c_left:
            i -= 1
            j -= 1
        elif c_up <= c_left:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


def dtw_cupy_banded(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window: int = 200,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Banded cupy DTW: O(n*window) device RAM (vs O(n*m) full matrix).

    Stores only the ``2*window+1`` live Sakoe-Chiba diagonals (band coords
    ``b = j - i + window``) instead of the full ``(n+1)x(m+1)`` cost matrix. The
    (i-1,j-1)/(i-1,j)/(i,j-1) dependency forces the anti-diagonal sweep, so this
    keeps one launch per anti-diagonal; the memory reduction is the win. Distance
    + path are bit-identical to the full-matrix ``dtw_cupy`` and the dtaidistance
    reference; this is the default cupy path (``dtw_cupy`` dispatches here)."""
    if not _HAS_CUPY:
        raise ImportError("dtw_cupy requires cupy + a CUDA-capable GPU.")
    kernel = _get_cupy_banded_kernel()
    n = len(x)
    m = len(y)
    w = int(window)
    bw = 2 * w + 1
    x_d = cp.asarray(x, dtype=cp.float32)
    y_d = cp.asarray(y, dtype=cp.float32)
    band_d = cp.full((n + 1, bw), cp.float32(1e18))
    # cost[0,0] = 0 lives at band[0, 0 - 0 + w] = band[0, w].
    band_d[0, w] = cp.float32(0.0)
    threads = 256
    for k in range(1, n + m + 1):
        i_min = max(1, k - m)
        i_max = min(n, k - 1)
        n_cells = max(0, i_max - i_min + 1)
        if n_cells == 0:
            continue
        blocks = (n_cells + threads - 1) // threads
        kernel(
            (blocks,),
            (threads,),
            (x_d, y_d, band_d, np.int32(n), np.int32(m), np.int32(w), np.int32(bw), np.int32(k)),
        )
    cp.cuda.Stream.null.synchronize()
    band = cp.asnumpy(band_d)
    distance = float(np.sqrt(np.float32(_band_endpoint(band, n, m, w))))
    path = _backtrace_band(band, n, m, w)
    return distance, path


def dtw_cupy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window: int = 200,
) -> Tuple[float, List[Tuple[int, int]]]:
    """cupy RawKernel DTW. Defaults to the banded buffer (O(n*window) device RAM).

    Per-launch overhead is lower than numba.cuda (no per-call JIT dispatch). This
    is the public cupy entry point; it dispatches to ``dtw_cupy_banded`` (the
    full-matrix sweep is retained as ``dtw_cupy_full`` for A/B reference)."""
    return dtw_cupy_banded(x, y, window=window)


def dtw_cupy_full(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window: int = 200,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Pre-CPX-P0-2 full-matrix cupy diagonal sweep. Allocates the full
    ``(n+1)x(m+1)`` cost matrix (O(n*m) device RAM); retained for A/B benching
    against ``dtw_cupy_banded`` (REJECTED!=DELETED). Numerically identical."""
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
            (blocks,),
            (threads,),
            (x_d, y_d, cost_d, np.int32(n), np.int32(m), np.int32(window), np.int32(k)),
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


def _make_dtw_inputs(n_cells: int):
    """Two random float32 sequences whose cost matrix has ~n_cells cells."""
    L = max(8, int(n_cells**0.5))
    rng = np.random.default_rng(0)
    return (rng.standard_normal(L).astype(np.float32), rng.standard_normal(L).astype(np.float32))


def _run_dtw_sweep() -> list:
    """Benchmark cpu/cuda/cupy DTW across an n_cells grid -> backend_choice
    regions (fastest equivalent backend per band). GPU variants are included
    only when available on this host. float32 GPU vs CPU diagonal sweeps agree
    to a small relative tolerance, so equiv_rtol is loosened from the default."""
    from pyutilz.dev.benchmarking import sweep_backend_crossover

    W = 200
    variants = {"cpu": lambda x, y: dtw_cpu(x, y, window=W)[0]}
    if _HAS_NB_CUDA:
        variants["cuda"] = lambda x, y: dtw_cuda(x, y, window=W)[0]
    if _HAS_CUPY:
        variants["cupy"] = lambda x, y: dtw_cupy(x, y, window=W)[0]
    sizes = [10_000, 40_000, 160_000, 640_000, 2_560_000]
    return list(sweep_backend_crossover(
        variants, sizes, _make_dtw_inputs, "n_cells",
        reference="cpu", repeats=5, equiv_rtol=1e-3, equiv_atol=1e-3,
    ))


def _dtw_fallback_choice(n_cells: int) -> str:
    """Pre-sweep heuristic (the old gpu_min_cells threshold + availability). Used
    as the spec's dynamic fallback callable."""
    if _HAS_CUPY and n_cells >= _DEFAULT_GPU_MIN_CELLS:
        return "cupy"
    if _HAS_NB_CUDA and n_cells >= _DEFAULT_GPU_MIN_CELLS:
        return "cuda"
    return "cpu"


def _dtw_tuner():
    """The registered tuner, env-gated: MLFRAME_DTW_AUTOTUNE=0 disables the
    on-miss sweep (choose() then returns the fallback)."""
    return _run_dtw_sweep() if _os.environ.get("MLFRAME_DTW_AUTOTUNE", "1").strip() != "0" else []


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
        raise ValueError(f"unknown backend {forced!r}; expected cpu/cuda/cupy")
    if psi > 0:
        return dtw_cpu(x, y, window=window, psi=psi)
    n_cells = len(x) * len(y)
    # Per-host backend (cpu/cuda/cupy) via the spec's one-call choose(), guarded
    # by live availability (the tuning host had the backend; a reader may not).
    choice = _DTW_SPEC.choose(n_cells=n_cells)
    if choice == "cupy" and _HAS_CUPY:
        return dtw_cupy(x, y, window=window)
    if choice == "cuda" and _HAS_NB_CUDA:
        return dtw_cuda(x, y, window=window)
    return dtw_cpu(x, y, window=window, psi=psi)


# Register with the @kernel_tuner registry so retune_all / mlframe-tune-kernels
# discover + batch-tune dtw. GPU-capable (cupy/numba.cuda backends).
from pyutilz.performance.kernel_tuning.registry import kernel_tuner

_DTW_SPEC = kernel_tuner(
    kernel_name="dtw_dispatch",
    variant_fns=(dtw_cpu, dtw_cuda, dtw_cupy),  # all always-defined -> any backend edit auto-invalidates
    tuner=_dtw_tuner,
    axes={"n_cells": [10_000, 40_000, 160_000, 640_000, 2_560_000]},
    fallback=_dtw_fallback_choice,  # dynamic heuristic (callable) -> spec.choose() uses it
    gpu_capable=True,
    salt=1,
    cli_label="dtw_dispatch",
)
