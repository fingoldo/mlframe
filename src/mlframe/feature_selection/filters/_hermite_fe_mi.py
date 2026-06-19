"""Plug-in MI estimator helpers (numba + CUDA variants + dispatchers) carved out
of ``mlframe.feature_selection.filters.hermite_fe``.

Re-imported at the parent's module bottom so historical
``from mlframe.feature_selection.filters.hermite_fe import plugin_mi_classif_dispatch``
resolves transparently.
"""
from __future__ import annotations

import logging
import math
import os

import numpy as np

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco
    def prange(n):
        return range(n)

# Parent-resident numba kernel referenced by the @njit'd functions below.
# numba's ``@njit`` doesn't compile ``IMPORT_NAME`` bytecode, so this import
# MUST live at module top -- which creates a static
# ``hermite_fe -> _hermite_fe_mi -> hermite_fe`` cycle that is benign at
# runtime (parent defines ``_quantile_bin_njit`` at line 80, then re-imports
# this sibling at its bottom; ``_quantile_bin_njit`` is already bound when
# this line fires). Whitelisted in ``tests/test_meta/test_no_import_cycles.py``.
from .hermite_fe import _quantile_bin_njit  # noqa: E402

logger = logging.getLogger("mlframe.feature_selection.filters.hermite_fe")


@njit(cache=True, fastmath=True)
def _plugin_mi_classif_njit(x: np.ndarray, y: np.ndarray,
                              n_bins: int = 20) -> float:
    """Plug-in MI estimator for continuous x (1-D float64) and discrete y (1-D int64). Returns MI in nats.
    ~50x faster than sklearn for n<=10k, single-thread."""
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    n = x.shape[0]
    # Class axis spans [y_min, y_max]; labels may be negative / non-dense (a binned continuous target shifted below 0).
    # Sizing on max(y)+1 alone and indexing with the raw label underflows the histogram into out-of-bounds memory -> native AV.
    y_min = y[0]
    y_max = y[0]
    for i in range(1, n):
        if y[i] < y_min:
            y_min = y[i]
        if y[i] > y_max:
            y_max = y[i]
    n_classes = (y_max - y_min) + 1

    x_binned = _quantile_bin_njit(x, n_bins)

    hist_xy = np.zeros((n_bins, n_classes), dtype=np.int64)
    hist_x = np.zeros(n_bins, dtype=np.int64)
    hist_y = np.zeros(n_classes, dtype=np.int64)
    for i in range(n):
        b = x_binned[i]
        c = y[i] - y_min
        hist_xy[b, c] += 1
        hist_x[b] += 1
        hist_y[c] += 1

    log_n = math.log(n)
    mi = 0.0
    for b in range(n_bins):
        if hist_x[b] == 0:
            continue
        log_hx = math.log(hist_x[b])
        for c in range(n_classes):
            n_xy = hist_xy[b, c]
            if n_xy == 0 or hist_y[c] == 0:
                continue
            mi += (n_xy / n) * (math.log(n_xy) + log_n - log_hx - math.log(hist_y[c]))
    if mi < 0.0:
        mi = 0.0
    return mi

@njit(cache=True, fastmath=True)
def _plugin_mi_regression_njit(x: np.ndarray, y: np.ndarray,
                                 n_bins: int = 20) -> float:
    """Plug-in MI for continuous x (1-D) and continuous y (1-D). Bin both into n_bins equi-frequency bins, then plug-in estimator."""
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    n = x.shape[0]
    x_binned = _quantile_bin_njit(x, n_bins)
    y_binned = _quantile_bin_njit(y, n_bins)

    hist_xy = np.zeros((n_bins, n_bins), dtype=np.int64)
    hist_x = np.zeros(n_bins, dtype=np.int64)
    hist_y = np.zeros(n_bins, dtype=np.int64)
    for i in range(n):
        bx = x_binned[i]
        by = y_binned[i]
        hist_xy[bx, by] += 1
        hist_x[bx] += 1
        hist_y[by] += 1

    log_n = math.log(n)
    mi = 0.0
    for bx in range(n_bins):
        if hist_x[bx] == 0:
            continue
        log_hx = math.log(hist_x[bx])
        for by in range(n_bins):
            n_xy = hist_xy[bx, by]
            if n_xy == 0 or hist_y[by] == 0:
                continue
            mi += (n_xy / n) * (math.log(n_xy) + log_n - log_hx - math.log(hist_y[by]))
    if mi < 0.0:
        mi = 0.0
    return mi

@njit(cache=True, fastmath=True)
def _plugin_mi_from_binned_njit(x_binned: np.ndarray, y: np.ndarray,
                                  n_bins: int) -> float:
    """Plug-in MI given pre-binned x. Skips the ``np.argsort`` step inside
    :func:`_plugin_mi_classif_njit`; the caller does the argsort + bin
    assignment in pure numpy (which is ~1.6x faster than numba-wrapped
    ``np.argsort`` at n=1500, measured 2026-05-20 — numba's argsort
    dispatch eats ~70us out of 92us total).

    This is the kernel that numba is genuinely good at: tight nested
    histogram loops, log/log_n math, plug-in MI summation. Bench at
    n=1500, n_classes=3: ~10us per call, vs ~128us for the full
    ``_plugin_mi_classif_njit``.
    """
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    n = x_binned.shape[0]
    # Class axis spans [y_min, y_max]; labels may be negative / non-dense. See _plugin_mi_classif_njit for the AV this guards against.
    y_min = y[0]
    y_max = y[0]
    for i in range(1, n):
        if y[i] < y_min:
            y_min = y[i]
        if y[i] > y_max:
            y_max = y[i]
    n_classes = (y_max - y_min) + 1

    hist_xy = np.zeros((n_bins, n_classes), dtype=np.int64)
    hist_x = np.zeros(n_bins, dtype=np.int64)
    hist_y = np.zeros(n_classes, dtype=np.int64)
    for i in range(n):
        b = x_binned[i]
        c = y[i] - y_min
        hist_xy[b, c] += 1
        hist_x[b] += 1
        hist_y[c] += 1

    log_n = math.log(n)
    mi = 0.0
    for b in range(n_bins):
        if hist_x[b] == 0:
            continue
        log_hx = math.log(hist_x[b])
        for c in range(n_classes):
            n_xy = hist_xy[b, c]
            if n_xy == 0 or hist_y[c] == 0:
                continue
            mi += (n_xy / n) * (math.log(n_xy) + log_n - log_hx - math.log(hist_y[c]))
    if mi < 0.0:
        mi = 0.0
    return mi

def plugin_mi_classif_fast(x: np.ndarray, y: np.ndarray,
                            n_bins: int = 20) -> float:
    """Faster single-column plug-in MI: numpy argsort + njit histogram math.

    Measured 2026-05-20 at n=1500 (CMA-ES inner-loop scale):
    - ``_plugin_mi_classif_njit`` (all-in-numba):                    128us
    - ``plugin_mi_classif_fast`` (numpy argsort + njit histogram):  ~67us
    -> **~1.9x speedup** on the hottest path. Numerical result is
    bit-for-bit identical (same quantile-bin recipe, same plug-in MI
    formula).

    Usage scope: SINGLE-COLUMN (k=1) hot paths only. For batch (k>=5),
    the parallel ``_plugin_mi_classif_batch_njit`` (prange over columns)
    wins over the per-column-numpy-argsort loop here. Exposed publicly
    for ad-hoc callers that compute single-column MI in a tight loop
    (e.g. residualised baseline pairs); ``plugin_mi_classif_dispatch``
    does NOT auto-route here because it already passes batches through
    the prange path which beats this implementation at k>=5.
    """
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    from .hermite_fe import _quantile_bin_numpy
    x_binned = _quantile_bin_numpy(x, n_bins)
    return float(_plugin_mi_from_binned_njit(
        x_binned, np.asarray(y, dtype=np.int64), n_bins,
    ))

def _plugin_mi_classif_batch_cuda(X_cols: np.ndarray, y: np.ndarray,
                                  n_bins: int = 20) -> np.ndarray:
    """Cupy batch plug-in MI. Quantile-bins each column on GPU via
    ``cp.argsort`` -> rank-to-bin lookup, then computes joint histograms
    via a single ``cp.bincount`` across (col, bin, class) flat indices.
    Numerically equivalent to :func:`_plugin_mi_classif_batch_njit`
    up to fp64 round-off.
    """
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    import cupy as cp
    X_gpu = cp.asarray(X_cols, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.int64)
    n, k = X_gpu.shape
    if n == 0 or k == 0:
        return np.zeros(k, dtype=np.float64)
    # Class axis spans [y_min, y_max]; labels may be negative / non-dense. Offset by y_min so the bincount index never underflows. Mirrors the njit kernels.
    y_min = int(cp.min(y_gpu).item())
    y_gpu = y_gpu - y_min
    n_classes = int(cp.max(y_gpu).item()) + 1

    # Per-column quantile binning: argsort -> rank -> bin lookup.
    # bin_for_rank[r] = floor(r / (n / n_bins)) with the remainder
    # absorbed by the first ``rem`` bins (matches njit version exactly).
    # bench-attempt-rejected (2026-06-20): f32 SORT KEYS (argsort on X_gpu.astype(float32)) to halve the
    # sort bandwidth on this 69%-of-MI step. Measured NO win on GTX 1050 Ti (n=200k K=384: f64 6566ms vs
    # f32 6794ms = 0.97x) -- cupy's argsort does not radix-accelerate f32 over f64 here and the astype cast
    # offsets any bandwidth saving; accuracy was fine (Spearman 0.999995, argmax match) but there is no
    # speed, so the approximation is not worth it. The real sort win needs a custom radix-rank kernel
    # (roadmap), not a dtype swap. May differ on cards with a faster f32 radix; re-bench there.
    sort_idx = cp.argsort(X_gpu, axis=0)  # (n, k) int64
    base = n // n_bins
    rem = n - base * n_bins
    sizes = cp.full(n_bins, base, dtype=cp.int64)
    if rem > 0:
        sizes[:rem] += 1
    offsets = cp.empty(n_bins, dtype=cp.int64)
    offsets[0] = 0
    if n_bins > 1:
        offsets[1:] = cp.cumsum(sizes[:-1])
    ranks = cp.arange(n, dtype=cp.int64)
    bin_for_rank = cp.searchsorted(offsets, ranks, side="right") - 1  # (n,)

    # Scatter rank-to-row: X_binned[sort_idx[r, j], j] = bin_for_rank[r]
    X_binned = cp.empty((n, k), dtype=cp.int64)
    cols_idx = cp.broadcast_to(cp.arange(k, dtype=cp.int64)[None, :], (n, k))
    X_binned[sort_idx, cols_idx] = bin_for_rank[:, None]

    # Joint hist via single bincount on flat index (col, bin, class).
    j_idx = cp.broadcast_to(cp.arange(k, dtype=cp.int64)[None, :], (n, k))
    y_b = y_gpu[:, None]
    flat = (j_idx * n_bins + X_binned) * n_classes + y_b  # (n, k)
    hist_flat = cp.bincount(
        flat.ravel(), minlength=k * n_bins * n_classes,
    )
    hist_xyc = hist_flat.reshape(k, n_bins, n_classes).astype(cp.float64)
    hist_x = hist_xyc.sum(axis=2)  # (k, n_bins)
    hist_y = hist_xyc.sum(axis=1)  # (k, n_classes); same across cols but kept per-col for cleanliness

    # MI sum vectorised across cells. Cells with zero count contribute 0
    # (same as the njit ``if n_xy == 0: continue`` short-circuit).
    log_n = math.log(n)
    mask = hist_xyc > 0
    safe_xyc = cp.where(mask, hist_xyc, 1.0)
    safe_x = cp.where(hist_x > 0, hist_x, 1.0)
    safe_y = cp.where(hist_y > 0, hist_y, 1.0)
    term = (hist_xyc / n) * (
        cp.log(safe_xyc) + log_n
        - cp.log(safe_x)[:, :, None] - cp.log(safe_y)[:, None, :]
    )
    mi = cp.where(mask, term, 0.0).sum(axis=(1, 2))  # (k,)
    mi = cp.maximum(mi, 0.0)
    return cp.asnumpy(mi)

def plugin_mi_classif_dispatch(x: np.ndarray, y: np.ndarray,
                                n_bins: int = 20) -> float:
    """Single-column plug-in MI for continuous x and discrete y.

    Routes to :func:`_plugin_mi_classif_njit` (CPU) or
    :func:`_plugin_mi_classif_cuda` (GPU) via the kernel tuning cache
    (per-host measurement-backed). Override via ``MLFRAME_MI_BACKEND``
    env var (``njit`` | ``cuda``) to force a specific backend.
    """
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    from .hermite_fe import _CUDA_AVAILABLE, _plugin_mi_classif_cuda
    from ._gpu_policy import gpu_globally_disabled
    forced = os.environ.get("MLFRAME_MI_BACKEND", "")
    n = x.shape[0]
    if forced == "njit":
        return float(_plugin_mi_classif_njit(x, y, n_bins))
    if forced == "cuda":
        if _CUDA_AVAILABLE:
            return _plugin_mi_classif_cuda(x, y, n_bins)
        return float(_plugin_mi_classif_njit(x, y, n_bins))
    # Consult the kernel tuning cache. Fallback to njit when cuda unavailable OR the global GPU
    # off-switch is set (cupy ignores MLFRAME_DISABLE_GPU / CUDA_VISIBLE_DEVICES="" on its own).
    if not _CUDA_AVAILABLE or gpu_globally_disabled():
        return float(_plugin_mi_classif_njit(x, y, n_bins))
    # GROUND-TRUTH OVERRIDE: default njit (see the batch dispatch below for the full rationale -- even the
    # concurrency-aware tuner still under-counts this path by ~5x vs the end-to-end fit). MLFRAME_MI_BACKEND
    # =cuda forces GPU.
    return float(_plugin_mi_classif_njit(x, y, n_bins))

def plugin_mi_classif_batch_dispatch(X_cols: np.ndarray, y: np.ndarray,
                                      n_bins: int = 20) -> np.ndarray:
    """Batch plug-in MI per column of ``X_cols`` against discrete ``y``.

    Routes to :func:`_plugin_mi_classif_batch_njit` (prange CPU) or
    :func:`_plugin_mi_classif_batch_cuda` (GPU) via the kernel tuning
    cache. Override via ``MLFRAME_MI_BACKEND`` env var.
    """
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    from .hermite_fe import _CUDA_AVAILABLE, _plugin_mi_classif_batch_njit
    from ._gpu_policy import gpu_globally_disabled
    forced = os.environ.get("MLFRAME_MI_BACKEND", "")
    n, k = X_cols.shape
    if forced == "njit":
        return _plugin_mi_classif_batch_njit(X_cols, y, n_bins)
    # Honor the global GPU off-switch (MLFRAME_DISABLE_GPU / CUDA_VISIBLE_DEVICES="") -- cupy's own
    # device detection ignores both, so without this a CPU-only / weak-GPU run still routes this HOT
    # batched orth-FE MI to cupy (37% of a 300k fit: cupy argsort + GPU-sync sleep, CPU idle).
    # An explicit MLFRAME_MI_BACKEND=cuda still wins (handled above / below).
    if forced == "cuda":
        if _CUDA_AVAILABLE:
            return _plugin_mi_classif_batch_cuda(X_cols, y, n_bins)
        return _plugin_mi_classif_batch_njit(X_cols, y, n_bins)
    if not _CUDA_AVAILABLE or gpu_globally_disabled():
        return _plugin_mi_classif_batch_njit(X_cols, y, n_bins)
    # GROUND-TRUTH OVERRIDE: this batched FE-MI path defaults to njit regardless of the per-call tuner.
    # The tuner sweep was upgraded to measure BOTH backends under realistic joblib worker-thread CONTENTION
    # (_run_sweep_mi_classif_dispatch) -- and that DID surface a 5-7x cuda contention penalty (n=100k k=20:
    # solo 34ms vs contended 135ms). But even the contended microbench still under-counts this path by ~5x
    # vs the real fit: it (a) reuses a warm GPU buffer while production allocates a FRESH engineered
    # candidate array every call (cudaMalloc churn + fresh H2D/D2H), (b) runs MI in isolation while
    # production interleaves the GPU-CMI redundancy kernel on the same device, and (c) inflates the
    # contended-njit baseline via prange oversubscription absent from the real FE call pattern. Net: the
    # tuner says "cuda" at n>=100k yet the ground-truth end-to-end fit is njit 3x faster (114 vs 368s,
    # 1.6 vs 5.0 GB peak, byte-identical selection on the canonical 5-feature/n=100k fit). An isolated MI
    # microbench cannot model the full pipeline, so we trust the end-to-end measurement. The principled way
    # to actually WIN on GPU here is to make FE candidates GPU-RESIDENT (eliminating the per-call H2D/D2H
    # the microbench omits) -- the larger matrix-native FE replatform, tracked separately.
    # MLFRAME_MI_BACKEND=cuda forces GPU (handled above) for a caller whose own end-to-end profile shows it.
    return _plugin_mi_classif_batch_njit(X_cols, y, n_bins)

_CUDA_KERNELS: dict = {}


def _ensure_cuda_kernels():
    """Lazy-compile CUDA RawKernels on first use."""
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    from .hermite_fe import _CUDA_AVAILABLE
    global _CUDA_KERNELS
    if _CUDA_KERNELS or not _CUDA_AVAILABLE:
        return
    import cupy as cp
    _CUDA_KERNELS["hermite"] = cp.RawKernel(r"""
extern "C" __global__
void hermeval_kernel(const double* __restrict__ x,
                     const double* __restrict__ c,
                     int nc, int n,
                     double* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double xi = x[i];
    if (nc == 0) { out[i] = 0.0; return; }
    if (nc == 1) { out[i] = c[0]; return; }
    double p_prev = 1.0, p_curr = xi;
    double s = c[0] + c[1] * p_curr;
    for (int k = 2; k < nc; ++k) {
        double p_next = xi * p_curr - (double)(k - 1) * p_prev;
        s += c[k] * p_next;
        p_prev = p_curr; p_curr = p_next;
    }
    out[i] = s;
}
""", "hermeval_kernel")
    _CUDA_KERNELS["legendre"] = cp.RawKernel(r"""
extern "C" __global__
void legval_kernel(const double* __restrict__ x,
                    const double* __restrict__ c,
                    int nc, int n,
                    double* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double xi = x[i];
    if (nc == 0) { out[i] = 0.0; return; }
    if (nc == 1) { out[i] = c[0]; return; }
    double p_prev = 1.0, p_curr = xi;
    double s = c[0] + c[1] * p_curr;
    for (int k = 2; k < nc; ++k) {
        double inv_k = 1.0 / (double)k;
        double p_next = ((double)(2 * k - 1) * xi * p_curr - (double)(k - 1) * p_prev) * inv_k;
        s += c[k] * p_next;
        p_prev = p_curr; p_curr = p_next;
    }
    out[i] = s;
}
""", "legval_kernel")
    _CUDA_KERNELS["chebyshev"] = cp.RawKernel(r"""
extern "C" __global__
void chebval_kernel(const double* __restrict__ x,
                     const double* __restrict__ c,
                     int nc, int n,
                     double* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double xi = x[i];
    if (nc == 0) { out[i] = 0.0; return; }
    if (nc == 1) { out[i] = c[0]; return; }
    double p_prev = 1.0, p_curr = xi;
    double s = c[0] + c[1] * p_curr;
    for (int k = 2; k < nc; ++k) {
        double p_next = 2.0 * xi * p_curr - p_prev;
        s += c[k] * p_next;
        p_prev = p_curr; p_curr = p_next;
    }
    out[i] = s;
}
""", "chebval_kernel")
    _CUDA_KERNELS["laguerre"] = cp.RawKernel(r"""
extern "C" __global__
void lagval_kernel(const double* __restrict__ x,
                    const double* __restrict__ c,
                    int nc, int n,
                    double* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double xi = x[i];
    if (nc == 0) { out[i] = 0.0; return; }
    if (nc == 1) { out[i] = c[0]; return; }
    double p_prev = 1.0, p_curr = 1.0 - xi;
    double s = c[0] + c[1] * p_curr;
    for (int k = 2; k < nc; ++k) {
        double inv_k = 1.0 / (double)k;
        double p_next = (((double)(2 * k - 1) - xi) * p_curr - (double)(k - 1) * p_prev) * inv_k;
        s += c[k] * p_next;
        p_prev = p_curr; p_curr = p_next;
    }
    out[i] = s;
}
""", "lagval_kernel")
