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
    if n == 0:
        return 0.0  # empty column (fully-filtered subsample / empty finite mask): y[0] below would OOB-crash
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
    if n == 0:
        return 0.0  # empty column: log(n) and the binning below are undefined on n=0
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
    if n == 0:
        return 0.0  # empty column: the y[0] below would OOB-crash (numba native access violation)
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
    return _plugin_mi_classif_batch_cuda_resident(X_gpu, y_gpu, n_bins)


def _plugin_mi_classif_batch_cuda_resident(X_gpu, y_gpu, n_bins: int = 20, *, y_min=None, n_classes=None):
    """MATRIX-NATIVE plug-in MI on ALREADY-RESIDENT cupy arrays -- the H2D-FREE core of
    :func:`_plugin_mi_classif_batch_cuda`. ``X_gpu`` is an (n, k) cupy float64 candidate
    matrix, ``y_gpu`` an (n,) cupy integer label vector, BOTH already on the device. This
    is the entry point a matrix-native caller uses when it built/holds its candidate
    columns on the GPU -- it skips the per-call ``cp.asarray`` H2D that the dispatcher's
    ground-truth note records as the 2x-slowdown cause (measured: forcing the H2D path on
    is 52s -> 105s), so MI runs on the device with NO transfer. Math is identical to the
    host-input variant (same percentile-edge equi-frequency binning + plug-in MI).
    Returns a host (k,) float64 array of MI values."""
    import cupy as cp
    n, k = X_gpu.shape
    if n == 0 or k == 0:
        return np.zeros(k, dtype=np.float64)
    if X_gpu.dtype != cp.float64:
        X_gpu = X_gpu.astype(cp.float64)
    if y_gpu.dtype != cp.int64:
        y_gpu = y_gpu.astype(cp.int64)
    # Class axis spans [y_min, y_max]; labels may be negative / non-dense. Offset by y_min so the bincount
    # index never underflows. y's min/max are a fit-CONSTANT (the same label vector across every pair x
    # chunk), so when the caller passes them (computed ONCE for the whole pair sweep) skip the per-call
    # cp.min/cp.max + scalar D2H -- nsys (2026-06-22) showed this exact line is the #1 source of the 71k
    # cp.max reductions and a huge slice of the 100k tiny D2H in the per-pair x per-chunk MRMR loop.
    # Bit-identical: y is invariant, so the offset + bincount layout are unchanged.
    if y_min is None or n_classes is None:
        _ymm = cp.asnumpy(cp.stack((cp.min(y_gpu), cp.max(y_gpu))))
        y_min = int(_ymm[0])
        n_classes = int(_ymm[1]) - y_min + 1   # == max(orig)-y_min+1, i.e. max of the shifted y plus 1
    y_gpu = y_gpu - y_min

    # Per-column quantile binning via cp.percentile EDGES + searchsorted (2026-06-20). Replaced the
    # argsort -> rank -> uncoalesced-scatter path (the dominant ~69%-of-MI cost). Measured 7.84x faster
    # (n=200k K=384: 7781ms -> 993ms) with the SAME feature ranking (Spearman 1.0, argmax match). Per-
    # column edges depend ONLY on that column, so the binning is chunk-INVARIANT (verified on CPU:
    # test_percentile_binning_chunk_invariant). TRADE-OFF: edge-based equi-frequency vs the njit MI's
    # rank-based -> not bit-identical to njit at ties, an approved trade (features unchanged; MRMR
    # selection-equivalence tests still pass). (bench-rejected f32 sort keys: 0.97x, no win.)
    qs = cp.linspace(0.0, 100.0, n_bins + 1)
    if k == 1:
        # CUPY BUG GUARD (2026-06-20): cp.percentile(X, axis=0) returns WRONG edges for a single-column
        # (n, 1) array (verified maxdiff ~23 vs numpy; multi-column is exact). A k==1 chunk occurs in
        # production whenever the last VRAM chunk holds one candidate, so this would silently corrupt that
        # column's binning. Ravel to 1D, where cp.percentile is correct, then restore the (n_bins+1, 1) shape.
        edges = cp.percentile(X_gpu.ravel(), qs).reshape(-1, 1)  # (n_bins+1, 1)
    else:
        edges = cp.percentile(X_gpu, qs, axis=0)  # (n_bins+1, k) per-column quantile edges
    # Fused single-launch binning: the per-column cp.searchsorted(side='right') loop fired k-1 extra kernel
    # launches + k int64 temporaries. _searchsorted_codes is the already-validated drop-in (bit-identical:
    # code = #(interior edges <= value), edges f64-promoted exactly as cp.searchsorted does, with its own
    # per-column fallback on kernel failure). Lazy import keeps the module-import graph acyclic (cupy-resident
    # _gpu_resident_fe never top-level-imports this module). astype(int64) only widens the int32 codes
    # (range [0, n_bins-1]) for the flat-index math below -- value-preserving, no truncation.
    from ._gpu_resident_fe import _searchsorted_codes
    X_binned = _searchsorted_codes(X_gpu, edges[1:-1]).astype(cp.int64, copy=False)

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
    # bench-attempt-rejected (2026-06-20, GTX 1050 Ti, "eliminate ALL f64 in the GPU MI math"): the
    # histogram freqs/log/entropy-sum here were cast to float32 to chase the consumer-GPU f64 1/32-rate
    # penalty. Measured NO win: math-only f64->f32 = 1.00-1.01x at n=100k-300k K=384 (the math runs on the
    # tiny (k,nbins,nclasses) integer-derived hist -- it is NOT where the f64 penalty bites). The penalty
    # is entirely in the BINNING sort (cp.asarray(...f64) + cp.percentile + cp.searchsorted): bin f64->f32
    # alone = 1.33-1.51x, and math f32 on top of that adds 0.00x more. So the MI MATH stays f64 (free
    # bit-fidelity to the njit reference); the binning-dtype lever is the real win and is owned by the
    # MLFRAME_FE_GPU_BINNING_DTYPE path. Full-chain f32 also FLIPPED the argmax at n=300k among the
    # equivalent a**2/b spellings (div(id,sqrt) -> mul(sqr,reciproc), a tie-cluster reorder) -- selection
    # instability for no math speedup. (proto D:/Temp/_bench_mi_dtype*.py)
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
