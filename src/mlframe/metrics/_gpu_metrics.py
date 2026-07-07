"""GPU-accelerated batch metrics for ``mlframe.metrics.core``.

Split out from ``core.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; every moved symbol is
re-exported from ``core`` so existing
``from mlframe.metrics.core import compute_batch_aucs`` (and the other
moved names) imports continue to work.

What lives here:
  - Constants + sentinels: ``_GPU_BATCH_THRESHOLD_N`` /
    ``_GPU_BATCH_THRESHOLD_M``, ``_GPU_AVAILABLE``,
    ``_NUMBA_CUDA_AVAILABLE``, ``_CUPY_SSE_PER_COL``,
    ``_NUMBA_RMSE_KERNEL``.
  - Probes: ``set_gpu_thresholds``, ``is_gpu_metrics_available``,
    ``_is_numba_cuda_available``, ``_require_cupy``.
  - Kernel factories: ``_get_cupy_sse_kernel``, ``_get_numba_rmse_kernel``.
  - GPU primitives: ``gpu_multiple_rmse_scores``,
    ``gpu_multiple_roc_auc_scores``, ``gpu_multiple_pr_auc_scores``.
  - CPU/GPU auto-dispatchers: ``_normalize_scores_2d``,
    ``compute_batch_rmse``, ``compute_batch_aucs``, ``_resolve_backend``.

GPU batch metrics vectorize RMSE / ROC-AUC / PR-AUC across multiple
prediction columns. cupy is optional - the helpers raise a clear
ImportError when missing and callers fall back to CPU (see
``compute_batch_aucs`` / ``compute_batch_rmse`` dispatchers). Below
N=100k the CPU/numba path wins (kernel-launch and host->device transfer
dominate sub-ms workloads).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Default crossover thresholds. Tunable via ``set_gpu_thresholds(...)``.
#
# ``_GPU_BATCH_THRESHOLD_M = 5`` (was 1) so binary-classification single-target callers
# (M=1) stay on the numba CPU path. The GPU AUC kernels carry ~1-3 s of cupy compile +
# host<->device transfer overhead per call and contain a Python loop over M columns
# inside ``gpu_multiple_roc_auc_scores``; below M=5 the per-column ``fast_aucs`` numba
# path wins decisively (~10x at N=1M, M=1 in fuzz iter#194). The
# ``gpu_multiple_*_auc_scores`` docstrings document "Use when N >= 100k AND M >= 5";
# the threshold previously contradicted that guidance, dispatching GPU for every binary
# classifier and inflating ``compute_batch_aucs`` to ~32 s of the ~55 s suite wall on a
# 1M-row binary classification run.
_GPU_BATCH_THRESHOLD_N: int = 100_000
_GPU_BATCH_THRESHOLD_M: int = 5

# Sentinels: None = unchecked, True/False = cached result.
_GPU_AVAILABLE: Optional[bool] = None

# numba.cuda used by the RMSE fast-path; cupy ReductionKernel stays as fallback when
# numba CUDA isn't usable (no toolkit, mismatched runtime, etc.).
_NUMBA_CUDA_AVAILABLE: Optional[bool] = None


def set_gpu_thresholds(*, n: Optional[int] = None, m: Optional[int] = None) -> None:
    """Override the (N, M) thresholds that gate GPU dispatch in ``compute_batch_aucs`` / ``compute_batch_rmse``. Pass ``None`` to leave a threshold unchanged. For tests / benchmarking only - production callers should let auto-dispatch handle this."""
    global _GPU_BATCH_THRESHOLD_N, _GPU_BATCH_THRESHOLD_M
    if n is not None:
        _GPU_BATCH_THRESHOLD_N = int(n)
    if m is not None:
        _GPU_BATCH_THRESHOLD_M = int(m)


def is_gpu_metrics_available() -> bool:
    """True iff cupy is importable AND a CUDA device is visible AND a small reduction kernel actually compiles via NVRTC.

    The NVRTC compile probe is essential: cupy can import cleanly and report
    devices on hosts with renamed/mismatched cublas/nvrtc DLLs, then crash
    later inside ``_SimpleReductionKernel._get_function`` with a RecursionError
    when its softlink-retry path re-enters itself (observed on a full-suite run
    after renaming cublas64_11.dll to unblock torch).

    Result is cached after the first call. ``except BaseException`` covers
    RecursionError too.
    """
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE
    try:
        import cupy as cp  # type: ignore
        if cp.cuda.runtime.getDeviceCount() < 1:
            _GPU_AVAILABLE = False
            return False
        # NVRTC compile probe - mirrors _utils.is_gpu_available().
        _ = cp.asarray([1.0], dtype=cp.float32).sum().item()
        _GPU_AVAILABLE = True
        return True
    except Exception:  # nosec B110 - best-effort/optional path, no module logger
        pass
    _GPU_AVAILABLE = False
    return False


def _is_numba_cuda_available() -> bool:
    """Probe numba.cuda once. Cached."""
    global _NUMBA_CUDA_AVAILABLE
    if _NUMBA_CUDA_AVAILABLE is not None:
        return _NUMBA_CUDA_AVAILABLE
    try:
        from numba import cuda  # type: ignore
        if cuda.is_available():
            _NUMBA_CUDA_AVAILABLE = True
            return True
    except Exception:  # nosec B110 - best-effort/optional path, no module logger
        pass
    _NUMBA_CUDA_AVAILABLE = False
    return False


def _require_cupy():
    """Lazy cupy import; raise ImportError with install hint if missing."""
    try:
        import cupy as cp  # type: ignore
        return cp
    except ImportError as e:
        raise ImportError(
            "GPU metrics require cupy. Install for your CUDA version, e.g. " "`pip install cupy-cuda12x` (CUDA 12) or `cupy-cuda11x` (CUDA 11)."
        ) from e


# Cached cupy ReductionKernel for fused (y-p)**2 sum-per-col, avoids materialising
# the (y-p)**2 intermediate. Bit-equivalent to numpy. Used as fallback when
# numba.cuda is unavailable.
_CUPY_SSE_PER_COL = None
_NUMBA_RMSE_KERNEL = None


def _get_cupy_sse_kernel():
    """Build (or return cached) cupy ReductionKernel for SSE per column."""
    global _CUPY_SSE_PER_COL
    if _CUPY_SSE_PER_COL is None:
        cp = _require_cupy()
        _CUPY_SSE_PER_COL = cp.ReductionKernel(
            in_params="float64 y, float64 p",
            out_params="float64 z",
            map_expr="(y - p) * (y - p)",
            reduce_expr="a + b",
            post_map_expr="z = a",
            identity="0.0",
            name="mlframe_sse_per_col",
        )
    return _CUPY_SSE_PER_COL


def _get_numba_rmse_kernel():
    """Build (or return cached) numba.cuda kernel that computes per-block, per-column SSE via atomic.add. Final reduction + sqrt happens in cupy."""
    global _NUMBA_RMSE_KERNEL
    if _NUMBA_RMSE_KERNEL is None:
        from numba import cuda  # type: ignore

        @cuda.jit
        def _rmse_partial_sum(y, p, partial, N, M):
            # y: (N,) broadcast across M cols; p: (N, M); partial: (grid_x, M) per-block accumulator.
            j = cuda.blockIdx.y
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if j >= M or i >= N:
                return
            d = y[i] - p[i, j]
            cuda.atomic.add(partial, (cuda.blockIdx.x, j), d * d)

        _NUMBA_RMSE_KERNEL = _rmse_partial_sum
    return _NUMBA_RMSE_KERNEL


def gpu_multiple_rmse_scores(actual, predicted):
    """Vectorized RMSE across columns on GPU.

    Computes ``sqrt(mean((actual - predicted)**2, axis=0))`` for an ``(N,)``
    or ``(N, M)`` ``actual`` and an ``(N, M)`` ``predicted``, returning
    ``(M,)`` RMSEs as a cupy array.

    Backend selection (auto):
      - ``numba.cuda`` kernel when available: fused (subtract, square,
        atomic-add) in one pass; per-block partials finalised by cupy.sum.
        Tiny fp jitter (~1e-15) from non-deterministic atomic-add accumulation
        order.
      - ``cupy.ReductionKernel`` fallback: fuses subtract+square+reduce into
        one kernel pass, avoiding the ``(N, M)`` fp64 intermediate. Bit-
        equivalent to numpy.

    Use when N >= 100k. Below that the numpy path is faster. For automatic
    dispatch see ``compute_batch_rmse``.

    Inputs may be cupy or numpy arrays. Output is a cupy array; call
    ``cp.asnumpy(...)`` to bring back to host.
    """
    cp = _require_cupy()
    actual = cp.asarray(actual, dtype=cp.float64)
    predicted = cp.asarray(predicted, dtype=cp.float64)
    if predicted.ndim == 1:
        predicted = predicted[:, cp.newaxis]

    # numba.cuda fast-path expects 1-D y for broadcast; fall back to ReductionKernel for 2-D actual.
    can_use_numba = _is_numba_cuda_available() and actual.ndim == 1
    N = predicted.shape[0]
    M = predicted.shape[1]

    if can_use_numba:
        from numba import cuda
        kernel = _get_numba_rmse_kernel()
        # ``BLOCK_N`` was hardcoded at 256 (Ampere-tuned default; wrong on
        # Pascal at 128 and Hopper at 512+). Per
        # ``feedback_use_kernel_tuning_cache_for_gpu`` route through
        # ``kernel_tuning_cache`` so the dispatcher adapts to live HW. Fall
        # back to 256 when the cache helper is unavailable or no entry
        # exists for the live HW yet.
        try:
            from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
            _cache = KernelTuningCache.load_or_create()
            _choice = _cache.lookup(
                "rmse_partial_sum", n_samples=int(N), n_cols=int(M),
            )
            BLOCK_N = int(_choice["block_n"]) if _choice and "block_n" in _choice else 256
        except Exception:
            BLOCK_N = 256
        grid_x = (N + BLOCK_N - 1) // BLOCK_N
        partial = cp.zeros((grid_x, M), dtype=cp.float64)
        kernel[(grid_x, M), BLOCK_N](
            cuda.as_cuda_array(actual),
            cuda.as_cuda_array(predicted),
            cuda.as_cuda_array(partial),
            N, M,
        )
        return cp.sqrt(cp.sum(partial, axis=0) / N)

    if actual.ndim == 1:
        actual = actual[:, cp.newaxis]
    sse = _get_cupy_sse_kernel()(actual, predicted, axis=0)
    return cp.sqrt(sse / N)


def gpu_multiple_roc_auc_scores(actual, predicted):
    """Vectorized ROC AUC across columns on GPU (cupy), tie-correct.

    Computes ROC AUC via Mann-Whitney U with fractional (average) ranks on
    tied scores, matching ``sklearn.metrics.roc_auc_score`` and
    ``mlframe.metrics.core.fast_roc_auc`` bit-for-bit on continuous data and
    within fp64 noise on tied data.

    Use when N >= 100k AND M >= 5. Below N=100k the per-column numba loop is
    faster. For automatic dispatch see ``compute_batch_aucs``.

    Args:
        actual: ``(N,)`` binary 0/1 labels (numpy or cupy).
        predicted: ``(N, M)`` per-column scores (numpy or cupy). Higher score
            = more likely positive.

    Returns:
        ``(M,)`` cupy array of AUCs. NaN for columns where the label column
        is degenerate (all 0 or all 1).

    Notes:
        The naive ``argsort(argsort(x))+1`` Mann-Whitney trick gives strict
        (not fractional) ranks on tied scores, producing ~1e-5 error vs
        sklearn on calibrated probability bins. This impl replaces the second
        argsort with a cumsum-based fractional-rank computation: O(N) vs
        O(N log N).
    """
    cp = _require_cupy()
    actual = cp.asarray(actual)
    predicted = cp.asarray(predicted)
    if predicted.ndim == 1:
        predicted = predicted[:, cp.newaxis]

    N, M = predicted.shape
    n_pos = cp.sum(actual)
    n_neg = N - n_pos

    # Per-column fractional ranks: equal-value runs get the average of the strict ranks they occupy (e.g. three 0.5's at positions 4,5,6 all get rank 5).
    order = cp.argsort(predicted, axis=0)
    ranks = cp.empty((N, M), dtype=cp.float64)
    for j in range(M):
        col = predicted[:, j]
        col_sorted = col[order[:, j]]
        # ``is_new[i]`` True iff position i starts a new run of equal values.
        is_new = cp.concatenate([
            cp.array([True]),
            col_sorted[1:] != col_sorted[:-1],
        ])
        run_id = cp.cumsum(is_new) - 1
        run_starts = cp.where(is_new)[0]
        run_ends = cp.concatenate([run_starts[1:], cp.array([N])])
        run_sizes = run_ends - run_starts
        # Average strict rank per run = run_starts + (size+1)/2 (strict ranks are 1-based, run_starts 0-based; offset cancels).
        avg_rank_per_run = run_starts.astype(cp.float64) + (run_sizes + 1) / 2.0
        avg_ranks_sorted = avg_rank_per_run[run_id]
        col_ranks = cp.empty(N, dtype=cp.float64)
        col_ranks[order[:, j]] = avg_ranks_sorted
        ranks[:, j] = col_ranks

    aucs = (cp.sum(ranks[actual == 1, :], axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return aucs


def gpu_multiple_pr_auc_scores(actual, predicted):
    """Vectorized PR AUC (Average Precision) across columns on GPU.

    Computes Riemann-sum AP matching ``sklearn.metrics.average_precision_score``
    and ``mlframe.metrics.core.fast_aucs`` bit-for-bit (verified <= 5.55e-17 on
    continuous and tied data).

    Algorithm: descending-sort by score, build cumulative-TP / FP arrays,
    detect tie-run boundaries (one threshold per run), sum
    ``(R_n - R_{n-1}) * P_n`` anchored at R_0 = 0.

    Use when N >= 100k AND M >= 5. PR AUC is the most GPU-friendly of the
    three metrics here: at 1M x 20 cols, GPU = 170 ms vs CPU loop = 2016 ms
    (**11.8x speedup**); at 5M x 5 cols, GPU = 213 ms vs CPU = 4141 ms
    (**19.5x speedup**). For automatic dispatch see ``compute_batch_aucs``.

    Args:
        actual: ``(N,)`` binary 0/1 labels.
        predicted: ``(N,)`` or ``(N, M)`` scores.

    Returns:
        ``(M,)`` cupy array of average-precision scores. NaN for
        single-class label columns (sklearn behavior would raise).
    """
    cp = _require_cupy()
    actual = cp.asarray(actual).astype(cp.float64)
    predicted = cp.asarray(predicted)
    if predicted.ndim == 1:
        predicted = predicted[:, cp.newaxis]

    N, M = predicted.shape
    aps = cp.empty(M, dtype=cp.float64)
    total_pos = cp.sum(actual)
    if total_pos == 0 or total_pos == N:
        # Single-class data -- AP is undefined for all columns.
        aps.fill(cp.nan)
        return aps

    idx = cp.arange(1, N + 1, dtype=cp.float64)
    for j in range(M):
        col = predicted[:, j]
        order = cp.argsort(col, kind="stable")[::-1]  # descending
        col_sorted = col[order]
        y_sorted = actual[order]

        cumtps = cp.cumsum(y_sorted)
        precision = cumtps / idx
        recall = cumtps / total_pos
        # Threshold boundaries: i is a boundary iff i == N-1 OR
        # col_sorted[i] != col_sorted[i+1].
        is_boundary = cp.concatenate([
            col_sorted[:-1] != col_sorted[1:],
            cp.array([True]),
        ])
        recall_b = recall[is_boundary]
        precision_b = precision[is_boundary]
        # delta_recall[0] anchors at recall=0 (matches sklearn AP definition).
        delta_recall = cp.concatenate([
            cp.array([recall_b[0]]),
            recall_b[1:] - recall_b[:-1],
        ])
        aps[j] = cp.sum(delta_recall * precision_b)
    return aps


# ----------------------------------------------------------------------------------------------------------------------------
# Auto-dispatching wrappers: pick GPU or CPU based on (N, M) + availability
# ----------------------------------------------------------------------------------------------------------------------------


def _normalize_scores_2d(y_score: np.ndarray) -> np.ndarray:
    """Promote ``(N,)`` -> ``(N, 1)``. Pass-through ``(N, M)``."""
    if y_score.ndim == 1:
        return y_score[:, np.newaxis]
    return y_score


def compute_batch_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    force_backend: Optional[str] = None,
) -> np.ndarray:
    """RMSE per column with auto GPU/CPU dispatch.

    Returns an ``(M,)`` numpy array (always host-side; the GPU result
    is copied back via ``cp.asnumpy``) so callers don't need to know
    which backend ran.

    Dispatch:
      - ``force_backend='gpu'``: always GPU (raises if cupy missing).
      - ``force_backend='cpu'``: always numpy reference.
      - ``None`` (default): GPU iff cupy + CUDA visible AND N >=
        ``_GPU_BATCH_THRESHOLD_N``; else CPU.
    """
    yt = np.asarray(y_true)
    yp = _normalize_scores_2d(np.asarray(y_pred))
    N = yp.shape[0]

    use_gpu = _resolve_backend(force_backend, N, yp.shape[1], "batch_rmse")
    if use_gpu:
        import cupy as cp  # lazy
        out = gpu_multiple_rmse_scores(yt, yp)
        return cp.asnumpy(out)
    # CPU reference
    if yt.ndim == 1:
        yt = yt[:, np.newaxis]
    return np.sqrt(np.mean((yt - yp) ** 2.0, axis=0))


def compute_batch_aucs(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    force_backend: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Both ROC AUC and PR AUC per column, auto GPU/CPU dispatch.

    Returns ``(roc_aucs, pr_aucs)`` as host-side numpy arrays of shape
    ``(M,)``. NaNs are returned for label columns where ROC/PR is
    undefined (single-class).

    Dispatch policy: same as ``compute_batch_rmse`` -- GPU when
    ``cupy`` is present, a CUDA device is visible, and N exceeds the
    threshold (default 100k). Override via ``force_backend``.
    """
    yt = np.asarray(y_true)
    ys = _normalize_scores_2d(np.asarray(y_score))
    N, M = ys.shape

    use_gpu = _resolve_backend(force_backend, N, M, "batch_aucs")
    if use_gpu:
        import cupy as cp  # lazy
        roc = cp.asnumpy(gpu_multiple_roc_auc_scores(yt, ys))
        pr = cp.asnumpy(gpu_multiple_pr_auc_scores(yt, ys))
        return roc, pr
    # CPU loop over per-column ``fast_aucs`` (returns roc, pr in one pass).
    # ``fast_aucs`` lives in ``core`` and is imported lazily here to
    # sidestep the core <-> _gpu_metrics circular dependency.
    from .core import fast_aucs as _fast_aucs
    roc = np.empty(M, dtype=np.float64)
    pr = np.empty(M, dtype=np.float64)
    for j in range(M):
        roc[j], pr[j] = _fast_aucs(yt, ys[:, j])
    return roc, pr


def _resolve_backend(force: Optional[str], N: int, M: int, kernel_name: str = "batch_rmse") -> bool:
    """Return True iff the GPU path should be used. ``force`` may be
    ``'gpu'`` / ``'cpu'`` / ``None``. For ``None`` (auto) the per-host
    kernel_tuning_cache decides GPU-vs-CPU for this ``kernel_name`` at
    ``(N, M)`` -- the old hardcoded ``_GPU_BATCH_THRESHOLD_*`` is the
    measurement-backed fallback."""
    if force == "gpu":
        if not is_gpu_metrics_available():
            raise RuntimeError("compute_batch_*: force_backend='gpu' but no GPU is " "available (cupy missing or no CUDA device).")
        return True
    if force == "cpu":
        return False
    if force is not None:
        raise ValueError(f"force_backend must be 'gpu', 'cpu', or None; got {force!r}")
    if not is_gpu_metrics_available():
        return False
    return _batch_metric_backend_choice(kernel_name, int(N), int(M)) == "gpu"


# ----- per-host GPU-vs-CPU tuning for the batch-metric dispatchers -----
# RMSE (cheap elementwise) and AUC (sort-heavy) have DIFFERENT crossovers, so
# they tune as separate kernels. Inputs are host-resident (eval feeds host
# arrays) -> no residency axis. The old _GPU_BATCH_THRESHOLD_* (N>=100k AND
# M>=5) is the measurement-backed fallback.
# Grid bounded so the one-time AUCS sweep stays reasonable: its CPU reference
# loops fast_aucs over M columns (O(M * N log N)), so huge N*M cells are avoided.
# The catch-all region extrapolates the largest-cell winner beyond the grid.
_BATCH_METRIC_SWEEP_N = [50_000, 200_000, 1_000_000]
_BATCH_METRIC_SWEEP_M = [1, 5, 20]
_BATCH_METRIC_SALT = 1


def _make_batch_metric_inputs(dims: dict):
    """``(y_true, y_pred)`` at ``dims['n_samples']`` rows x ``dims['n_targets']``
    columns. Binary labels (valid for ROC/PR; rmse timing is label-agnostic)."""
    rng = np.random.default_rng(0)
    N = int(dims["n_samples"])
    M = int(dims["n_targets"])
    y_true = rng.integers(0, 2, size=N).astype(np.float64)
    y_pred = rng.random((N, M))
    return (y_true, y_pred)


def _run_batch_metric_sweep(metric: str) -> list:
    """Full (n_samples x n_targets) grid sweep, cpu vs gpu, fastest equivalent per
    cell. ``metric`` in {"rmse", "aucs"}. Both backends agree to ~1e-9 (the GPU
    primitives match the CPU reference bit-for-bit on continuous data)."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    def _cpu(y_true, y_pred):
        if metric == "rmse":
            yt = y_true[:, np.newaxis] if y_true.ndim == 1 else y_true
            return np.sqrt(np.mean((yt - y_pred) ** 2.0, axis=0))
        from .core import fast_aucs as _fast_aucs
        m = y_pred.shape[1]
        roc = np.empty(m, dtype=np.float64)
        for j in range(m):
            roc[j], _pr = _fast_aucs(y_true, y_pred[:, j])
        return roc

    variants = {"cpu": _cpu}
    if is_gpu_metrics_available():
        def _gpu(y_true, y_pred):
            import cupy as cp
            if metric == "rmse":
                return cp.asnumpy(gpu_multiple_rmse_scores(y_true, y_pred))
            roc = cp.asnumpy(gpu_multiple_roc_auc_scores(y_true, y_pred))
            gpu_multiple_pr_auc_scores(y_true, y_pred)  # full roc+pr cost, return roc for equiv
            return roc
        variants["gpu"] = _gpu

    return sweep_backend_grid(
        variants,
        {"n_samples": _BATCH_METRIC_SWEEP_N, "n_targets": _BATCH_METRIC_SWEEP_M},
        _make_batch_metric_inputs,
        reference="cpu",
        repeats=3, equiv_rtol=1e-6, equiv_atol=1e-6,
    )


def _batch_metric_fallback_choice(n_samples: int, n_targets: int) -> str:
    """Pre-sweep heuristic (the specs' dynamic fallback callable): the old N>=100k
    AND M>=5 threshold."""
    return "gpu" if (n_samples >= _GPU_BATCH_THRESHOLD_N and n_targets >= _GPU_BATCH_THRESHOLD_M) else "cpu"


def _batch_metric_backend_choice(kernel_name: str, N: int, M: int) -> str:
    """Per-host cpu/gpu choice for a batch-metric kernel at (N, M) via the matching
    spec's choose() (memoized per dims)."""
    spec = _BATCH_RMSE_SPEC if kernel_name == "batch_rmse" else _BATCH_AUCS_SPEC
    return spec.choose(n_samples=int(N), n_targets=int(M))


# Register the two batch-metric dispatchers with the kernel-tuner registry so
# retune_all / mlframe-tune-kernels tune their per-host GPU-vs-CPU crossover.
# GPU-capable; CPU reference covered by salt. Inputs host-resident -> no residency.
from pyutilz.performance.kernel_tuning.registry import kernel_tuner

_BATCH_RMSE_SPEC = kernel_tuner(
    kernel_name="batch_rmse",
    variant_fns=(gpu_multiple_rmse_scores,),  # GPU primitive; CPU ref + edits covered by salt
    tuner=(lambda: _run_batch_metric_sweep("rmse")),
    axes={"n_samples": list(_BATCH_METRIC_SWEEP_N), "n_targets": list(_BATCH_METRIC_SWEEP_M)},
    fallback=_batch_metric_fallback_choice,  # callable (n_samples, n_targets) -> str
    gpu_capable=True,
    salt=_BATCH_METRIC_SALT,
    cli_label="batch_rmse",
)

_BATCH_AUCS_SPEC = kernel_tuner(
    kernel_name="batch_aucs",
    variant_fns=(gpu_multiple_roc_auc_scores, gpu_multiple_pr_auc_scores),
    tuner=(lambda: _run_batch_metric_sweep("aucs")),
    axes={"n_samples": list(_BATCH_METRIC_SWEEP_N), "n_targets": list(_BATCH_METRIC_SWEEP_M)},
    fallback=_batch_metric_fallback_choice,  # callable (n_samples, n_targets) -> str
    gpu_capable=True,
    salt=_BATCH_METRIC_SALT,
    cli_label="batch_aucs",
)
