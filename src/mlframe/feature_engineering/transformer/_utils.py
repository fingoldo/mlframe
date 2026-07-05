"""Shared utilities for the transformer FE subpackage.

Includes:
- GPU detection with smoke-call (catches "wheel installs but CUDA driver mismatch" — a cupy-cuda12x install on a CUDA 11.x driver imports OK but crashes inside the first kernel call).
- VRAM budget helper that accounts for cupy memory-pool retained bytes (``memGetInfo`` returns physical-free only, not pool-free).
- Input validation that fails fast on NaN / sparse / non-numeric dtype rather than silently coercing — the attention math has no defensible policy for missing values, so callers must impute upstream.
- Median-heuristic bandwidth estimator for RBF / cosine kernels, capped to a subsample so the O(n^2) pairwise step doesn't dominate for N >> 2048.

All ``logger.*`` and ``print`` output here is ASCII-only by project rule; cp1251 Windows consoles crash on non-ASCII writes from the streams.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - numba is a hard dep in practice
    prange = range

    def njit(*args, **kwargs):  # no-op fallback so the module imports
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def deco(fn):
            return fn

        return deco

logger = logging.getLogger(__name__)


# Above this element count the prange thread-launch floor is amortised and the parallel counter
# scales across cores; below it the serial kernel avoids the spawn overhead. Env-overridable per host.
_NONFINITE_PAR_THRESHOLD = int(os.environ.get("MLFRAME_NONFINITE_PAR_THRESHOLD", "1000000"))


@njit(cache=True)
def _count_nonfinite_serial(Xf: np.ndarray) -> int:
    flat = Xf.ravel()
    total = 0
    for i in range(flat.size):
        v = flat[i]
        # v - v == 0.0 holds for every finite value; NaN-NaN and (+/-inf)-(+/-inf) both yield NaN
        # (NaN != 0.0), so this single test flags NaN AND +/-Inf without a separate isinf/isnan pass.
        if not (v - v == 0.0):
            total += 1
    return total


@njit(cache=True, parallel=True)
def _count_nonfinite_parallel(Xf: np.ndarray) -> int:
    flat = Xf.ravel()
    n = flat.size
    total = 0
    for i in prange(n):
        v = flat[i]
        if not (v - v == 0.0):
            total += 1
    return total


def _count_nonfinite_cells(Xf: np.ndarray) -> int:
    """Count NaN/+/-Inf cells in a float array via a fused single-pass numba kernel.

    Replaces ``int(np.count_nonzero(~np.isfinite(X)))`` which allocated TWO full N*d
    temporaries (the ``isfinite`` bool array AND its bitwise inverse) and walked the data
    twice. The njit counter holds no temporary at all -- a meaningful peak-RAM saving on the
    100+ GB frames this validator runs on -- and is ~8-14x faster (bench:
    ``_benchmarks/bench_validate_nonfinite_count.py``). Bit-identical count to the numpy form
    (verified NaN / +inf / -inf / mixed / all-bad on f32 and f64). Size-dispatched: the
    parallel twin is selected above ``_NONFINITE_PAR_THRESHOLD`` elements.
    """
    if Xf.size >= _NONFINITE_PAR_THRESHOLD:
        return int(_count_nonfinite_parallel(Xf))
    return int(_count_nonfinite_serial(Xf))


# Per-process cache. ``None`` = not probed yet; True/False = probed result.
# Module-level (not function attribute) so multiprocessing children inherit the un-probed state and re-probe in their own CUDA context.
_GPU_AVAILABLE: Optional[bool] = None


def is_gpu_available() -> bool:
    """Return True iff cupy imports AND can actually compile + run a reduction kernel.

    Two-step probe:
      1) ``cp.zeros(1).get()`` - device allocation + D2H copy. Catches the
         ``cupy-cuda12x`` vs CUDA-11-driver mismatch (imports cleanly,
         raises CUDARuntimeError on first kernel call).
      2) ``cp.asarray([1.0]).sum().item()`` - forces an NVRTC kernel
         compilation. Catches broken nvrtc/cublas DLL installs where the
         allocation succeeds but kernel compilation enters cupy's softlink
         retry loop and raises RecursionError (observed 2026-05-20 on a
         host with renamed cublas64_11 DLLs - cupy's _get_nvrtc_version
         exception handler re-enters _get_softlink and recurses).

    ``except BaseException`` covers RecursionError too (it's a RuntimeError
    subclass so ``except Exception`` would also catch it, but be explicit
    in case future cupy versions raise SystemError or similar).
    Result is memoised per-process.
    """
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE
    try:
        import cupy as cp
        _ = cp.zeros(1, dtype=cp.float32).get()
        _ = cp.asarray([1.0], dtype=cp.float32).sum().item()
        _GPU_AVAILABLE = True
        logger.info("GPU (cupy) detected and usable.")
    except Exception as exc:  # pragma: no cover - environment-dependent
        # Broad catch: ImportError, CUDARuntimeError, RecursionError from
        # broken nvrtc DLL retry loop, any driver/runtime mismatch.
        _GPU_AVAILABLE = False
        logger.info("GPU (cupy) unavailable, falling back to CPU. Reason: %s: %s", type(exc).__name__, exc)
    return _GPU_AVAILABLE


def reset_gpu_probe() -> None:
    """Clear the cached GPU-availability flag. Used by tests that want to re-probe after monkeypatching cupy.

    Production callers should not need this; the per-process cache is intentional.
    """
    global _GPU_AVAILABLE
    _GPU_AVAILABLE = None


def gpu_available_bytes(safety_fraction: float = 0.6) -> int:
    """Return usable GPU memory in bytes = ``(memGetInfo_free + pool_free_bytes) * safety_fraction``.

    ``cp.cuda.runtime.memGetInfo`` reports only the physical free VRAM and treats the cupy memory pool's already-allocated chunks as "used" even though the pool
    can hand them back to the next allocation. Adding ``pool.free_bytes()`` recovers that headroom. The 60% safety multiplier leaves room for cuBLAS workspace
    (typically 128-256 MB on Ampere+) and pool fragmentation. Tune via ``safety_fraction`` for tight tile-budget sizing.

    Returns 0 if cupy is unavailable; callers must check before treating the result as a real budget.
    """
    if not is_gpu_available():
        return 0
    import cupy as cp
    free_phys = int(cp.cuda.runtime.memGetInfo()[0])
    pool_free = int(cp.get_default_memory_pool().free_bytes())
    return int((free_phys + pool_free) * safety_fraction)


def free_gpu_memory_pool() -> None:
    """Release all retained blocks from the default cupy memory pool.

    Called at the end of ``compute_row_attention`` / ``compute_rff_features`` when ``release_memory_after=True`` so the next downstream GPU operator (CatBoost-GPU,
    another mlframe filter) doesn't OOM on residual ~5-10 GB of pool-retained but caller-finished buffers.
    """
    if not is_gpu_available():
        return
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()


def validate_numeric_input(
    X: np.ndarray,
    *,
    name: str = "X",
    allow_fp16: bool = True,
) -> None:
    """Reject inputs that the attention / RFF math has no clean policy for.

    Rules (per ML critic #13, #14, #16, #27):
    - Must be ndarray, not sparse: scipy.sparse interop with cupy/numba is fragile and one-hot inputs should be re-engineered upstream (use the categorical path).
    - Must be a numeric dtype: silent coercion of ``pl.Categorical`` / object would produce meaningless distances.
    - Must contain no NaN / +/-Inf: silent imputation has three defensible policies (mask, fill, ignore-in-softmax) all of which deceive callers; force them to pick.
    - Hard cap on ``d``: ``> 32_768`` is rejected as a foot-gun (the projection matrix alone is too big and indicates an upstream pre-densification mistake).

    Errors include the dtype, shape, and the count of bad cells so the caller can fix without re-running.
    """
    if hasattr(X, "ndim") and not isinstance(X, np.ndarray):
        # Cheap duck-type check; sparse matrices have ``ndim`` but are not ``np.ndarray``.
        raise TypeError(f"{name} must be np.ndarray, got {type(X).__name__} (sparse / pandas-extension / polars inputs are not supported; convert upstream).")
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray, got {type(X).__name__}.")
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (N, d); got ndim={X.ndim} shape={X.shape}.")
    if X.dtype.kind not in ("f", "i", "u"):
        raise TypeError(f"{name} must be a numeric dtype (float / int / uint); got dtype={X.dtype}.")
    if X.dtype == np.float16 and not allow_fp16:
        raise TypeError(f"{name} dtype float16 not allowed here; convert to float32 upstream.")
    n_bad = _count_nonfinite_cells(X) if X.dtype.kind == "f" else 0
    if n_bad > 0:
        raise ValueError(
            f"{name} contains {n_bad} non-finite cells (NaN or +/-Inf). Impute upstream via polars.fill_null('mean') or sklearn KNNImputer; "
            "this module does not silently mask missing values because the three plausible policies (mask, fill, ignore-in-softmax) give different results."
        )
    if X.shape[1] > 32_768:
        raise ValueError(
            f"{name} has d={X.shape[1]} > 32768 (hard cap). At this width the projection matrix alone exceeds 1 GB and indicates an upstream one-hot / "
            "pre-densification mistake. Reduce dimensionality first (PCA, feature_selection, polars one-hot to Enum, etc.)."
        )
    # Float32 precision warning: every transformer in this package follows the validator
    # with ``np.asarray(X, dtype=np.float32)``. float32 mantissa is 24 bits (~16.7M);
    # int inputs with values larger than that lose the low bits SILENTLY when cast,
    # corrupting kNN distances / SMOTE neighbour lookups / RFF projections without any
    # user-visible error. Common offenders: epoch-seconds timestamps (~1.7e9),
    # high-cardinality hash IDs, monotone counters. Surface this at validation time
    # so the caller can promote to float64 or rescale upstream.
    if X.dtype.kind in ("i", "u"):
        _abs_max = int(np.abs(X).max()) if X.size else 0
        _F32_SAFE_MAX = 1 << 24  # 16_777_216 -- float32 mantissa
        if _abs_max >= _F32_SAFE_MAX:
            import warnings as _w
            _w.warn(
                f"{name}: integer dtype {X.dtype} with abs-max {_abs_max:,} >= 2^24 "
                f"({_F32_SAFE_MAX:,}). Downstream transformers in this package cast "
                f"to float32, which has a 24-bit mantissa; values above that limit "
                f"lose low bits SILENTLY (kNN / SMOTE / RFF compute on rounded values, "
                f"degrading model quality with no error). Either promote to float64 "
                f"upstream, rescale to a smaller range, or accept the precision loss.",
                stacklevel=2,
            )


def sigma_median_heuristic(
    X: np.ndarray,
    *,
    n_sub: int = 2_048,
    seed: int,
) -> float:
    """Median pairwise Euclidean distance on a random subsample - classical RBF-bandwidth heuristic (Garreau et al. 2017 surveys it).

    Subsample cap (default 2048) keeps the pairwise cost at 2048 * 2048 = 4M distance evaluations regardless of input N. The cap is configurable but values
    above 4096 don't measurably improve the estimate while quadrupling runtime each doubling.

    ``seed`` is required (no default) to keep reproducibility explicit — feeding a derived seed silently changes the bandwidth estimate across runs.

    Reference: Rahimi & Recht 2007 use ``sigma = 1 / median_distance`` as the kernel bandwidth in their RBF feature map; we expose the median distance itself and
    let the caller convert.
    """
    if X.shape[0] == 0:
        raise ValueError("sigma_median_heuristic: cannot estimate bandwidth on empty X.")
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n > n_sub:
        idx = rng.choice(n, size=n_sub, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    # Always use the block-wise pairwise reduction. The naive broadcast (X_sub[:, None, :] - X_sub[None, :, :]) materialises an n_sub^2 * d float32 cube which
    # OOMs at d > 16 for n_sub=2048 (e.g. d=50 -> 800 MB). The block-wise path stores only the n_sub^2 distance matrix and one (chunk, d) gemm input at a time -
    # peak memory ~n_sub^2 * 8 bytes = 32 MB for n_sub=2048, independent of d.
    return _median_pairwise_chunked(X_sub, chunk=256)


def _median_pairwise_chunked(X_sub: np.ndarray, chunk: int) -> float:
    """Pairwise-distance median via ``scipy.spatial.distance.pdist`` (returns only the upper triangle).

    Memory: ``n_sub * (n_sub - 1) / 2 * 8 bytes`` ~~ 16 MB for n_sub=2048. Half the cost of materialising the full n_sub^2 distance matrix and avoids the
    Windows page-file pressure that the full-matrix path hit under repeated calls in the same process.

    ``chunk`` is accepted for back-compat but ignored (scipy's C implementation handles its own blocking internally).
    """
    from scipy.spatial.distance import pdist
    # pdist returns the condensed upper-triangle distance vector; all entries are pairwise distances between distinct points (no self-pairs to filter).
    dists = pdist(X_sub.astype(np.float32, copy=False), metric="euclidean")
    if dists.size == 0:
        return 1.0
    # Filter exact zeros (duplicate rows) so they don't bias the median; pdist returns 0 distance only for exact-equal rows.
    nonzero = dists[dists > 0]
    if nonzero.size == 0:
        return 1.0
    return float(np.median(nonzero))


def require_seed(seed: object) -> int:
    """Validate that ``seed`` is a literal Python int and not ``None`` or derived-from-data.

    Per ML critic #7: a seed derived from input data (e.g. ``hash(X.shape)``) silently makes the projection matrices a function of the train fold, which is a
    leakage channel even when the OOF discipline correctly excludes y. Forcing ``int`` at the API boundary doesn't catch all foot-guns but makes the cheap ones
    impossible (``None`` -> implicit randomness, ``np.int64`` from a derived hash -> caught by a runtime type-check, etc.).
    """
    if seed is None:
        raise TypeError("seed is required (no default). Pass a literal int, e.g. seed=42.")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}. Do not derive seeds from input data (e.g. hash(X.shape)) — that leaks data into the projection matrices.")
    seed_int = int(seed)
    if seed_int < 0 or seed_int >= 2**32:
        raise ValueError(f"seed must be in [0, 2**32); got {seed_int}.")
    return seed_int
