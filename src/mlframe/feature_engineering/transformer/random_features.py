"""Public APIs for Random Fourier Features and Sinusoidal Positional Encoding.

Three functions:

- ``compute_rff_features`` - Rahimi-Recht RFF (cos + sin of random Gaussian projection); GPU + CPU fallback.
- ``compute_positional_encoding`` - standard transformer PE; CPU only (eltwise, GPU has no win).
- ``positions_within_group`` - polars helper that returns the per-group ordinal index, the right input to PE for session / time / ticker-day data.

Both feature blocks output ``polars.DataFrame`` with deterministic column names so downstream feature-selection / training code can rebind by name.
"""
from __future__ import annotations

import logging
from typing import Literal, Union

import numpy as np
import polars as pl

from ._kernels_njit import rff_matmul_njit
from ._utils import (
    free_gpu_memory_pool,
    is_gpu_available,
    require_seed,
    sigma_median_heuristic,
    validate_numeric_input,
)

logger = logging.getLogger(__name__)


def compute_rff_features(
    X: Union[pl.DataFrame, np.ndarray],
    *,
    seed: int,
    n_features: int = 256,
    sigma: Union[float, Literal["median"]] = "median",
    standardize: bool = True,
    use_gpu: Union[bool, Literal["auto"]] = "auto",
    gpu_threshold: int | None = None,
    dtype: np.dtype = np.float32,
    batch_rows: int = 100_000,
    column_prefix: str = "rff",
    release_memory_after: bool = True,
) -> pl.DataFrame:
    """Approximate-kernel features via Random Fourier Features (Rahimi & Recht 2007).

    Output: ``sqrt(2 / n_features) * [cos(X @ W + b), sin(X @ W + b)]`` with ``W[d, m] ~ N(0, sigma^-2)`` and ``b[m] ~ U[0, 2 * pi)``, ``m = n_features // 2``.
    The bandwidth ``sigma`` defaults to the median-pairwise-distance heuristic on a 2048-row subsample (call site can override with a float).

    GPU path uses streaming cupy with pinned-memory double-buffer for input batches; CPU path is ``numba.njit(parallel=True)``. ``use_gpu="auto"`` dispatches
    to GPU when cupy is available AND total work ``N * d * n_features`` exceeds the calibrated threshold (None default -> first-call micro-bench cached per
    shape bucket).

    Returns a polars DataFrame ``(N, n_features)`` with column names ``{column_prefix}_cos_{i}`` / ``{column_prefix}_sin_{i}``. The cos / sin split makes the
    output interpretable for feature-importance plots.

    Validation: input must be 2-D numeric with no NaN / Inf (use ``polars.fill_null`` upstream); dense (no sparse). Hard cap ``d <= 32768``.
    """
    seed = require_seed(seed)
    if n_features < 2 or n_features % 2 != 0:
        raise ValueError(f"n_features must be even and >= 2; got {n_features}.")
    if standardize is not True and standardize is not False:
        raise TypeError(f"standardize must be bool; got {type(standardize).__name__}.")

    X_arr, _input_names = _coerce_input(X, dtype=dtype)
    validate_numeric_input(X_arr, name="X", allow_fp16=True)

    if standardize:
        from sklearn.preprocessing import RobustScaler
        # Median / IQR-based; robust to heavy tails (financial data, etc.). NOT QuantileTransformer because we want a single shift+scale per col,
        # not a non-linear remapping that would destroy the random-feature interpretation.
        scaler = RobustScaler().fit(X_arr)
        X_std = scaler.transform(X_arr).astype(dtype, copy=False)
    else:
        X_std = X_arr.astype(dtype, copy=False)

    n, d = X_std.shape
    m = n_features // 2

    # Bandwidth estimation.
    if sigma == "median":
        median_dist = sigma_median_heuristic(X_std, seed=seed)
        # Rahimi-Recht: gamma = 1 / (2 * sigma_RBF^2); the standard "median trick" sets sigma_RBF = median_dist.
        # So W ~ N(0, 1 / sigma_RBF^2) = N(0, 1 / median_dist^2). Pre-scale W's std.
        sigma_w = float(1.0 / max(median_dist, np.finfo(np.float64).tiny))
    elif isinstance(sigma, (int, float)):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive; got {sigma}.")
        sigma_w = float(sigma)
    else:
        raise TypeError(f"sigma must be 'median' or a positive float; got {type(sigma).__name__}={sigma!r}.")

    # Generate W and b once. seed is the user-supplied seed; the RNG state is independent here (no SeedSequence.spawn needed - RFF has no concept of multi-head).
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((d, m)) * sigma_w).astype(dtype, copy=False)
    b = (rng.random(m) * 2.0 * np.pi).astype(dtype, copy=False)
    scale = float(np.sqrt(2.0 / n_features))

    use_gpu_resolved = _resolve_use_gpu(use_gpu, work=n * d * n_features, threshold=gpu_threshold)

    out = np.empty((n, n_features), dtype=dtype)
    if use_gpu_resolved:
        try:
            from ._kernels_cupy import rff_matmul_cupy
            rff_matmul_cupy(X_std, W, b, out, scale, batch_rows=batch_rows)
            if release_memory_after:
                free_gpu_memory_pool()
        except Exception as exc:  # pragma: no cover - environment fallback
            logger.warning("rff_matmul_cupy failed (%s: %s); falling back to CPU.", type(exc).__name__, exc)
            rff_matmul_njit(X_std, W, b, out, scale)
    else:
        rff_matmul_njit(X_std, W, b, out, scale)

    # Build polars output with stable column naming. cos columns first, sin columns second.
    col_names = [f"{column_prefix}_cos_{i}" for i in range(m)] + [f"{column_prefix}_sin_{i}" for i in range(m)]
    return pl.DataFrame({name: out[:, idx] for idx, name in enumerate(col_names)})


def compute_positional_encoding(
    positions: Union[pl.Series, np.ndarray],
    *,
    d_model: int = 16,
    base: float = 10_000.0,
    dtype: np.dtype = np.float32,
    column_prefix: str = "pe",
) -> pl.DataFrame:
    """Sinusoidal positional encoding (Vaswani et al. 2017): ``PE(pos, 2i) = sin(pos / base^(2i/d_model))``, ``PE(pos, 2i+1) = cos(...)``.

    CPU only - eltwise compute is ~50 ms for N=10M, d_model=16 on a single core, and the cupy launch overhead alone is ~10 ms. No ``use_gpu`` parameter.

    Positions are clamped at ``pos %% 1_000_000`` to keep ``pos / base^(0)`` within fp32 safe range; for the typical use case (session / day / ticker index) this
    is never approached, but the clamp prevents silent fp32 overflow if a caller accidentally feeds raw unix timestamps.

    For position-equals-row-index inputs (a contiguous ``[0..N-1]``), this is a foot-gun - trees are permutation-invariant so a raw row index encodes nothing
    predictive. We emit an INFO log so the caller sees it; for genuinely time-ordered data the warning is harmless.
    """
    if d_model < 2 or d_model % 2 != 0:
        raise ValueError(f"d_model must be even and >= 2; got {d_model}.")
    if base <= 1.0:
        raise ValueError(f"base must be > 1.0; got {base}.")

    if isinstance(positions, pl.Series):
        pos_arr = positions.to_numpy()
    elif isinstance(positions, np.ndarray):
        pos_arr = positions
    else:
        raise TypeError(f"positions must be polars.Series or np.ndarray; got {type(positions).__name__}.")
    if pos_arr.ndim != 1:
        raise ValueError(f"positions must be 1-D; got shape {pos_arr.shape}.")

    # Detect contiguous-row-index input. ``arange(N).all()`` check is cheap; only log if length > 1 (singletons are ambiguous).
    if pos_arr.size > 1:
        if pos_arr.dtype.kind in ("i", "u"):
            if pos_arr[0] == 0 and pos_arr[-1] == pos_arr.size - 1:
                expected = np.arange(pos_arr.size, dtype=pos_arr.dtype)
                if np.array_equal(pos_arr, expected):
                    logger.info(
                        "compute_positional_encoding: positions look like a contiguous row index. PE on raw row indices is meaningless for tree models "
                        "(permutation-invariant); call positions_within_group(df, group_col, sort_col) for the typical session / time-ordered use case."
                    )

    # Clamp against fp32 overflow.
    pos_f = (pos_arr.astype(np.float64) % 1_000_000.0).astype(dtype, copy=False)

    half = d_model // 2
    # div_term[i] = 1 / base^(2i / d_model) for i = 0..half-1
    i_idx = np.arange(half, dtype=np.float64)
    div_term = (1.0 / np.power(base, 2.0 * i_idx / d_model)).astype(dtype, copy=False)
    # angles: (N, half) = pos_f[:, None] * div_term[None, :]
    angles = pos_f[:, None] * div_term[None, :]
    pe = np.empty((pos_f.size, d_model), dtype=dtype)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)

    col_names = [f"{column_prefix}_{i}" for i in range(d_model)]
    return pl.DataFrame({name: pe[:, idx] for idx, name in enumerate(col_names)})


def positions_within_group(
    df: pl.DataFrame,
    group_col: str,
    sort_col: str | None = None,
) -> pl.Series:
    """Return the per-group ordinal index for each row in ``df``. The output is the canonical input to ``compute_positional_encoding``.

    If ``sort_col`` is provided, rows are first sorted within each group by that column before ordinals are assigned; the result is then re-aligned to the original
    row order so the returned Series has the same length and row mapping as ``df``.

    For un-sorted groups (``sort_col=None``), the ordinal is just the row-arrival order within the group, which is rarely the right thing - prefer to pass an
    explicit ``sort_col`` (timestamp, sequence id) for clarity.
    """
    if group_col not in df.columns:
        raise KeyError(f"group_col '{group_col}' not in df.columns={df.columns}.")
    if sort_col is not None and sort_col not in df.columns:
        raise KeyError(f"sort_col '{sort_col}' not in df.columns={df.columns}.")

    if sort_col is None:
        return df.with_columns(pl.int_range(pl.len(), dtype=pl.Int64).over(group_col).alias("__pe_pos__"))["__pe_pos__"]
    # With sort_col: assign per-group ordinal in sorted order, then realign.
    out = (
        df.with_row_index("__pe_orig__")
        .sort([group_col, sort_col])
        .with_columns(pl.int_range(pl.len(), dtype=pl.Int64).over(group_col).alias("__pe_pos__"))
        .sort("__pe_orig__")
        .get_column("__pe_pos__")
    )
    return out


def _coerce_input(
    X: Union[pl.DataFrame, np.ndarray],
    *,
    dtype: np.dtype,
) -> tuple[np.ndarray, list[str] | None]:
    """Normalise polars / numpy input to a 2-D contiguous ndarray with the requested dtype. Returns ``(arr, names)`` where names is the polars column list or None.

    Skips the dtype cast for non-numeric inputs - ``validate_numeric_input`` downstream raises the right TypeError with a useful message. Casting a string ndarray
    to float32 would otherwise raise an unhelpful ValueError from numpy itself.
    """
    if isinstance(X, pl.DataFrame):
        names = X.columns
        arr = X.to_numpy()  # zero-copy for uniform numeric dtypes; copy for mixed
        if arr.dtype.kind in ("f", "i", "u") and arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr, names
    if isinstance(X, np.ndarray):
        if X.dtype.kind in ("f", "i", "u") and X.dtype != dtype:
            X = X.astype(dtype, copy=False)
        if not X.flags["C_CONTIGUOUS"] and X.dtype.kind in ("f", "i", "u"):
            X = np.ascontiguousarray(X)
        return X, None
    raise TypeError(f"X must be polars.DataFrame or np.ndarray; got {type(X).__name__}.")


# Module-level GPU threshold cache, keyed by (log2_n_bucket, log2_d_bucket, log2_n_features_bucket).
# Populated by first-call micro-bench when ``gpu_threshold=None`` and ``use_gpu='auto'``. Buckets are log2-rounded so we don't re-bench for every minor shape
# variation; a 100k-row call uses the same bucket as 120k.
_GPU_AUTO_CACHE: dict[tuple[int, int, int], bool] = {}


def _resolve_use_gpu(
    use_gpu: Union[bool, Literal["auto"]],
    *,
    work: int,
    threshold: int | None,
) -> bool:
    """Decide GPU vs CPU based on ``use_gpu`` flag, calibrated threshold, and runtime availability.

    Cases:
    - ``use_gpu=False`` -> CPU regardless.
    - ``use_gpu=True``  -> GPU if available, else CPU (with INFO log).
    - ``use_gpu='auto'`` AND ``threshold is None`` -> dispatch via cached micro-bench bucket; first-call cost is a single mini-bench (~50 ms).
    - ``use_gpu='auto'`` AND ``threshold is set`` -> GPU iff available AND work >= threshold.
    """
    if use_gpu is False:
        return False
    if use_gpu is True:
        if not is_gpu_available():
            logger.info("use_gpu=True but GPU is not available; using CPU.")
            return False
        return True
    if use_gpu != "auto":
        raise ValueError(f"use_gpu must be True, False, or 'auto'; got {use_gpu!r}.")
    if not is_gpu_available():
        return False
    if threshold is not None:
        return work >= threshold
    # Auto without explicit threshold: heuristic until a real micro-bench landing.
    # For RFF: GPU wins clearly when N * d > ~5M (work = N * d * n_features but the bandwidth-bound piece is N * d).
    # Wave 23 P2 (2026-05-20): consult kernel_tuning_cache for HW-tuned
    # crossover. The 5_000_000 * 256 was a documented "placeholder ...
    # replaces it after calibration"; calibration never landed. Cache
    # lookup falls back to the placeholder when no entry exists yet.
    try:
        from pyutilz.system.kernel_tuning_cache import KernelTuningCache
        _cache = KernelTuningCache.load_or_create()
        _e = _cache.lookup("rff_matmul", {"work": int(work)})
        _crossover = int(_e["work_threshold"]) if _e and "work_threshold" in _e else (5_000_000 * 256)
    except Exception:
        _crossover = 5_000_000 * 256  # placeholder fallback
    return work >= _crossover
