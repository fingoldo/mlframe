"""Public APIs for Random Fourier Features and Sinusoidal Positional Encoding.

Three functions:

- ``compute_rff_features`` - Rahimi-Recht RFF (cos + sin of random Gaussian projection); GPU + CPU fallback.
- ``compute_positional_encoding`` - standard transformer PE; CPU only (eltwise, GPU has no win).
- ``positions_within_group`` - polars helper that returns the per-group ordinal index, the right input to PE for session / time / ticker-day data.

Both feature blocks output ``polars.DataFrame`` with deterministic column names so downstream feature-selection / training code can rebind by name.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Union

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


class RFFState:
    """Fitted RFF projection state (train-only): RobustScaler + random matrix W + phase b + scale.

    Returned by ``compute_rff_features(..., return_state=True)`` so a caller can apply the SAME train-fitted projection to held-out / predict frames via ``rff_apply_state`` -- the leakage-safe contract (fit on train, apply to query). The state is small (``W`` is d x m float32) and picklable for predict-time replay.
    """

    __slots__ = ("scaler", "W", "b", "scale", "n_features", "column_prefix", "dtype")

    def __init__(self, scaler, W, b, scale, n_features, column_prefix, dtype):
        self.scaler = scaler
        self.W = W
        self.b = b
        self.scale = scale
        self.n_features = n_features
        self.column_prefix = column_prefix
        self.dtype = dtype


def _rff_fit_state(X_arr, *, seed, n_features, sigma, standardize, column_prefix, dtype) -> RFFState:
    """Fit the RFF projection (scaler + W + b) on ``X_arr`` ONLY. No query rows touch this fit."""
    if standardize:
        from sklearn.preprocessing import RobustScaler
        # Median / IQR-based; robust to heavy tails (financial data, etc.). NOT QuantileTransformer because we want a single shift+scale per col, not a non-linear remapping that would destroy the random-feature interpretation.
        scaler = RobustScaler().fit(X_arr)
        X_std = scaler.transform(X_arr).astype(dtype, copy=False)
    else:
        scaler = None
        X_std = X_arr.astype(dtype, copy=False)

    _n, d = X_std.shape
    m = n_features // 2
    if sigma == "median":
        median_dist = sigma_median_heuristic(X_std, seed=seed)
        # Rahimi-Recht: gamma = 1 / (2 * sigma_RBF^2); the standard "median trick" sets sigma_RBF = median_dist. So W ~ N(0, 1 / median_dist^2). Pre-scale W's std.
        sigma_w = float(1.0 / max(median_dist, np.finfo(np.float64).tiny))
    elif isinstance(sigma, (int, float)):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive; got {sigma}.")
        sigma_w = float(sigma)
    else:
        raise TypeError(f"sigma must be 'median' or a positive float; got {type(sigma).__name__}={sigma!r}.")

    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((d, m)) * sigma_w).astype(dtype, copy=False)
    b = (rng.random(m) * 2.0 * np.pi).astype(dtype, copy=False)
    scale = float(np.sqrt(2.0 / n_features))
    return RFFState(scaler, W, b, scale, n_features, column_prefix, dtype)


def _rff_project(state: RFFState, X_arr, *, use_gpu, gpu_threshold, batch_rows, release_memory_after) -> np.ndarray:
    """Apply a fitted ``RFFState`` to ``X_arr``; returns the ``(n, n_features)`` cos/sin feature matrix."""
    if state.scaler is not None:
        X_std = state.scaler.transform(X_arr).astype(state.dtype, copy=False)
    else:
        X_std = X_arr.astype(state.dtype, copy=False)
    n, d = X_std.shape
    out = np.empty((n, state.n_features), dtype=state.dtype)
    use_gpu_resolved = _resolve_use_gpu(use_gpu, work=n * d * state.n_features, threshold=gpu_threshold)
    if use_gpu_resolved:
        try:
            from ._kernels_cupy import rff_matmul_cupy
            rff_matmul_cupy(X_std, state.W, state.b, out, state.scale, batch_rows=batch_rows)
            if release_memory_after:
                free_gpu_memory_pool()
        except Exception as exc:  # pragma: no cover - environment fallback
            logger.warning("rff_matmul_cupy failed (%s: %s); falling back to CPU.", type(exc).__name__, exc)
            rff_matmul_njit(X_std, state.W, state.b, out, state.scale)
    else:
        rff_matmul_njit(X_std, state.W, state.b, out, state.scale)
    return out


def rff_apply_state(
    state: RFFState,
    X: Union[pl.DataFrame, np.ndarray],
    *,
    use_gpu: Union[bool, Literal["auto"]] = "auto",
    gpu_threshold: int | None = None,
    batch_rows: int = 100_000,
    release_memory_after: bool = True,
) -> pl.DataFrame:
    """Apply a train-fitted ``RFFState`` to a held-out / predict frame (leakage-safe replay)."""
    X_arr, _ = _coerce_input(X, dtype=state.dtype)
    validate_numeric_input(X_arr, name="X", allow_fp16=True)
    out = _rff_project(state, X_arr, use_gpu=use_gpu, gpu_threshold=gpu_threshold, batch_rows=batch_rows, release_memory_after=release_memory_after)
    m = state.n_features // 2
    col_names = [f"{state.column_prefix}_cos_{i}" for i in range(m)] + [f"{state.column_prefix}_sin_{i}" for i in range(m)]
    return pl.DataFrame({name: out[:, idx] for idx, name in enumerate(col_names)})


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
    X_query: Optional[Union[pl.DataFrame, np.ndarray]] = None,
    splitter: Optional[Any] = None,
    return_state: bool = False,
) -> Union[pl.DataFrame, tuple[pl.DataFrame, RFFState]]:
    """Approximate-kernel features via Random Fourier Features (Rahimi & Recht 2007).

    Output: ``sqrt(2 / n_features) * [cos(X @ W + b), sin(X @ W + b)]`` with ``W[d, m] ~ N(0, sigma^-2)`` and ``b[m] ~ U[0, 2 * pi)``, ``m = n_features // 2``.
    The bandwidth ``sigma`` defaults to the median-pairwise-distance heuristic on a 2048-row subsample (call site can override with a float).

    Train-only-fit contract (mirrors ``compute_local_lift_features`` / ``compute_class_distance_features``):
      * Mode B (``X_query`` given): fit scaler + bandwidth + W/b on ``X`` (train) ONLY, then project ``X_query`` -- the leakage-safe path for held-out / predict frames. ``X`` itself is NOT projected.
      * Mode A (``splitter`` given, ``X_query=None``): OOF -- for each fold, fit on the train indices and project the held-out indices, assembling a full ``(len(X), n_features)`` matrix with no row's own data informing its own scaler / bandwidth.
      * Default (both None): fit AND project on the full ``X`` (the historical behaviour). This is IN-SAMPLE -- the scaler and median bandwidth see every row they then encode, a mild leak when the output feeds a supervised model. Prefer Mode A / Mode B when the RFF block feeds a downstream learner.

    ``return_state=True`` additionally returns the fitted ``RFFState`` (Mode B / default only) so the caller can replay the projection on later frames via ``rff_apply_state``.

    GPU path uses streaming cupy with pinned-memory double-buffer for input batches; CPU path is ``numba.njit(parallel=True)``. ``use_gpu="auto"`` dispatches to GPU when cupy is available AND total work ``N * d * n_features`` exceeds the calibrated threshold.

    Returns a polars DataFrame ``(N, n_features)`` with column names ``{column_prefix}_cos_{i}`` / ``{column_prefix}_sin_{i}`` (or ``(df, state)`` when ``return_state``).

    Validation: input must be 2-D numeric with no NaN / Inf (use ``polars.fill_null`` upstream); dense (no sparse). Hard cap ``d <= 32768``.
    """
    seed = require_seed(seed)
    if n_features < 2 or n_features % 2 != 0:
        raise ValueError(f"n_features must be even and >= 2; got {n_features}.")
    # Wave 28 P0 fix (2026-05-20): the strict ``is not True and is not False`` type-guard rejected ``np.bool_(True)`` / ``numpy.True_`` / ``int(1)`` from upstream config or sklearn output, all of which are semantically valid bools. isinstance covers Python bool + numpy bool uniformly.
    import numpy as _np_for_bool
    if not isinstance(standardize, (bool, _np_for_bool.bool_)):
        raise TypeError(
            f"standardize must be bool (or numpy bool); got {type(standardize).__name__}."
        )

    X_arr, _input_names = _coerce_input(X, dtype=dtype)
    validate_numeric_input(X_arr, name="X", allow_fp16=True)

    m = n_features // 2
    col_names = [f"{column_prefix}_cos_{i}" for i in range(m)] + [f"{column_prefix}_sin_{i}" for i in range(m)]

    # Mode A: OOF over a splitter. Each fold fits on its train slice, projects its held-out slice -- no row's own data informs its own scaler/bandwidth.
    if splitter is not None:
        if X_query is not None:
            raise ValueError("Pass either X_query (Mode B) or splitter (Mode A), not both.")
        if return_state:
            raise ValueError("return_state is not supported in Mode A (each fold fits its own state).")
        n = X_arr.shape[0]
        out = np.zeros((n, n_features), dtype=dtype)
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_arr)):
            _state = _rff_fit_state(X_arr[train_idx], seed=seed, n_features=n_features, sigma=sigma, standardize=standardize, column_prefix=column_prefix, dtype=dtype)
            out[val_idx] = _rff_project(_state, X_arr[val_idx], use_gpu=use_gpu, gpu_threshold=gpu_threshold, batch_rows=batch_rows, release_memory_after=release_memory_after)
        return pl.DataFrame({name: out[:, idx] for idx, name in enumerate(col_names)})

    # Mode B / default: fit on X (train) only.
    state = _rff_fit_state(X_arr, seed=seed, n_features=n_features, sigma=sigma, standardize=standardize, column_prefix=column_prefix, dtype=dtype)
    if X_query is not None:
        Xq_arr, _ = _coerce_input(X_query, dtype=dtype)
        validate_numeric_input(Xq_arr, name="X_query", allow_fp16=True)
        proj_input = Xq_arr
    else:
        proj_input = X_arr
    out = _rff_project(state, proj_input, use_gpu=use_gpu, gpu_threshold=gpu_threshold, batch_rows=batch_rows, release_memory_after=release_memory_after)
    df = pl.DataFrame({name: out[:, idx] for idx, name in enumerate(col_names)})
    if return_state:
        return df, state
    return df


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

    # Clamp against fp32 overflow. Compute the modulo directly in the caller's float dtype (typically float32) so we skip the prior float64 intermediate (80MB on N=10M when dtype=float32, accumulates GC pressure across ticker-group loops).
    _np_dtype = np.dtype(dtype)
    pos_f = np.fmod(pos_arr.astype(_np_dtype, copy=False), _np_dtype.type(1_000_000.0))

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
        # bench-attempt-rejected (2026-05-24): allow_copy=False polars hint raises unconditionally on any multi-column DataFrame even after rechunk() (each column
        # is a separate Arrow chunk on the to-numpy boundary). Try/except fallback adds branches without speedup. The dtype gate immediately below is already
        # correct (skips astype when arr.dtype already == dtype); astype(dtype, copy=False) is a 0.3 us no-op when dtypes match.
        arr = X.to_numpy()
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
    # Wave 28 P0 fix (2026-05-20): pre-fix used ``is True``/``is False``,
    # which silently rejected ``np.bool_(True)`` and ``np.bool_(False)``
    # from config dicts -- numpy bools fail the identity check and
    # fell through to the ``!= "auto"`` ValueError. Normalise: handle
    # the string "auto" first explicitly, then ``bool(use_gpu)``
    # accepts both Python bool and numpy bool uniformly.
    if isinstance(use_gpu, str):
        if use_gpu != "auto":
            raise ValueError(f"use_gpu must be True, False, or 'auto'; got {use_gpu!r}.")
        # Fall through to auto-dispatch below.
    else:
        _flag = bool(use_gpu)
        if not _flag:
            return False
        if not is_gpu_available():
            logger.info("use_gpu=True but GPU is not available; using CPU.")
            return False
        return True
    if not is_gpu_available():
        return False
    if threshold is not None:
        return work >= threshold
    # Auto without explicit threshold: per-host backend (numpy/cupy) for this
    # matmul ``work`` via the shared get_or_tune orchestrator; measurement-backed
    # threshold fallback. We still gate on live ``is_gpu_available()`` above, so a
    # ``cupy`` choice here only routes to GPU when the device is actually usable.
    return _RFF_SPEC.choose(work=int(work)) == "cupy"


# ---------------------------------------------------------------------
# rff_matmul backend selection (numpy vs cupy matmul) via kernel_tuning_cache
# ---------------------------------------------------------------------


# Source-code default crossover for the RFF projection matmul: below this total
# ``work = N * d * n_features`` the numpy/numba CPU path beats streaming cupy once
# the H2D + D2H round trip is amortised. The 5_000_000 * 256 was a documented
# placeholder ("replaces it after calibration"); it remains the fallback when no
# per-host kernel_tuning_cache entry exists yet.
_RFF_DEFAULT_WORK_THRESHOLD = 5_000_000 * 256

_RFF_SWEEP_WORK = [1_000_000, 8_000_000, 64_000_000, 512_000_000, 4_096_000_000]


def _rff_matmul_numpy(X, W, b, scale):
    """Reference CPU RFF projection matmul: ``scale * [cos(XW+b), sin(XW+b)]``.

    A plain numpy implementation (not the numba kernel) used purely as the sweep
    reference + correctness baseline; the production CPU path is ``rff_matmul_njit``,
    which computes the identical result."""
    proj = X @ W + b
    return np.concatenate([np.cos(proj), np.sin(proj)], axis=1) * scale


def _rff_matmul_cupy(X, W, b, scale):
    """cupy RFF projection matmul mirroring ``_rff_matmul_numpy`` on device."""
    import cupy as cp

    Xd = cp.asarray(X)
    Wd = cp.asarray(W)
    bd = cp.asarray(b)
    proj = Xd @ Wd + bd
    res = cp.concatenate([cp.cos(proj), cp.sin(proj)], axis=1) * scale
    return cp.asnumpy(res)


def _make_rff_inputs(work: int):
    """Representative matmul operands whose total work ``N * d * n_features`` ~= ``work``.

    Fixes ``d`` and ``n_features`` to representative RFF sizes (d=32, n_features=256
    so m=128) and derives ``N`` from the requested work; returns ``(X, W, b, scale)``
    matching the variant call signature."""
    d = 32
    n_features = 256
    m = n_features // 2
    n = max(8, int(work / (d * n_features)))
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)
    W = rng.standard_normal((d, m)).astype(np.float32)
    b = (rng.random(m) * 2.0 * np.pi).astype(np.float32)
    scale = float(np.sqrt(2.0 / n_features))
    return (X, W, b, scale)


def _run_rff_sweep() -> list:
    """Benchmark numpy-vs-cupy RFF matmul across a ``work`` grid -> backend_choice
    regions (fastest equivalent backend per band). The cupy variant is included
    only when a usable GPU is present. float32 GPU vs CPU matmul + sin/cos agree to
    a small relative tolerance, so the equivalence tolerance is loosened."""
    from pyutilz.dev.benchmarking import sweep_backend_crossover

    variants = {"numpy": _rff_matmul_numpy}
    if is_gpu_available():
        variants["cupy"] = _rff_matmul_cupy
    return sweep_backend_crossover(
        variants,
        _RFF_SWEEP_WORK,
        _make_rff_inputs,
        "work",
        reference="numpy",
        repeats=5,
        equiv_rtol=1e-4,
        equiv_atol=1e-6,
    )


def _rff_fallback_choice(work: int = 0, **_dims) -> str:
    """Pre-sweep heuristic (the spec's dynamic fallback callable): cupy above the
    work threshold when a GPU is available, else numpy.

    ``work`` defaults to 0 and extra dims are absorbed so the callable never raises
    when get_or_tune invokes the fallback with no/partial dims (e.g. an offline
    sweep with an empty dims probe) -- a missing ``work`` simply means "no GPU win
    assumed" -> numpy, never a TypeError that aborts the whole sweep/dispatch."""
    if is_gpu_available() and work >= _RFF_DEFAULT_WORK_THRESHOLD:
        return "cupy"
    return "numpy"


# Register with the @kernel_tuner registry so retune_all / mlframe-tune-kernels
# discover + batch-tune rff_matmul. GPU-capable (numpy CPU vs cupy matmul).
from pyutilz.performance.kernel_tuning.registry import kernel_tuner

_RFF_SPEC = kernel_tuner(
    kernel_name="rff_matmul",
    variant_fns=(_rff_matmul_numpy, _rff_matmul_cupy),  # both always-defined -> auto-invalidate
    tuner=_run_rff_sweep,
    axes={"work": list(_RFF_SWEEP_WORK)},
    fallback=_rff_fallback_choice,  # callable (work) -> str
    gpu_capable=True,
    salt=1,
    cli_label="rff_matmul",
)
