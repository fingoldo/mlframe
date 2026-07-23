"""Small standalone utility functions shared across mlframe: infinity sanitisation for pandas/polars/numpy frames, ES-related model introspection, and interactive diagnostic printers (BLAS/LAPACK install check, sklearn classifier listing, system RAM usage)."""
from __future__ import annotations

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import hashlib
import logging

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from mlframe.config import XGBOOST_MODEL_TYPES, LGBM_MODEL_TYPES, CATBOOST_MODEL_TYPES

import psutil

import polars.selectors as cs
import pandas as pd, polars as pl, numpy as np

from sklearn.pipeline import Pipeline

import pyutilz.polarslib as pllib

########################################################################################################################################################################################################################################
# Helper functions
########################################################################################################################################################################################################################################


def derive_seed(master_seed: int, key: str) -> int:
    """Derive a stable, reproducible sub-seed for ``key`` from a single ``master_seed``.

    Threading one ``random_state`` through several independent randomness sources (e.g. MI sampling,
    a CV split, a bootstrap draw) correlates them -- an "easy" sample for one source can coincide with
    an "easy" split for another. Hashing ``(master_seed, key)`` together breaks that correlation while
    staying deterministic (same master seed + key -> same sub-seed, stable across processes/runs
    regardless of ``PYTHONHASHSEED``, unlike Python's builtin salted ``hash()``). Range is
    ``[0, 2**31 - 1)`` for sklearn/numpy splitter compatibility (both reject negative/oversized seeds).

    This is the canonical seed-derivation helper for mlframe; do not hand-roll a new one -- see
    ``training.honest_diagnostics._derive_seed`` and ``training.composite.ensemble.derive_seeds``,
    which delegate here.
    """
    h = hashlib.blake2b(f"{int(master_seed)}|{key}".encode(), digest_size=4).digest()
    return int.from_bytes(h, "big") % (2**31 - 1)


def MakeSureBlasAndLaPackAreInstalled():
    """Print numpy's detected BLAS/LAPACK backend info (interactive diagnostic to confirm numpy is linked against an optimized linear-algebra library rather than the slow reference implementation)."""
    from numpy.distutils.system_info import get_info

    print(get_info("blas_opt"))  # noqa: T201 -- interactive diagnostic utility, this IS the function's job
    print(get_info("lapack_opt"))  # noqa: T201


def ListAllSkLearnClassifiers():
    """Print module path + name of every registered sklearn estimator whose name contains "Class" (interactive discovery helper for classifier-family estimators)."""
    from sklearn.utils.testing import all_estimators

    for name, Class in all_estimators():
        if name.find("Class") > 0:
            print(Class.__module__, name)  # noqa: T201 -- interactive diagnostic utility, this IS the function's job


def has_early_stopping_support(model_type: str) -> bool:
    """Return True if ``model_type`` is one of the XGBoost / LightGBM / CatBoost model-type strings, i.e. a booster family that supports validation-set early stopping (as opposed to sklearn-style estimators that don't)."""
    if model_type in XGBOOST_MODEL_TYPES + LGBM_MODEL_TYPES + CATBOOST_MODEL_TYPES:
        return True
    else:
        return False


def get_model_best_iter(model: object) -> int | None:
    """Extracts ES best iteration number from a model.

    Unwraps both ``sklearn.Pipeline`` AND ``CompositeTargetEstimator``
    (and any other wrapper exposing ``.estimator_``) so composite-target
    components also report their inner booster's best_iter. Tries the
    underscore-suffixed sklearn-canonical name FIRST (this is what
    CatBoost/XGBoost ship in modern versions), then no-suffix, then
    ``best_epoch`` (Keras-style). Returns ``None`` (not 0) when nothing
    is exposed so callers can distinguish "ES didn't fire" from "ES
    fired at iter 0".

    CatBoost fallback: when ES didn't trigger but the model trained to
    full iterations, ``tree_count_`` is the number of trees built and
    is a reasonable @iter substitute for chart titles.
    """
    real_model = model
    for _ in range(8):
        if isinstance(real_model, Pipeline):
            real_model = real_model.steps[-1][1]
            continue
        _has_iter = hasattr(real_model, "best_iteration_") or hasattr(real_model, "best_iteration") or hasattr(real_model, "best_epoch")
        # sklearn TransformedTargetRegressor (and _TTRWithEvalSetScaling
        # subclass) exposes the inner model via ``.regressor_`` not
        # ``.estimator_``; without this branch PytorchLightningRegressor's
        # ``best_epoch`` stays invisible to chart titles when wrapped in
        # _TTRWithEvalSetScaling and the report header loses @iter=N.
        if not _has_iter and hasattr(real_model, "regressor_"):
            real_model = real_model.regressor_
            continue
        if not _has_iter and hasattr(real_model, "estimator_"):
            real_model = real_model.estimator_
            continue
        break
    for field in ("best_iteration_", "best_iteration", "best_epoch"):
        val = getattr(real_model, field, None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                # A bad value on THIS field (e.g. a model exposing best_iteration_="n/a") must fall
                # through to the next field / tree_count_ fallback / None, matching this function's own
                # docstring -- re-calling int(val) here re-raised the identical exception instead of
                # degrading gracefully.
                continue
    tree_count = getattr(real_model, "tree_count_", None)
    if tree_count is not None:
        try:
            return int(tree_count)
        except (TypeError, ValueError):
            pass
    return None


def ensure_no_infinity_np(arr: np.ndarray, nans_filler: float = 0, verbose: int = 1) -> np.ndarray:
    """``ensure_no_infinity`` for a raw ndarray (some model pre-pipelines -- e.g. PytorchLightning's
    torch-tensor prep -- materialise the frame to numpy BEFORE the generic pre-fit infinity check runs).
    Mirrors ``ensure_no_infinity_pd``'s contract: replace +-inf with ``nans_filler`` IN PLACE (integer /
    bool arrays can't hold inf, so they're a no-op) and warn once if any were found."""
    if not np.issubdtype(arr.dtype, np.floating):
        return arr
    mask = np.isinf(arr)
    if mask.any():
        arr[mask] = nans_filler
        if verbose:
            logger.warning("ndarray contained infinity in %s cell(s); replaced with %s.", f"{int(mask.sum()):_}", nans_filler)
    return arr


def ensure_no_infinity(df: "pd.DataFrame | pl.DataFrame | np.ndarray", num_cols_only: bool = True) -> "pd.DataFrame | pl.DataFrame | np.ndarray | None":
    """Dispatch to the pandas / polars / numpy specific ``ensure_no_infinity_*`` implementation based on the runtime type of ``df``. Raises ``TypeError`` for any other carrier type."""
    if isinstance(df, pd.DataFrame):
        return ensure_no_infinity_pd(df=df, num_cols_only=num_cols_only)
    elif isinstance(df, pl.DataFrame):
        return ensure_no_infinity_pl(df=df, num_cols_only=num_cols_only)
    elif isinstance(df, np.ndarray):
        return ensure_no_infinity_np(df)
    raise TypeError(f"ensure_no_infinity expects a pandas or polars DataFrame or a numpy ndarray; got {type(df).__name__}.")


def ensure_no_infinity_pl(df: pl.DataFrame, num_cols_only: bool = True, nans_filler: float = 0, verbose: int = 1) -> pl.DataFrame:
    """``ensure_no_infinity`` for a polars DataFrame: replace ±inf with ``nans_filler`` in the columns selected by ``num_cols_only`` (numeric-only by default), warn once with the affected column names, and return a new frame (polars ``with_columns`` is not in-place)."""
    cols = cs.all() if not num_cols_only else cs.numeric()
    inf_mask = df.select(cols.is_infinite().any())
    # No matching columns (e.g. an all-categorical / all-text frame with num_cols_only=True) -> nothing can hold infinity.
    # Guarding here is required because transpose() with an explicit column_names list raises ShapeError on a 0-column frame
    # (the transposed row count is 0 but column_names=["is_inf"] has length 1).
    if inf_mask.width == 0:
        return df
    inf_cols = [c for c, flag in zip(inf_mask.columns, inf_mask.row(0)) if flag]

    if len(inf_cols) > 0:
        df = df.with_columns(pllib.clean_numeric(pl.col(inf_cols), nans_filler=nans_filler))

        if verbose:
            logger.warning("Some factors (%s) contained infinity: %s", f"{len(inf_cols):_}", ", ".join(inf_cols))

    return df


def ensure_no_infinity_pd(df: pd.DataFrame, num_cols_only: bool = True, nans_filler: float = 0, verbose: int = 1) -> pd.DataFrame:
    """Replace ±inf with ``nans_filler`` in float columns.

    Only **float** columns can hold infinity, so integer / boolean / category /
    datetime columns are skipped -- including pandas nullable extension types
    like ``Int8`` (which the polars->pandas bridge produces for nullable
    Boolean inputs since 2026-04-23). Earlier versions called
    ``df[num_cols].to_numpy()`` over all numeric dtypes, which on an Int8
    column with ``pd.NA`` materializes a Python-object array and then crashed
    with ``TypeError: ufunc 'isinf' not supported for the input types``
    (2026-04-23 LGB prod regression: ``hide_budget`` was Int8 + pd.NA).

    Float extension dtypes (``Float32Dtype`` / ``Float64Dtype`` with
    ``pd.NA``) are handled per-column with ``to_numpy(dtype=float, na_value=
    np.nan)`` so the isinf check works on a real float array.

    bench-attempt-rejected (2026-05-21, c0095 / iter141): numba @njit
    short-circuit walk to skip the bool-array allocation is 24% SLOWER
    on the CLEAN production case (n=1M, no inf) because numpy's
    np.isinf().any() is SIMD-vectorised inside the C kernel. Bench:
    profiling/bench_any_isinf_short_circuit.py.

    bench-attempt-rejected (2026-05-28, c0060): block-vectorising the
    plain-float columns via one ``df[float_cols].to_numpy()`` + a single
    ``np.isinf(block).any(axis=0)`` (to cut per-column Python/dtype-check
    dispatch) is 6.2x SLOWER on a clean 1M x 30 float64 frame (23.8ms ->
    148ms). pandas is column-major, so the block to_numpy() forces a full
    transpose+copy into one C-contiguous (n, k) array and the (n, k) bool
    intermediate, whereas the per-column path reads each column as a cheap
    contiguous view. Keep the per-column loop.
    """
    # Restrict to float-only columns. Integer + bool can't hold inf, so
    # there's no work to do for them; skipping avoids the extension-dtype
    # to_numpy() pitfall above.
    inf_cols = []
    candidate_cols = list(df.columns) if not num_cols_only else [c for c in df.columns if pd.api.types.is_float_dtype(df[c].dtype)]
    if not candidate_cols:
        return df

    for col in candidate_cols:
        s = df[col]
        dt = s.dtype
        try:
            if pd.api.types.is_extension_array_dtype(dt):
                # Float64Dtype / Float32Dtype with pd.NA -> cast to numpy
                # float, replacing pd.NA with NaN. NaN is not inf, so this
                # doesn't change the inf-detection answer.
                arr = s.to_numpy(dtype=np.float64, na_value=np.nan)
            else:
                arr = s.to_numpy()
            if not np.issubdtype(arr.dtype, np.floating):
                # Should not happen given the float-only filter, but guards
                # against unexpected object-dtype slip-throughs from upstream.
                continue
            if np.isinf(arr).any():
                inf_cols.append(col)
        except (TypeError, ValueError) as exc:
            # Don't let a single weird column abort the whole pre-fit check --
            # log and move on. The column will simply not be sanitised.
            if verbose:
                logger.warning(
                    "ensure_no_infinity_pd: skipped %r (dtype=%s) -- "
                    "isinf check failed: %s",
                    col, dt, exc,
                )

    if inf_cols:
        for col in inf_cols:
            df[col] = np.nan_to_num(df[col], posinf=nans_filler, neginf=nans_filler)
        if verbose:
            logger.warning("Some factors (%s) contained infinity: %s", f"{len(inf_cols):_}", ", ".join(inf_cols))
    return df


def show_sys_ram_usage():
    """Print total/available/used/free system RAM (GB) and used-percentage via ``psutil.virtual_memory`` (interactive diagnostic for spotting OOM risk before a large fit)."""

    mem = psutil.virtual_memory()

    print(f"Total: {mem.total / 1e9:.2f} GB")  # noqa: T201 -- interactive diagnostic utility, this IS the function's job
    print(f"Available: {mem.available / 1e9:.2f} GB")  # noqa: T201
    print(f"Used: {mem.used / 1e9:.2f} GB")  # noqa: T201
    print(f"Free: {mem.free / 1e9:.2f} GB")  # noqa: T201
    print(f"Memory Usage: {mem.percent}%")  # noqa: T201
