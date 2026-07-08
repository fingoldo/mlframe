"""CatBoost / Polars helpers extracted from ``trainer.py``.

CB-specific training utilities: Pool caching, GPU probing,
Polars nullable-categorical diagnostics and filling.

Key functions:
- ``_maybe_get_or_build_cb_pool`` — cached CB Pool reuse across weight schemas
- ``_maybe_rewrite_eval_set_as_cb_pool`` — replace eval_set DataFrames with cached Pools
- ``_polars_schema_diagnostic`` — log dtypes/nullability before CB training
- ``_polars_fill_null_in_categorical`` — fill nulls in cat columns (prevent CB crash)
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from mlframe.config import CATBOOST_MODEL_TYPES
from pyutilz.system import get_gpuinfo_gpu_info

from ..phases import phase

logger = logging.getLogger(__name__)

# Guards concurrent first-time probing of the GPU info cache and the CB-GPU
# usable cache. Two parallel suite invocations can otherwise both pay the
# `nvidia-smi` subprocess cost or duplicate the tiny CB probe fit; the lock
# costs nothing on the hot path (single test + return). RLock (not Lock):
# _cb_gpu_usable acquires this lock and then calls _cached_gpu_info which
# acquires it again; with a plain Lock the second acquire deadlocks on the
# very first probe (before _GPU_INFO_PROBED is True).
_GPU_PROBE_LOCK = threading.RLock()


def _label_float32_is_lossless(arr: np.ndarray) -> bool:
    """True iff every finite value in ``arr`` round-trips through float32 exactly.

    float32 has only ~7 significant digits; a large-magnitude regression target
    (int counts/IDs > 2**24, prices like 1234567.89) loses precision silently and
    collapses adjacent label values, biasing the fit. CatBoost accepts float64
    labels natively, so we only downcast when it is bit-safe.
    """
    finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr
    if finite.size == 0:
        return True
    return bool(np.array_equal(finite.astype(np.float32).astype(np.float64), finite.astype(np.float64)))


def _coerce_label_for_cb_pool(target):
    """Convert target to dtype/shape CatBoost Pool expects.
    CB infers loss family from first label cell; crashes on Python list cells (polars List->pandas object roundtrip).
    Stack object-of-arrays into 2-D (N,K). Downcast to float32 only when lossless; else keep float64 (CB accepts it).
    Falls through on failure."""
    arr = np.asarray(target)
    if arr.dtype == object and arr.ndim == 1 and arr.shape[0] > 0:
        _first = arr[0]
        if hasattr(_first, "shape") or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes))):
            try:
                arr = np.stack([np.asarray(c) for c in arr], axis=0)
            except Exception as e:
                logger.debug("swallowed exception in _cb_pool.py: %s", e)
                pass
    if arr.dtype.kind in ("i", "u", "b", "f") and arr.dtype != np.float32:
        arr = arr.astype(np.float32) if _label_float32_is_lossless(arr) else arr.astype(np.float64)
    return arr


def _polars_schema_diagnostic(
    df: pl.DataFrame,
    cat_features: list[str] | None = None,
    text_features: list[str] | None = None,
    max_cols_logged: int = 30,
) -> str:
    """Render a per-column diagnostic of a Polars DataFrame for CatBoost
    fastpath incidents.

    CatBoost 1.2.x's `_set_features_order_data_polars_categorical_column`
    is a Cython fused cpdef with a finite set of compiled dispatch
    overloads. When a column's dtype variant isn't in the table, it
    raises the opaque ``TypeError: No matching signature found`` with no
    indication of which column is at fault. This helper dumps every
    column's dtype, its role (cat / text / other), and the specific
    sub-variant that matters for CB's dispatcher:

      - ``pl.Categorical`` with a validity bitmap (``null_count > 0``)
        -- verified 2026-04-19 culprit: CB 1.2.10 has no dispatch overload
        for nullable Categorical. `_polars_nullable_categorical_cols` +
        `fill_null` is the fix.
      - ``pl.Enum`` without nulls -- empirically works on the CB 1.2.10
        fastpath (reproduced 2026-04-21 -- fit + eval_set succeed). Still
        reported in the dump for visibility, but not automatically
        flagged as the culprit.
      - ``pl.List[...]`` -- nested types not supported in fastpath.

    Keeps output compact: logs up to ``max_cols_logged`` cat_features
    verbatim; the rest are summarised by dtype count. Safe to call in
    error paths -- swallows exceptions and returns a note instead.
    """
    try:
        import polars as _pl

        cat_set = set(cat_features or [])
        text_set = set(text_features or [])
        lines: list[str] = []
        enum_cat_cols: list[str] = []  # the smoking-gun list

        cat_cols = [c for c in df.columns if c in cat_set]
        # Prioritise cat_features for full logging (they're the usual
        # culprit); summarise non-cat columns.
        shown = 0
        for col in cat_cols[:max_cols_logged]:
            dt = df.schema.get(col)
            role = "cat"
            variant = str(dt)
            if isinstance(dt, _pl.Enum):
                variant = f"Enum(n_values={len(dt.categories)})"
                enum_cat_cols.append(col)
            elif dt == _pl.Categorical or (hasattr(_pl, "Categorical") and isinstance(dt, type(_pl.Categorical))):
                try:
                    ordering = getattr(dt, "ordering", "?")
                    variant = f"Categorical(ordering={ordering!r})"
                except Exception:
                    variant = "Categorical"
            try:
                nu = df[col].n_unique()
                nn = int(df[col].null_count())
                lines.append(f"    {col} [{role}]: {variant}, n_unique={nu}, nulls={nn}")
            except Exception:
                lines.append(f"    {col} [{role}]: {variant}")
            shown += 1

        if len(cat_cols) > max_cols_logged:
            lines.append(f"    ... +{len(cat_cols) - max_cols_logged} more cat_features")

        # Roll-up of everything else by dtype.
        other_dtype_counts: dict[str, int] = {}
        for col in df.columns:
            if col in cat_set or col in text_set:
                continue
            dt_str = str(df.schema.get(col))
            other_dtype_counts[dt_str] = other_dtype_counts.get(dt_str, 0) + 1
        if other_dtype_counts:
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(other_dtype_counts.items()))
            lines.append(f"    (non-cat, non-text cols by dtype) {summary}")

        if text_set:
            text_cols_in_df = [c for c in df.columns if c in text_set]
            text_dt_counts: dict[str, int] = {}
            for col in text_cols_in_df:
                dt_str = str(df.schema.get(col))
                text_dt_counts[dt_str] = text_dt_counts.get(dt_str, 0) + 1
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(text_dt_counts.items()))
            lines.append(f"    (text_features by dtype) {summary}")

        nullable_cat_cols = [c for c in cat_cols if c in df.columns and int(df[c].null_count()) > 0]
        header = f"  Polars schema diagnostic for {df.shape[0]:_}x{df.shape[1]}:"
        if nullable_cat_cols:
            header += (
                f"\n  [!] cat_features with null values: {nullable_cat_cols}. "
                "CatBoost 1.2.x Polars fastpath has no dispatch overload for "
                "Categorical with a validity bitmap; this is the most common "
                "cause of 'No matching signature found'. Fix: "
                "fill_null('__MISSING__') before fit."
            )
        elif enum_cat_cols:
            header += (
                f"\n  (info) cat_features contain pl.Enum columns: {enum_cat_cols}. "
                "Empirically compatible with CB 1.2.10 fastpath when nulls are "
                "filled; reported for visibility only."
            )
        return header + "\n" + "\n".join(lines)
    except Exception as _diag_err:
        return f"  (schema diagnostic failed: {_diag_err!r})"


def _polars_nullable_categorical_cols(df: Any, cat_features: list[str] | None = None) -> list[str]:
    """Return cat_feature column names with ``null_count > 0`` -- the
    set of columns that trigger CatBoost 1.2.x's Polars fastpath
    dispatch miss.

    Root cause (verified 2026-04-19 via direct repro in
    ``bench_polars_cb_nullfrac.py``): CatBoost 1.2.10's
    ``_set_features_order_data_polars_categorical_column`` (Cython
    fused cpdef) has no dispatch signature for a Polars Categorical
    column carrying a validity bitmap. A single null anywhere in any
    cat_feature raises ``TypeError: No matching signature found``.
    Fix: ``pl.col(c).fill_null("__MISSING__")`` before fit -- Polars
    auto-extends the category dict, the column loses its validity
    bitmap, CB's fastpath matches the non-nullable signature.

    Null-fraction sweep:
        0.0   -> OK        0.5   -> FAIL
        0.1   -> FAIL      0.99  -> FAIL
                          1.0   -> FAIL

    Performance: uses ``df.select(cat_cols).null_count()`` which runs a
    SINGLE polars query -- one scan over the selected columns, polars
    computes per-column null counts in parallel. The previous
    per-column implementation (``df[c].null_count()`` in a Python loop)
    cost N separate queries and showed up in prod profiling on
    810k-row frames.

    Args:
        df: Polars DataFrame.
        cat_features: Column names to consider. If None, inspects all
            ``pl.Categorical`` / ``pl.Enum`` columns in the schema.

    Returns:
        List of nullable Categorical column names (order-preserving
        against ``cat_features`` when provided). Empty list on any
        exception or non-Polars input -- callers can use the list's
        truthiness directly, so the function doubles as a boolean
        detector without requiring a separate wrapper.
    """
    try:
        import polars as _pl

        if not isinstance(df, _pl.DataFrame):
            return []

        schema = df.schema

        # 2026-04-23: extended to include pl.Utf8 / pl.String. Raw Utf8
        # cat_features with nulls trigger the same CB 'Invalid type for
        # cat_feature ... NaN' error on the Polars fastpath -- the
        # fill_null('__MISSING__') pre-fit pass must cover them too.
        # Fuzz c0061/c0084/c0096 (cb + polars_utf8 + nulls) all crashed
        # because Utf8 cols weren't in this candidate list.
        def _is_cat_like(dt):
            return dt == _pl.Categorical or dt == _pl.Utf8 or dt == _pl.String or (hasattr(_pl, "Enum") and isinstance(dt, _pl.Enum))

        if cat_features:
            candidate = [c for c in cat_features if c in schema and _is_cat_like(schema[c])]
        else:
            candidate = [name for name, dtype in schema.items() if dtype == _pl.Categorical or (hasattr(_pl, "Enum") and isinstance(dtype, _pl.Enum))]
        if not candidate:
            return []

        # SINGLE-PASS: df.select([...]).null_count() returns a 1-row DF
        # with per-column counts. All cat_features' null counts computed
        # in one scan. Previous per-column loop was N separate queries.
        counts_row = df.select(candidate).null_count().row(0)
        return [c for c, n in zip(candidate, counts_row) if n > 0]
    except Exception:
        return []


def _polars_df_has_null_in_categorical(df: Any, cat_features: list[str] | None = None) -> bool:
    """Boolean wrapper around ``_polars_nullable_categorical_cols`` --
    kept for callers that only need the yes/no answer."""
    return bool(_polars_nullable_categorical_cols(df, cat_features=cat_features))


def _polars_fill_null_in_categorical(
    df: Any,
    nullable_cat_cols: list[str],
    sentinel: str = "__MISSING__",
) -> Any:
    """Apply ``pl.col(c).fill_null(sentinel)`` across the listed
    Categorical columns on a Polars DataFrame.

    Separated out so the same expression set can be reused across
    train / val / test (same sentinel -> same category code across
    splits) without rebuilding the expr list per split.

    Returns df unchanged if ``nullable_cat_cols`` is empty or df is
    not a Polars DataFrame -- caller can unconditionally wrap
    train/val/test without pre-checking.

    2026-04-23 (fuzz c0088 / c0121): ``fill_null(sentinel)`` on a
    ``pl.Enum`` whose category list does NOT already include the
    sentinel is a SILENT NO-OP in polars 1.40 -- no error, no warning,
    nulls survive. The caller then hands the still-nullable Enum to
    CB's pandas fallback, which converts null->NaN and crashes with
    ``Invalid type for cat_feature ... =NaN``. Guard: for Enum
    columns we rebuild the Enum with the sentinel appended BEFORE
    filling. For ``pl.Categorical`` the Arrow-level dict auto-extends
    on fill_null, and for ``pl.Utf8/String`` no category list
    constraint applies, so those paths stay as-is.
    """
    try:
        import polars as _pl

        if not nullable_cat_cols or not isinstance(df, _pl.DataFrame):
            return df
        fill_exprs = []
        for c in nullable_cat_cols:
            dt = df.schema.get(c)
            # Enum: rebuild the category list to include the sentinel,
            # cast, THEN fill. Without the cast step the fill is a no-op.
            if dt is not None and hasattr(_pl, "Enum") and isinstance(dt, _pl.Enum):
                orig_cats = list(dt.categories)
                if sentinel not in orig_cats:
                    new_enum = _pl.Enum(orig_cats + [sentinel])
                    fill_exprs.append(_pl.col(c).cast(new_enum).fill_null(sentinel).alias(c))
                    continue
                # Enum already allowed the sentinel -- plain fill_null works.
            fill_exprs.append(_pl.col(c).fill_null(sentinel))
        return df.with_columns(fill_exprs)
    except Exception:
        return df


def _recover_cb_feature_names(model: Any) -> tuple[list[str], list[str]]:
    """Extract (cat_features, text_features) as column-name lists from a
    fitted CatBoost model.

    At predict time we don't have the original Python-side cat_features /
    text_features lists -- the caller is evaluation code with no knowledge
    of how the model was trained. CatBoost exposes its internal
    per-feature metadata via:
      - ``_get_cat_feature_indices()``  -- integer indices into feature_names_
      - ``_get_text_feature_indices()`` -- ditto
      - ``feature_names_``              -- list of column names

    Returns ``([], [])`` on any failure (e.g. non-fitted model, non-CB
    estimator, older CB builds without those private hooks) -- callers
    wrap the fallback so missing names just means a less-specific prep
    path, not a crash.
    """
    try:
        feat_names = list(getattr(model, "feature_names_", []) or [])
        cat_idx: list = getattr(model, "_get_cat_feature_indices", lambda: [])() or []
        text_idx: list = getattr(model, "_get_text_feature_indices", lambda: [])() or []
        if not feat_names:
            return [], []
        cat_feat = [feat_names[i] for i in cat_idx if 0 <= i < len(feat_names)]
        text_feat = [feat_names[i] for i in text_idx if 0 <= i < len(feat_names)]
        return cat_feat, text_feat
    except Exception:
        return [], []


def _predict_with_fallback(
    model: Any,
    X: Any,
    method: str = "predict",
    verbose: bool = False,
) -> np.ndarray:
    """Call ``model.{method}(X)`` with guards for common edge-cases.

    Guards (ordered by cost, cheapest first):
    1. LGBM Polars → pandas auto-convert
    2. CB val Pool cache hit (saves 50-70s rebuild on large frames)
    3. CB sticky-pandas short-circuit (skip Polars dispatch after first miss)
    4. NaN safety net (one-shot impute+scale for NaN-intolerant models)
    5. CB Polars dispatch-miss fallback (TypeError → pandas retry)

    Symmetric to ``_train_model_with_fallback`` for CB paths.
    See ``_predict_guards.py`` for per-guard rationale and incident history.
    """
    fn = getattr(model, method, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"{type(model).__name__} has no callable {method!r}")
    n_rows = len(X) if hasattr(X, "__len__") else None
    _model_type: str = type(model).__name__

    # ── Import the guard helpers ──────────────────────────────────────
    from .._predict_guards import (
        _apply_nan_guard,
        _cb_polars_to_pandas,
        _cb_val_pool_cache_lookup,
        _ensure_lgbm_gets_pandas,
        _pl_DataFrame,
    )

    # ── 0. CB cat-feature NaN sentinel-fill at predict time ──────────
    # CatBoost.predict() rejects category-dtype columns containing NaN
    # cells with "Invalid type for cat_feature ... NaN: cat_features must
    # be integer or string". The OOV-null mode of the joint train+val
    # categorical alignment leaves val/test rows with values not in the
    # joint union as NaN; without this guard predict crashes the moment a
    # held-out row carries an unseen cat value. Mirror the training-time
    # ``__MISSING__`` fill the suite uses at fit time.
    if _model_type in CATBOOST_MODEL_TYPES and isinstance(X, pd.DataFrame):
        try:
            _feat_names_list = list(getattr(model, "feature_names_", []) or [])
            _cat_idx_list = list(getattr(model, "_get_cat_feature_indices", lambda: [])() or [])
            _cat_names = [_feat_names_list[i] for i in _cat_idx_list if 0 <= i < len(_feat_names_list)]
            _cat_in_df = [c for c in _cat_names if c in X.columns]
            if _cat_in_df:
                _has_any_nan_in_cat = any(X[c].isna().any() for c in _cat_in_df)
                if _has_any_nan_in_cat:
                    # Shallow copy: only the cat columns carrying NaN are reassigned below; deep-copying a 100+ GB predict frame to mutate a few columns OOMs. ``deep=False`` shares untouched buffers, caller frame unmutated.
                    X = X.copy(deep=False) if not getattr(X, "_mlframe_filled", False) else X
                    for _c in _cat_in_df:
                        if X[_c].isna().any():
                            _orig = X[_c]
                            X[_c] = _orig.astype("string").fillna("__MISSING__")
                            if isinstance(_orig.dtype, pd.CategoricalDtype):
                                X[_c] = X[_c].astype("category")
                    X._mlframe_filled = True
        except Exception as e:
            logger.debug("swallowed exception in _cb_pool.py: %s", e)
            pass

    # ── 1. LGBM Polars → pandas auto-convert ──────────────────────────
    X = _ensure_lgbm_gets_pandas(model, X, method)

    # ── 2. CB val Pool cache lookup ────────────────────────────────────
    if _model_type in CATBOOST_MODEL_TYPES and hasattr(X, "columns") and hasattr(X, "shape"):
        try:
            _hit = _cb_val_pool_cache_lookup(X, method)
            if _hit is not None:
                logger.info(
                    "[cb-val-pool-reuse] %s hit on cached val Pool -- " "skipping redundant Pool rebuild.",
                    method,
                )
                with phase(method, model=_model_type, n_rows=n_rows):
                    return np.asarray(fn(_hit))
        except (AttributeError, KeyError, TypeError, ValueError) as _exc:
            logger.debug(
                "[cb-val-pool-reuse] %s cache probe failed (%s: %s); "
                "falling through.", method, type(_exc).__name__, _exc,
            )

    # ── 3. Sticky-pandas short-circuit (previous Polars miss) ──────────
    _pl_df = _pl_DataFrame()
    _is_cb = _model_type in CATBOOST_MODEL_TYPES
    if _pl_df is not type(None) and isinstance(X, _pl_df) and _is_cb and getattr(model, "_mlframe_polars_fastpath_broken", False):
        X_pd = _cb_polars_to_pandas(model, X, method, verbose=verbose)
        with phase(method, model=_model_type, n_rows=n_rows):
            return np.asarray(fn(X_pd))

    # ── 4. Normal path (with NaN guard + CB Polars fallback) ──────────
    try:
        with phase(method, model=_model_type, n_rows=n_rows):
            result = fn(X)
        # Sample first 500 rows — non-finite values in output signal
        # NaN input silently propagated by a NaN-tolerant model.
        if hasattr(result, "dtype"):
            _r_sample = result[:500] if hasattr(result, "__getitem__") else result
            if not np.all(np.isfinite(_r_sample)):
                logger.warning(
                    "[NaN-guard] %s.%s returned non-finite predictions "
                    "(likely NaN input silently propagated).  Applying "
                    "one-shot imputation + scaling before retry.",
                    _model_type, method,
                )
                return np.asarray(_apply_nan_guard(model, X, fn, n_rows))
        return np.asarray(result)
    except ValueError as e:
        if "NaN" not in str(e) and "contains NaN" not in str(e):
            raise
        return np.asarray(_apply_nan_guard(model, X, fn, n_rows))
    except TypeError as e:
        if not (_is_cb and _pl_df is not type(None) and isinstance(X, _pl_df) and "No matching signature found" in str(e)):
            raise
        logger.warning(
            "CatBoost %s Polars fastpath rejected the data (%s); " "converting to pandas and retrying.",
            method,
            str(e).splitlines()[-1][:240],
        )
        try:
            model._mlframe_polars_fastpath_broken = True
        except AttributeError:
            pass
        X_pd = _cb_polars_to_pandas(model, X, method, verbose=verbose)
        with phase(method, model=_model_type, n_rows=n_rows):
            return np.asarray(fn(X_pd))


# Fix 9.4.3 + Fix Orch-1: process-wide CatBoost Pool cache. Keys: tuple
# of (id(df), cols, shape, sorted cat/text/embedding features). Values:
# the Pool object whose label/weight we mutate in place between fits.
# The train-side cache survives weight schemas and same-type targets;
# the val-side cache adds ~2x on top (val eval_set is rebuilt on every
# fit by _setup_eval_set). Entries stay valid as long as the Python df
# reference is alive; the next train_mlframe_models_suite call produces
# fresh dfs with new id()s and the cache naturally evolves (also
# explicitly cleared at suite entry in core.py). Deliberate plain dict
# (not WeakValueDictionary) so transient GC between weight iterations
# doesn't flush the Pool we're about to reuse.
# 2026-05-13 refactor: _CB_VAL_POOL_CACHE lives in _predict_guards.py
# (shared between fit-time populate in _maybe_get_or_build_cb_pool and
# predict-time lookup in _predict_with_fallback).
from .._predict_guards import _CB_VAL_POOL_CACHE  # noqa: E402,F401
from ..pipeline import (  # noqa: E402,F401
    _apply_pre_pipeline_transforms,
    _extract_feature_selector,
    _is_fitted,
    _multilabel_target_to_1d_for_supervised_encoders,
    _passthrough_cols_fit_transform,
    _pipeline_signature_for_cache,
    _pre_pipeline_cache_clear,
    _pre_pipeline_cache_get,
    _pre_pipeline_cache_set,
    _prepare_test_split,
    _PRE_PIPELINE_CACHE,
    _PRE_PIPELINE_CACHE_LOCK,
    _PRE_PIPELINE_CACHE_MAX,
)

_CB_POOL_CACHE: dict[tuple, Any] = {}
_CB_POOL_CACHE_MAX_ENTRIES = 16  # hard cap per cache; ring-buffer eviction oldest-first


# 2026-04-29: cache the GPU probe result. ``get_gpuinfo_gpu_info`` shells
# out to ``nvidia-smi`` via GPUtil, which costs ~0.5s on Windows per call.
# ``configure_training_params`` invokes the probe once per
# ``train_mlframe_models_suite`` call, so a long-lived process running
# many suites pays the subprocess startup repeatedly. GPU topology
# doesn't change during a process lifetime (no hot-plug under CUDA), so
# a one-shot cache is safe. Override with ``MLFRAME_NO_GPU_INFO_CACHE=1``
# if a future use case needs live re-probing.
_GPU_INFO_CACHE: list | None = None
_GPU_INFO_PROBED: bool = False


def _cached_gpu_info() -> list:
    """Memoised wrapper over ``pyutilz.system.get_gpuinfo_gpu_info``.

    First call runs the real probe (nvidia-smi subprocess); subsequent
    calls return the cached list. Saves ~0.5s on every
    ``configure_training_params`` invocation past the first.

    Cheap pre-check via ``shutil.which("nvidia-smi")`` short-circuits the
    GPUtil import (which transitively pulls setuptools / _distutils_hack /
    distutils, ~0.7s cold) on machines without a CUDA toolkit. The
    distinction between "probed and empty" vs "not yet probed" is held
    in ``_GPU_INFO_PROBED`` so the empty-list cache hit also short-circuits.
    """
    global _GPU_INFO_CACHE, _GPU_INFO_PROBED
    # Fast lock-free read of already-probed cache; only enter the lock when
    # we actually need to probe so callers don't queue behind nvidia-smi.
    if _GPU_INFO_PROBED and not os.environ.get("MLFRAME_NO_GPU_INFO_CACHE"):
        return _GPU_INFO_CACHE or []
    with _GPU_PROBE_LOCK:
        # Re-check inside lock; another thread may have populated while we
        # waited.
        if _GPU_INFO_PROBED and not os.environ.get("MLFRAME_NO_GPU_INFO_CACHE"):
            return _GPU_INFO_CACHE or []
        import shutil
        if shutil.which("nvidia-smi") is None:
            _GPU_INFO_CACHE = []
            _GPU_INFO_PROBED = True
            return _GPU_INFO_CACHE
        result = list(get_gpuinfo_gpu_info())
        _GPU_INFO_CACHE = result
        _GPU_INFO_PROBED = True
        return result


_CB_GPU_USABLE_CACHE: bool | None = None


def _cb_gpu_usable() -> bool:
    """Verify the installed CatBoost wheel can actually fit on GPU.

    Decouples "machine has NVIDIA GPU" (``nvidia-smi`` on PATH) from
    "this CatBoost binary was compiled with CUDA support". Default
    catboost wheels from PyPI on Windows are CPU-only despite the host
    having a working CUDA toolkit; passing ``task_type="GPU"`` to such
    a wheel raises ``CatBoostError: Environment for task type [GPU]
    not found`` on every fit. Probe once via a tiny dummy fit; cache
    the result for the process lifetime.
    """
    global _CB_GPU_USABLE_CACHE
    if _CB_GPU_USABLE_CACHE is not None:
        return _CB_GPU_USABLE_CACHE
    # Serialize the probe across threads; the tiny CB GPU fit is otherwise
    # paid N times for N concurrent first-time callers.
    with _GPU_PROBE_LOCK:
        if _CB_GPU_USABLE_CACHE is not None:
            return _CB_GPU_USABLE_CACHE
        if not _cached_gpu_info():
            _CB_GPU_USABLE_CACHE = False
            return False
        # ``CUDA_VISIBLE_DEVICES`` set to "" or "-1" hides every device from
        # this process (the standard CPU-only / CI signal). The probe fit below
        # would spend ~4s spinning up CatBoost's CUDA backend only to fail
        # finding a device and return False anyway; short-circuit to that same
        # False result without paying the probe cost. ``nvidia-smi`` still sees
        # the physical card, so ``_cached_gpu_info`` above can't catch this. A
        # concrete device list ("0", "0,1") or an unset var falls through to the
        # real probe unchanged.
        _cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if _cvd is not None and _cvd.strip() in ("", "-1"):
            _CB_GPU_USABLE_CACHE = False
            return False
        try:
            from catboost import CatBoostRegressor
            import numpy as _np
            _probe = CatBoostRegressor(
                iterations=1, task_type="GPU", devices="0",
                allow_writing_files=False, verbose=False,
            )
            _probe.fit(_np.zeros((2, 1), dtype=_np.float32), _np.array([0.0, 1.0], dtype=_np.float32))
            _CB_GPU_USABLE_CACHE = True
        except Exception:
            _CB_GPU_USABLE_CACHE = False
        return _CB_GPU_USABLE_CACHE


def _cb_reuse_capable() -> bool:
    """True iff installed CatBoost Pool exposes both set_label and
    set_weight (the two mutators we rely on for in-place label/weight
    swap between fits)."""
    try:
        from catboost import Pool as _Pool
    except ImportError:
        return False
    return callable(getattr(_Pool, "set_label", None)) and callable(getattr(_Pool, "set_weight", None))


def _maybe_rewrite_eval_set_as_cb_pool(fit_params: dict[str, Any]) -> None:
    """Fix Orch-1 (2026-04-21): for CatBoost, rewrite
    ``fit_params['eval_set']`` from ``[(val_df, val_target)]`` /
    ``(val_df, val_target)`` to a cached ``catboost.Pool`` in place.

    Cache key mirrors the train-side cache (id(val_df) + cols + shape +
    cat/text/embedding features). On a cache hit, swap label + weight on
    the cached Pool; on miss, build a fresh Pool with float32-cast label
    (for CB set_label compatibility across classification targets).

    Called from ``_train_model_with_fallback`` AFTER the train-side
    reuse decision and AFTER ``_setup_eval_set`` has populated
    fit_params['eval_set']. Idempotent per fit: if eval_set already
    contains a Pool, leave it alone.
    """
    if not _cb_reuse_capable():
        return
    try:
        from catboost import Pool as _Pool
    except ImportError:
        return

    es = fit_params.get("eval_set")
    if es is None:
        return

    # Normalise to a list of (df, target) tuples.
    if isinstance(es, tuple) and len(es) == 2:
        es = [es]
    elif isinstance(es, list):
        pass
    else:
        return

    rewritten: list = []
    changed = False
    cat_features = tuple(sorted(fit_params.get("cat_features") or []))
    text_features = tuple(sorted(fit_params.get("text_features") or []))
    embedding_features = tuple(sorted(fit_params.get("embedding_features") or []))

    for entry in es:
        if isinstance(entry, _Pool):
            rewritten.append(entry)
            continue
        if not (isinstance(entry, tuple) and len(entry) == 2):
            rewritten.append(entry)
            continue
        val_df, val_target = entry
        if val_df is None or val_target is None:
            rewritten.append(entry)
            continue

        # Content-fingerprint via shared helper (2026-05-23): pre-fix
        # ``id(val_df)`` cache key broke across sklearn.clone() and
        # .iloc[...] slicing -- same id(X) bug as xgb_shim / lgb_shim /
        # CB train Pool. Consolidated.
        from .._dataset_cache_fingerprint import compute_signature
        key = compute_signature(
            val_df,
            extra=(cat_features, text_features, embedding_features),
        )

        cached = _CB_VAL_POOL_CACHE.get(key)
        if cached is not None:
            last_target_id = getattr(cached, "_mlframe_last_target_id", None)
            try:
                if last_target_id != id(val_target):
                    try:
                        _lab = _coerce_label_for_cb_pool(val_target)
                    except Exception:
                        _lab = val_target
                    cached.set_label(_lab)
                    cached._mlframe_last_target_id = id(val_target)
                logger.info(
                    "[cb-val-pool-reuse] hit key=(cols=%d,n=%d,cat=%d,text=%d,emb=%d) " "swapped%s without rebuild",
                    len(key[0]) if key[0] is not None else 0,
                    key[1] if key[1] is not None else 0,
                    len(cat_features),
                    len(text_features),
                    len(embedding_features),
                    " label" if last_target_id != id(val_target) else "",
                )
                rewritten.append(cached)
                changed = True
                continue
            except Exception as exc:
                logger.info(f"[cb-val-pool-reuse] swap failed ({type(exc).__name__}: " f"{str(exc).splitlines()[0][:120]}); rebuilding val Pool.")
                _CB_VAL_POOL_CACHE.pop(key, None)

        # Miss: build fresh val Pool with float32-cast label.
        while len(_CB_VAL_POOL_CACHE) >= _CB_POOL_CACHE_MAX_ENTRIES:
            _CB_VAL_POOL_CACHE.pop(next(iter(_CB_VAL_POOL_CACHE)))

        try:
            _lab_build = _coerce_label_for_cb_pool(val_target)
        except Exception:
            _lab_build = val_target

        try:
            val_pool = _Pool(
                data=val_df,
                label=_lab_build,
                cat_features=list(cat_features) or None,
                text_features=list(text_features) or None,
                embedding_features=list(embedding_features) or None,
            )
        except Exception as exc:
            logger.info(
                "[cb-val-pool-reuse] Pool build failed (%s: %s); " "leaving eval_set entry as (df, target) tuple for sklearn-wrapper rebuild.",
                type(exc).__name__,
                exc,
            )
            rewritten.append(entry)
            continue

        val_pool._mlframe_last_target_id = id(val_target)
        # Stash a content-fingerprint on the Pool so the predict-side
        # lookup in ``_predict_with_fallback`` can do a cols + shape +
        # dtypes content match when ``id(val_df)`` has shifted between
        # fit and metrics phases (2026-04-24 prod regression -- same
        # frame, different Python object due to upstream pre_pipeline
        # transforms).
        try:
            if hasattr(val_df, "dtypes"):
                val_pool._mlframe_dtypes_sig = tuple(str(d) for d in val_df.dtypes)
            elif hasattr(val_df, "schema"):
                val_pool._mlframe_dtypes_sig = tuple(str(d) for d in val_df.schema.values())
            else:
                val_pool._mlframe_dtypes_sig = None
        except Exception:
            val_pool._mlframe_dtypes_sig = None
        _CB_VAL_POOL_CACHE[key] = val_pool
        rewritten.append(val_pool)
        changed = True
        logger.info(
            "[cb-val-pool-reuse] miss; stored fresh val Pool (cache size=%d)",
            len(_CB_VAL_POOL_CACHE),
        )

    if changed:
        # Preserve original shape -- single-tuple or list.
        if len(rewritten) == 1 and not isinstance(es, list):
            fit_params["eval_set"] = rewritten[0]
        else:
            fit_params["eval_set"] = rewritten


# ----------------------------------------------------------------------
# Sibling-module re-export. The 289-LOC ``_maybe_get_or_build_cb_pool``
# body lives in ``_cb_pool_build.py`` so this file stays below the
# 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._cb_pool_build import _maybe_get_or_build_cb_pool  # noqa: E402,F401
