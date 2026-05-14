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

from .phases import phase
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple, Union

from mlframe.config import CATBOOST_MODEL_TYPES
from .phases import phase

from pyutilz.system import get_gpuinfo_gpu_info

import numpy as np

logger = logging.getLogger(__name__)

def _polars_schema_diagnostic(
    df: "pl.DataFrame",
    cat_features: Optional[List[str]] = None,
    text_features: Optional[List[str]] = None,
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
        lines: List[str] = []
        enum_cat_cols: List[str] = []  # the smoking-gun list

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
        other_dtype_counts: Dict[str, int] = {}
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
            text_dt_counts: Dict[str, int] = {}
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


def _polars_nullable_categorical_cols(df: Any, cat_features: Optional[List[str]] = None) -> "List[str]":
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


def _polars_df_has_null_in_categorical(df: Any, cat_features: Optional[List[str]] = None) -> bool:
    """Boolean wrapper around ``_polars_nullable_categorical_cols`` --
    kept for callers that only need the yes/no answer."""
    return bool(_polars_nullable_categorical_cols(df, cat_features=cat_features))


def _polars_fill_null_in_categorical(
    df: Any,
    nullable_cat_cols: "List[str]",
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


def _recover_cb_feature_names(model: Any) -> Tuple[List[str], List[str]]:
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
        cat_idx = getattr(model, "_get_cat_feature_indices", lambda: [])() or []
        text_idx = getattr(model, "_get_text_feature_indices", lambda: [])() or []
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
    from ._predict_guards import (
        _apply_nan_guard,
        _cb_polars_to_pandas,
        _cb_val_pool_cache_lookup,
        _ensure_lgbm_gets_pandas,
        _pl_DataFrame,
    )

    # ── 1. LGBM Polars → pandas auto-convert ──────────────────────────
    X = _ensure_lgbm_gets_pandas(model, X, method)

    # ── 2. CB val Pool cache lookup ────────────────────────────────────
    if _model_type in CATBOOST_MODEL_TYPES and hasattr(X, "columns") and hasattr(X, "shape"):
        try:
            _hit = _cb_val_pool_cache_lookup(X, method)
            if _hit is not None:
                logger.info(
                    "[cb-val-pool-reuse] %s hit on cached val Pool -- "
                    "skipping redundant Pool rebuild.", method,
                )
                with phase(method, model=_model_type, n_rows=n_rows):
                    return fn(_hit)
        except (AttributeError, KeyError, TypeError, ValueError) as _exc:
            logger.debug(
                "[cb-val-pool-reuse] %s cache probe failed (%s: %s); "
                "falling through.", method, type(_exc).__name__, _exc,
            )

    # ── 3. Sticky-pandas short-circuit (previous Polars miss) ──────────
    _pl_df = _pl_DataFrame()
    _is_cb = _model_type in CATBOOST_MODEL_TYPES
    if (
        _pl_df is not type(None)
        and isinstance(X, _pl_df)
        and _is_cb
        and getattr(model, "_mlframe_polars_fastpath_broken", False)
    ):
        X_pd = _cb_polars_to_pandas(model, X, method, verbose=verbose)
        with phase(method, model=_model_type, n_rows=n_rows):
            return fn(X_pd)

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
                return _apply_nan_guard(model, X, fn, n_rows)
        return result
    except ValueError as e:
        if "NaN" not in str(e) and "contains NaN" not in str(e):
            raise
        return _apply_nan_guard(model, X, fn, n_rows)
    except TypeError as e:
        if not (_is_cb and _pl_df is not type(None) and isinstance(X, _pl_df)
                and "No matching signature found" in str(e)):
            raise
        logger.warning(
            "CatBoost %s Polars fastpath rejected the data (%s); "
            "converting to pandas and retrying.",
            method, str(e).splitlines()[-1][:240],
        )
        try:
            model._mlframe_polars_fastpath_broken = True
        except AttributeError:
            pass
        X_pd = _cb_polars_to_pandas(model, X, method, verbose=verbose)
        with phase(method, model=_model_type, n_rows=n_rows):
            return fn(X_pd)



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
from ._predict_guards import _CB_VAL_POOL_CACHE  # noqa: E402,F401
from ._pipeline_helpers import (  # noqa: E402,F401
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

_CB_POOL_CACHE: "Dict[tuple, Any]" = {}
_CB_POOL_CACHE_MAX_ENTRIES = 16  # hard cap per cache; ring-buffer eviction oldest-first


# 2026-04-29: cache the GPU probe result. ``get_gpuinfo_gpu_info`` shells
# out to ``nvidia-smi`` via GPUtil, which costs ~0.5s on Windows per call.
# ``configure_training_params`` invokes the probe once per
# ``train_mlframe_models_suite`` call, so a long-lived process running
# many suites pays the subprocess startup repeatedly. GPU topology
# doesn't change during a process lifetime (no hot-plug under CUDA), so
# a one-shot cache is safe. Override with ``MLFRAME_NO_GPU_INFO_CACHE=1``
# if a future use case needs live re-probing.
_GPU_INFO_CACHE: "Optional[list]" = None


def _cached_gpu_info() -> list:
    """Memoised wrapper over ``pyutilz.system.get_gpuinfo_gpu_info``.

    First call runs the real probe (nvidia-smi subprocess); subsequent
    calls return the cached list. Saves ~0.5s on every
    ``configure_training_params`` invocation past the first.
    """
    global _GPU_INFO_CACHE
    if _GPU_INFO_CACHE is not None and not os.environ.get("MLFRAME_NO_GPU_INFO_CACHE"):
        return _GPU_INFO_CACHE
    result = get_gpuinfo_gpu_info()
    _GPU_INFO_CACHE = result
    return result


def _cb_reuse_capable() -> bool:
    """True iff installed CatBoost Pool exposes both set_label and
    set_weight (the two mutators we rely on for in-place label/weight
    swap between fits)."""
    try:
        from catboost import Pool as _Pool
    except ImportError:
        return False
    return callable(getattr(_Pool, "set_label", None)) and callable(getattr(_Pool, "set_weight", None))


def _maybe_get_or_build_cb_pool(
    model_type_name: str,
    model: Any,
    train_df: Any,
    train_target: Any,
    fit_params: Dict[str, Any],
) -> Optional[Any]:
    """Return a cached/freshly-built ``catboost.Pool`` when the CB reuse
    fast-path applies; return None otherwise (caller falls back to
    ``model.fit(train_df, y, **fit_params)``).

    Fast-path activation requires ALL of:
      * ``model_type_name in CATBOOST_MODEL_TYPES``
      * installed CatBoost has Pool.set_label/set_weight
      * train_df is a recognised input type (polars/pandas/numpy)

    Cache-hit: swap label + weight in place, return the cached Pool.
    Cache-miss: build a new Pool, store, return it.
    """
    if model_type_name not in CATBOOST_MODEL_TYPES:
        return None

    # Empty-target guard: if the caller passed a 0-length train_target
    # (e.g. RFECV inner CV fold collapsed after MRMR dropped all rows of
    # one class on rare-imbalance combos -- fuzz c0079), skip the
    # Pool-reuse fast-path and let CB raise a clearer error from the
    # sklearn wrapper. Using the cached Pool would silently set an empty
    # label and CB then crashes deep in ``_check_label_empty`` with no
    # context about which combo / fold triggered it.
    try:
        if train_target is not None and hasattr(train_target, "__len__") and len(train_target) == 0:
            logger.warning("[cb-pool-reuse] empty train_target -- skipping Pool reuse " "(would set zero-length label); deferring to sklearn fallback.")
            return None
    except Exception:
        pass

    # Filter cat/text/embedding features to only those actually present
    # in train_df. Motivation: MRMR and similar selectors can drop columns
    # AFTER fit_params was built, leaving stale feature lists that CB's
    # Pool rejects with ``ValueError: 'feat' is not in list`` from the
    # sklearn-wrapper's ``_get_cat_feature_indices`` (observed 2026-04-21
    # on ``test_mrmr_with_text_column`` / ``_embedding_column``). Applied
    # to CB only -- XGB/LGB have their own handling for missing cols.
    try:
        _df_cols = set(train_df.columns) if hasattr(train_df, "columns") else None
    except Exception:
        _df_cols = None

    def _filter_to_df(feats):
        raw = fit_params.get(feats) or []
        if _df_cols is None:
            return tuple(sorted(raw))
        return tuple(sorted(c for c in raw if c in _df_cols))

    cat_features = _filter_to_df("cat_features")
    text_features = _filter_to_df("text_features")
    embedding_features = _filter_to_df("embedding_features")
    # Update fit_params in place so the fallback sklearn path (when reuse
    # is disabled or Pool construction fails) also sees the filtered
    # lists. Callers may rely on the same fit_params dict downstream; we
    # only narrow, never widen.
    if _df_cols is not None:
        if "cat_features" in fit_params and fit_params["cat_features"]:
            fit_params["cat_features"] = list(cat_features)
        if "text_features" in fit_params and fit_params["text_features"]:
            fit_params["text_features"] = list(text_features)
        if "embedding_features" in fit_params and fit_params["embedding_features"]:
            fit_params["embedding_features"] = list(embedding_features)

    if not _cb_reuse_capable():
        return None
    try:
        from catboost import Pool as _Pool
    except ImportError:
        return None

    sample_weight = fit_params.get("sample_weight")

    # Cache key: id(df) alone is unsafe because Python reuses ids after
    # GC. Two tests in the same process that each build a fresh frame
    # may land on the same ``id(train_df)`` value -- hitting a cache entry
    # built for a DIFFERENT frame with DIFFERENT cat_features/columns.
    # Include a content signature (columns tuple) so collisions with
    # distinct data produce a miss instead of a corrupted reuse.
    try:
        _cols = tuple(train_df.columns) if hasattr(train_df, "columns") else None
    except Exception:
        _cols = None
    try:
        _shape = getattr(train_df, "shape", None)
        _shape_sig = (int(_shape[0]), int(_shape[1])) if _shape and len(_shape) >= 2 else None
    except Exception:
        _shape_sig = None
    key = (id(train_df), _cols, _shape_sig, cat_features, text_features, embedding_features)

    # Verify train_target length matches train_df row count BEFORE the
    # Pool-reuse fast-path. RFECV's inner CV folds occasionally hand us
    # train_target / train_df pairs whose lengths disagree (subset of
    # rows but full target, or vice versa); the Pool then ends up with a
    # stale label and CB.fit raises "Labels variable is empty" deep in
    # C++ Pool init (fuzz c0079). Skip Pool reuse on mismatch and let
    # the sklearn fallback path build a fresh Pool with the current
    # (data, label) pair.
    _df_rows = train_df.shape[0] if hasattr(train_df, "shape") else None
    _tg_len = len(train_target) if train_target is not None and hasattr(train_target, "__len__") else None
    if _df_rows is not None and _tg_len is not None and _df_rows != _tg_len:
        # Hard contract violation -- raised 2026-04-28 (batch 4, was
        # logger.error+fallback). X/y length mismatch reaching this
        # point means an upstream slicing bug (fuzz c0079-style: RFECV
        # inner CV producing inconsistent train_target / train_df
        # lengths). Fall back to sklearn would just delay the same
        # error with less context; raise here gives a stack trace
        # rooted in mlframe's flow, not deep in CB's C++ ``Labels
        # variable is empty`` (which is misleading -- it's about
        # length, not emptiness).
        raise RuntimeError(
            f"[cb-pool-reuse] train_df rows ({_df_rows}) != "
            f"train_target len ({_tg_len}). This is a hard contract "
            f"violation; investigate upstream slicing (RFECV inner CV / "
            f"OD filter / aging trim) that produced the mismatch."
        )

    cached = _CB_POOL_CACHE.get(key)
    if cached is not None:
        # Installed CatBoost 1.2.10 rejects ``Pool.set_label`` on a
        # classification Pool (target type ``Integer``) -- the C++
        # ``SetNumericTarget`` path only accepts numeric / unset targets.
        # That means we can only reuse across WEIGHT swaps, not label
        # swaps, for classification pools. Strategy: skip ``set_label``
        # unless the caller actually supplied a different target (by id
        # against the last target we stored). Always mutate weight --
        # ``set_weight`` has no target-type restriction.
        last_target_id = getattr(cached, "_mlframe_last_target_id", None)
        try:
            if last_target_id is None or id(train_target) != last_target_id:
                # Label swap. Cast to float32 -- the Pool was built with a
                # float32 label (see build path below), and CB's C++
                # ``SetNumericTarget`` rejects anything but Float/None. If
                # rejection happens anyway, fall through to rebuild.
                try:
                    _label_for_swap = np.asarray(train_target)
                    if _label_for_swap.dtype != np.float32:
                        _label_for_swap = _label_for_swap.astype(np.float32)
                except Exception:
                    _label_for_swap = train_target
                cached.set_label(_label_for_swap)
                cached._mlframe_last_target_id = id(train_target)
            if sample_weight is not None:
                cached.set_weight(sample_weight)
            # Post-swap verification: confirm the cached Pool's label is
            # non-empty (set_label can silently set a 0-length array if
            # _label_for_swap was empty after some upstream filter, then
            # CB.fit raises "Labels variable is empty" deep in _check_-
            # label_empty with no diagnostics, fuzz c0079). Evict and
            # rebuild on miss.
            try:
                _post_label = cached.get_label()
                if _post_label is not None and hasattr(_post_label, "__len__") and len(_post_label) == 0:
                    logger.info("[cb-pool-reuse] cached Pool ended up with empty label after swap " "-- evicting and rebuilding.")
                    _CB_POOL_CACHE.pop(key, None)
                    raise RuntimeError("empty cached label after set_label")
            except Exception as _verify_exc:
                if "empty cached label" in str(_verify_exc):
                    raise
                # get_label() not exposed on this CB build; trust set_label.
                pass
            logger.info(
                "[cb-pool-reuse] hit key=(id=%s,cat=%d,text=%d,emb=%d) " "swapped weight%s without rebuild",
                key[0],
                len(cat_features),
                len(text_features),
                len(embedding_features),
                " + label" if last_target_id != id(train_target) else "",
            )
            return cached
        except Exception as exc:
            # Drop the stale entry and fall through to rebuild. Typical
            # trigger: classification Pool + set_label on Integer target
            # raises "SetNumericTarget requires numeric or unset target
            # type". Rebuild is safe.
            logger.info(
                "[cb-pool-reuse] swap path not usable (%s: %s); rebuilding Pool.",
                type(exc).__name__,
                str(exc).splitlines()[0][:120],
            )
            _CB_POOL_CACHE.pop(key, None)

    # Simple FIFO eviction -- unlikely to hit during normal runs (<= N
    # models x N tiers entries), but keeps the cache from growing
    # unboundedly across long-running sessions.
    while len(_CB_POOL_CACHE) >= _CB_POOL_CACHE_MAX_ENTRIES:
        _CB_POOL_CACHE.pop(next(iter(_CB_POOL_CACHE)))

    # Cast label to float32 at build time. CatBoost stores the label's
    # raw type on the Pool (Integer vs Float) and later ``Pool.set_label``
    # validates ``ERawTargetType == Float or None`` inside C++
    # ``SetNumericTarget`` -- if we built with Integer labels, subsequent
    # label swaps across classification targets would raise
    # ``SetNumericTarget requires numeric or unset target type, got
    # Integer``. Building with float32 pins the Pool's target type to
    # Float upfront; the user's upstream PR's classification tests all
    # pre-cast to float32 for exactly this reason. get_label() still
    # round-trips integer dtype via the Python-level ``target_type``
    # shadow on the Pool.
    try:
        _label_for_pool = _coerce_label_for_cb_pool(train_target)
    except Exception:
        _label_for_pool = train_target

    try:
        pool = _Pool(
            data=train_df,
            label=_label_for_pool,
            weight=sample_weight,
            cat_features=list(cat_features) or None,
            text_features=list(text_features) or None,
            embedding_features=list(embedding_features) or None,
        )
    except Exception as exc:
        # If Pool rejects the input (e.g. unsupported dtype combo),
        # fall back to the sklearn-wrapper path by returning None. The
        # operator sees the build-logger line above; we don't cache a
        # failed attempt.
        logger.warning(f"[cb-pool-reuse] Pool construction failed ({type(exc).__name__}: {exc}); " f"falling back to rebuild-every-fit sklearn path.")
        return None

    pool._mlframe_last_target_id = id(train_target)
    # Cache feature lists on the Pool so callers (notably the dynamic CB
    # ``text_processing`` injection in ``_train_model_with_fallback``)
    # can introspect them without round-tripping through fit_params,
    # which the Pool-reuse path strips before fit.
    pool._mlframe_text_features = list(text_features)
    pool._mlframe_cat_features = list(cat_features)
    pool._mlframe_embedding_features = list(embedding_features)
    _CB_POOL_CACHE[key] = pool
    logger.info(
        "[cb-pool-reuse] miss; stored fresh Pool (cache size=%d)",
        len(_CB_POOL_CACHE),
    )
    return pool


def _maybe_rewrite_eval_set_as_cb_pool(fit_params: Dict[str, Any]) -> None:
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

        try:
            _cols = tuple(val_df.columns) if hasattr(val_df, "columns") else None
        except Exception:
            _cols = None
        try:
            _shape = getattr(val_df, "shape", None)
            _shape_sig = (int(_shape[0]), int(_shape[1])) if _shape and len(_shape) >= 2 else None
        except Exception:
            _shape_sig = None
        key = (id(val_df), _cols, _shape_sig, cat_features, text_features, embedding_features)

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
                    "[cb-val-pool-reuse] hit key=(id=%s,cat=%d,text=%d,emb=%d) " "swapped%s without rebuild",
                    key[0],
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


