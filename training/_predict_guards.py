"""Prediction-time fallback guards for CatBoost, LGBM, and NaN-intolerant models.

Extracted from ``trainer._predict_with_fallback`` (2026-05-13 refactor).
Each guard handles one predict-time edge case so the main path stays a
single ``fn(X)`` call. The guards are ordered by cost (cheapest first).

Motivation for each guard (condensed from production incident reports):

1. **LGBM Polars auto-convert** — 2026-04-19: LGB's sklearn wrapper
   ``_LGBMValidateData`` converts ``pd.Categorical`` columns to numpy
   object arrays of strings on Polars input, then crashes on the first
   non-numeric cell. Convert upfront.

2. **CB val Pool cache** — 2026-04-22: CB's sklearn wrapper rebuilds a
   fresh Pool from the DataFrame on every predict call. On 7.3M rows,
   this cost 53-66 s *per metrics phase* (VAL + TEST + ensembles). The
   fit path already built a Pool; we cache it and reuse at predict time
   via a two-stage lookup (id match → content-fingerprint fallback).

3. **CB sticky-pandas** — 2026-04-24: once a CatBoost model instance
   has failed a Polars predict call (``TypeError: No matching signature
   found``), every subsequent call re-hits the same Cython dispatch miss
   → 1-2 s wasted per call. Set ``_mlframe_polars_fastpath_broken=True``
   on the model so later calls skip the retry dance.

4. **NaN safety net** — 2026-05-13: when the strategy pre_pipeline
   (SimpleImputer+StandardScaler) is skipped for test_df (cache-hit
   path), raw NaN reaches NaN-intolerant models (LinearRegression,
   Ridge). One-shot impute+scale is a safety net; the root cause is
   in ``_prepare_test_split`` + ``_build_process_model_kwargs``.

5. **CB Polars dispatch-miss fallback** — 2026-04-19: CB 1.2.x's
   Polars fastpath rejects certain nullable-Categorical / Enum dtypes
   with ``TypeError: No matching signature found``. Fall back to
   pandas + ``prepare_df_for_catboost``. This is where the
   sticky-pandas flag is first set.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from mlframe.config import CATBOOST_MODEL_TYPES

logger = logging.getLogger(__name__)


# ── Lazy Polars reference (avoids import if not installed) ────────────────


def _pl_DataFrame():
    """Return ``pl.DataFrame`` or ``type(None)`` if Polars not installed.

    Returns ``type(None)`` (not ``None``) so that ``isinstance(X, _pl_df)``
    is always safe: when Polars is absent, ``isinstance(X, NoneType)`` only
    matches ``None`` itself without raising ``TypeError``.
    """
    try:
        import polars as pl
        return pl.DataFrame
    except ImportError:
        return type(None)


# ═══════════════════════════════════════════════════════════════════════════
# 1. LGBM Polars → pandas auto-convert
# ═══════════════════════════════════════════════════════════════════════════


def _ensure_lgbm_gets_pandas(model: Any, X: Any, method: str) -> Any:
    """LGBM's sklearn-wrapper crashes on Polars input: it converts
    ``pd.Categorical`` columns to numpy object arrays → non-numeric
    cells trigger ValueError in ``_LGBMValidateData``.  Convert to
    pandas upfront so LGB takes its native fastpath.  (2026-04-19 prod.)"""
    if not (isinstance(X, _pl_DataFrame()) and "LGBM" in type(model).__name__):
        return X
    from .utils import get_pandas_view_of_polars_df
    logger.warning(
        "  [predict] %s.%s received pl.DataFrame; "
        "converting to pandas for LGB's sklearn-native fastpath.",
        type(model).__name__, method,
    )
    return get_pandas_view_of_polars_df(X)


# ═══════════════════════════════════════════════════════════════════════════
# 2. CB val Pool cache (shared with trainer._maybe_get_or_build_cb_pool)
# ═══════════════════════════════════════════════════════════════════════════

_CB_VAL_POOL_CACHE: Dict[tuple, Any] = {}
"""Module-level cache: (id, cols, shape) → catboost.Pool for val frames.

Populated at fit time by ``trainer._maybe_get_or_build_cb_pool``;
read at predict time by ``_predict_with_fallback``.  Same dict object
is imported by trainer.py so both paths share state.
"""


def _cb_val_pool_cache_lookup(X: Any, method: str) -> Optional[Any]:
    """Two-stage CB val Pool cache lookup.

    **Why**: CB's sklearn wrapper short-circuits rebuild on
    ``isinstance(X, Pool)``. Passing the cached Pool skips a 50-70s
    rebuild on 7M-row frames. Two-stage lookup because the Python
    object identity (``id(X)``) can change between fit and metrics
    phases (pre_pipeline transforms return fresh DataFrames).

    Stage 1 — exact ``id(X)`` match (fast, common case).
    Stage 2 — content fallback on cols + shape + dtypes (safe for
    predict-only reuse: the Pool's label isn't read at predict time).
    (2026-04-22 prod, hardened 2026-04-24.)
    """
    try:
        _cols = tuple(X.columns) if hasattr(X, "columns") else None
    except Exception:
        return None
    try:
        _shape = X.shape
        _shape_sig = (int(_shape[0]), int(_shape[1]))
    except Exception:
        _shape_sig = None
    try:
        if hasattr(X, "dtypes"):
            _dtypes_sig = tuple(str(d) for d in X.dtypes)
        elif hasattr(X, "schema"):
            _dtypes_sig = tuple(str(d) for d in X.schema.values())
        else:
            _dtypes_sig = None
    except Exception:
        _dtypes_sig = None

    _id = id(X)
    # Stage 1: id match
    for key, pool in _CB_VAL_POOL_CACHE.items():
        if key[0] == _id and key[1] == _cols and key[2] == _shape_sig:
            return pool
    # Stage 2: content fallback
    if _shape_sig is not None and _dtypes_sig is not None:
        for key, pool in _CB_VAL_POOL_CACHE.items():
            cached_dtypes = getattr(pool, "_mlframe_dtypes_sig", None)
            if key[1] == _cols and key[2] == _shape_sig and cached_dtypes == _dtypes_sig:
                return pool
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Shared: recover CB feature names from fitted model
# ═══════════════════════════════════════════════════════════════════════════


def _recover_cb_feature_names(model: Any) -> Tuple[List[str], List[str]]:
    """Extract (cat_features, text_features) from a fitted CatBoost model.

    At predict time the original Python-side lists aren't available;
    CatBoost exposes them via ``_get_cat_feature_indices`` /
    ``_get_text_feature_indices`` + ``feature_names_``.

    Returns ``([], [])`` on any failure — callers degrade gracefully
    (missing names → less specific prep path, not a crash).
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


# ═══════════════════════════════════════════════════════════════════════════
# Shared: CB Polars → pandas conversion (used by sticky-pandas + fallback)
# ═══════════════════════════════════════════════════════════════════════════


def _cb_polars_to_pandas(
    model: Any, X: Any, method: str, *, verbose: bool = False,
) -> Any:
    """Convert Polars DataFrame to pandas for CatBoost predict.

    Applies text-feature decategorisation + ``prepare_df_for_catboost``.
    Shared between the sticky-pandas short-circuit and the TypeError
    fallback (both do the same conversion, just triggered differently).
    """
    from .utils import get_pandas_view_of_polars_df
    from .preprocessing import prepare_df_for_catboost as _prep_cb

    cat_feat, text_feat = _recover_cb_feature_names(model)
    if verbose or not (cat_feat or text_feat):
        logger.info(
            "  [predict fallback] recovered from model: cat=%d, text=%d",
            len(cat_feat), len(text_feat),
        )

    from timeit import default_timer as timer
    t0 = timer()
    shape_str = f"{X.shape[0]:_}x{X.shape[1]}" if hasattr(X, "shape") else "?"
    X_pd = get_pandas_view_of_polars_df(X)
    logger.info("  [predict fallback] polars->pandas(%s) %s in %.1fs",
                method, shape_str, timer() - t0)

    # Decategorise text columns before prep_cb (same ordering as fit path)
    if text_feat:
        import pandas as pd
        for col in text_feat:
            if col in X_pd.columns and isinstance(X_pd[col].dtype, pd.CategoricalDtype):
                X_pd[col] = X_pd[col].astype("object").fillna("")

    t0 = timer()
    X_pd = _prep_cb(X_pd, cat_features=list(cat_feat), text_features=list(text_feat))
    logger.info("  [predict fallback] prepare_df_for_catboost(%s) in %.1fs",
                method, timer() - t0)
    return X_pd


# ═══════════════════════════════════════════════════════════════════════════
# 4. NaN safety net for intolerant models
# ═══════════════════════════════════════════════════════════════════════════


def _apply_nan_guard(
    model: Any, X: Any, fn: Callable, n_rows: Optional[int],
) -> np.ndarray:
    """One-shot impute + scale when X contains NaN and the model is
    NaN-intolerant (raw LinearRegression / Ridge without a Pipeline).

    **Why**: when ``skip_pre_pipeline_transform`` is True (tier-cache
    hit), the strategy pre_pipeline (SimpleImputer + StandardScaler)
    is skipped for test_df. Tree models handle NaN natively; linear
    models crash with ``ValueError: Input X contains NaN``.

    Checks a 500-row sample for NaN before invoking expensive full
    imputation. Applies BOTH imputation AND scaling because the model
    was trained on scaled data (2026-05-13 prod: one-shot SimpleImputer
    without scaling produced RMSE=539M vs expected 12).
    """
    # Fast NaN check: sample first 500 rows
    _has_nan = False
    try:
        if hasattr(X, "isna"):
            _sample = X.iloc[:500] if hasattr(X, "iloc") else X[:500]
            _has_nan = bool(_sample.isna().any().any())
        elif hasattr(X, "__array__"):
            _arr_check = np.asarray(X[:500]) if hasattr(X, "__getitem__") else np.asarray(X)
            _has_nan = bool(np.any(~np.isfinite(_arr_check[:500])))
    except Exception:
        _has_nan = False

    if not _has_nan:
        return fn(X)  # Let the real error surface

    logger.warning(
        "[NaN-guard] %s X contains NaN; applying one-shot "
        "SimpleImputer + StandardScaler before predict (n_rows=%s).  "
        "The strategy pre_pipeline was skipped (cache-hit path); "
        "this is a safety net, not a fix for the root cause.",
        type(model).__name__, n_rows,
    )

    from sklearn.impute import SimpleImputer as _SI
    from sklearn.preprocessing import StandardScaler as _SS

    # Convert to numpy, impute NaN with mean, standardise
    if hasattr(X, "to_numpy"):
        _arr = X.to_numpy(dtype=np.float64)
    elif hasattr(X, "values"):
        _arr = np.asarray(X.values, dtype=np.float64)
    else:
        _arr = np.asarray(X, dtype=np.float64)

    _arr = _SS().fit_transform(_SI(strategy="mean").fit_transform(_arr))

    # Re-wrap as the original container type
    if hasattr(X, "columns"):
        import pandas as pd
        X_clean: Any = pd.DataFrame(
            _arr, columns=list(X.columns),
            index=getattr(X, "index", None),
        )
    else:
        X_clean = _arr
    return fn(X_clean)
