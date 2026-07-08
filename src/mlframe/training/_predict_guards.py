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
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class NanGuardNotPrimedError(RuntimeError):
    """Raised when ``_apply_nan_guard`` encounters NaN at predict time
    on a model whose imputer/scaler statistics were never primed at
    fit time. Per audit 2026-05-17 (C10) the legacy fall-through that
    fit imputer+scaler on the current frame was a silent leakage
    vector: test-set statistics ended up persisted on the model and
    applied to every subsequent call. Callers that want that legacy
    behaviour must opt in explicitly via ``fit_at_predict=True``.
    """


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
        "  [predict] %s.%s received pl.DataFrame; " "converting to pandas for LGB's sklearn-native fastpath.",
        type(model).__name__,
        method,
    )
    return get_pandas_view_of_polars_df(X)


# ═══════════════════════════════════════════════════════════════════════════
# 2. CB val Pool cache (shared with trainer._maybe_get_or_build_cb_pool)
# ═══════════════════════════════════════════════════════════════════════════

_CB_VAL_POOL_CACHE: dict[tuple, Any] = {}
"""Module-level cache: (id, cols, shape) → catboost.Pool for val frames.

Populated at fit time by ``trainer._maybe_get_or_build_cb_pool``;
read at predict time by ``_predict_with_fallback``.  Same dict object
is imported by trainer.py so both paths share state.
"""


def _cb_val_pool_cache_lookup(X: Any, method: str) -> Any | None:
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
    # Stage 1: id match, CO-VALIDATED by the content key (cols + shape + dtypes). The id() alone can recycle
    # onto an unrelated frame after GC; requiring cols/shape/dtypes to also match makes an id false-hit
    # return a pool that is genuinely compatible (or fall through to the stage-2 content match / miss).
    for key, pool in _CB_VAL_POOL_CACHE.items():
        if key[0] == _id and key[1] == _cols and key[2] == _shape_sig:
            if _dtypes_sig is not None:
                cached_dtypes = getattr(pool, "_mlframe_dtypes_sig", None)
                if cached_dtypes is not None and cached_dtypes != _dtypes_sig:
                    continue
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


def _recover_cb_feature_names(model: Any) -> tuple[list[str], list[str]]:
    """Extract (cat_features, text_features) from a fitted CatBoost model.

    At predict time the original Python-side lists aren't available;
    CatBoost exposes them via ``_get_cat_feature_indices`` /
    ``_get_text_feature_indices`` + ``feature_names_``.

    Returns ``([], [])`` on any failure — callers degrade gracefully
    (missing names → less specific prep path, not a crash).
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
    from .pipeline import prepare_df_for_catboost as _prep_cb

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
    logger.info("  [predict fallback] polars->pandas(%s) %s in %.1fs", method, shape_str, timer() - t0)

    # Decategorise text columns before prep_cb (same ordering as fit path)
    if text_feat:
        import pandas as pd
        for col in text_feat:
            if col in X_pd.columns and isinstance(X_pd[col].dtype, pd.CategoricalDtype):
                X_pd[col] = X_pd[col].astype("object").fillna("")

    t0 = timer()
    _prep_cb(X_pd, cat_features=list(cat_feat))  # in-place; text_feat already decategorised above
    logger.info("  [predict fallback] prepare_df_for_catboost(%s) in %.1fs", method, timer() - t0)
    return X_pd


# ═══════════════════════════════════════════════════════════════════════════
# 4. NaN safety net for intolerant models
# ═══════════════════════════════════════════════════════════════════════════


def _apply_nan_guard(
    model: Any, X: Any, fn: Callable, n_rows: int | None,
    *, fit_at_predict: bool = False,
) -> np.ndarray:
    """Impute + scale when X contains NaN and the model is NaN-intolerant
    (raw LinearRegression / Ridge without a Pipeline).

    **Leakage-safe contract**: by default this transforms via the
    imputer + scaler persisted on the model object during the FIRST
    invocation (``model._mlframe_nan_imputer`` /
    ``model._mlframe_nan_scaler``). The persisted instances were fit on
    the very first frame this guard saw -- in practice that frame is
    the training tail handed in by ``_train_model_with_fallback`` /
    ``_predict_with_fallback`` AT FIT-TIME, before any predict-time
    frame ever reaches the guard.

    When the persisted attributes are absent, the guard takes one of
    two paths controlled by ``fit_at_predict``:

      * ``False`` (default, predict-time path): fit a fresh imputer +
        scaler on the *current* frame, persist them on the model, then
        transform. This is what the original implementation always
        did. A loud WARNING surfaces that this is a leakage-prone
        first-touch path. Subsequent calls will reuse the persisted
        statistics. The first-call leakage matters only when the FIRST
        frame ever passed through the guard is the predict frame --
        production training calls ``_train_model_with_fallback`` which
        primes the cache.
      * ``True`` (legacy compatibility): identical behaviour but
        without the warning. Used by callers that explicitly opt in to
        fit-on-predict semantics (currently none in production).

    The ``fit_at_predict=False`` warning is the regression sentinel
    that catches mis-wired predict paths. Production training MUST
    prime ``_mlframe_nan_imputer`` / ``_mlframe_nan_scaler`` at fit
    time so this WARN never fires at predict.

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
            # Wave 50 (2026-05-20): use np.isnan (not ~np.isfinite) for parity with
            # the pandas isna() branch above; the prior ~isfinite included +-inf
            # which SimpleImputer(strategy="mean", keep_empty_features=True) does NOT replace, so +-inf rows
            # would pass the gate but then propagate unchanged through imputer+scaler.
            _has_nan = bool(np.any(np.isnan(_arr_check[:500])))
    except Exception:
        _has_nan = False

    if not _has_nan:
        return np.asarray(fn(X))  # Let the real error surface

    # Persisted imputer/scaler shortcut -- transform-only, no leakage.
    _persisted_imp = getattr(model, "_mlframe_nan_imputer", None)
    _persisted_scl = getattr(model, "_mlframe_nan_scaler", None)
    if _persisted_imp is not None and _persisted_scl is not None:
        return _transform_with_persisted_stats(
            model, X, fn, n_rows, _persisted_imp, _persisted_scl,
        )

    # No persisted stats. Per audit 2026-05-17 (C10) the legacy
    # warn-and-fit-on-predict behaviour was a silent leakage path
    # whenever ``prime_nan_guard_stats`` wasn't called at fit time. We
    # now REFUSE by default: callers that want the legacy semantics
    # must opt in via ``fit_at_predict=True``.
    if not fit_at_predict:
        raise NanGuardNotPrimedError(
            f"[NaN-guard] {type(model).__name__} X contains NaN AND no "
            f"persisted _mlframe_nan_imputer / _mlframe_nan_scaler on "
            f"model (n_rows={n_rows}). Refusing to fit imputer/scaler "
            "on the current frame -- when the current frame is the "
            "predict frame, test-set statistics would leak into the "
            "model state. Call mlframe.training._predict_guards."
            "prime_nan_guard_stats(model, X_train) at fit time, or "
            "pass fit_at_predict=True to opt in to the legacy "
            "fit-on-current-frame behaviour."
        )

    logger.warning(
        "[NaN-guard] %s fit_at_predict=True explicitly requested; "
        "fitting imputer/scaler on current frame and persisting "
        "for future calls (n_rows=%s). Statistics derived from the "
        "current frame will be applied to all future calls -- ensure "
        "this is the training tail, not predict data.",
        type(model).__name__, n_rows,
    )

    return _fit_persist_and_transform(model, X, fn, n_rows)


def _transform_with_persisted_stats(
    model: Any, X: Any, fn: Callable, n_rows: int | None,
    imputer: Any, scaler: Any,
) -> np.ndarray:
    """Pure-transform path: apply ``imputer`` + ``scaler`` to X using
    the statistics already fit at training time. No fresh fit means no
    leakage.

    Polars-native short-circuit is intentionally NOT used here: the
    persisted sklearn estimators carry their own fitted statistics
    (``imputer.statistics_``, ``scaler.mean_`` / ``scale_``) that are
    the authoritative reference. We materialise the frame through the
    Arrow split-blocks bridge to numpy, then call sklearn's
    ``.transform()`` so the output matches train-time element-for-
    element.
    """
    if hasattr(X, "to_numpy"):
        _arr = X.to_numpy(dtype=np.float64) if not isinstance(X, _pl_DataFrame()) else None
        if _arr is None:
            # polars: bridge to numpy via Arrow. Try ``allow_copy=False`` first for the uniform-float64 case so the underlying Arrow buffer
            # is reused directly; fall back to the copy path on mixed / non-float64 dtypes.
            try:
                import polars as pl
                if isinstance(X, pl.DataFrame):
                    try:
                        _arr = X.to_numpy(allow_copy=False)
                        if _arr.dtype != np.float64:
                            _arr = _arr.astype(np.float64, copy=False)
                    except (TypeError, RuntimeError):
                        _arr = X.to_numpy()
                        if _arr.dtype != np.float64:
                            _arr = _arr.astype(np.float64, copy=False)
                else:
                    _arr = X.to_numpy(dtype=np.float64)
            except ImportError:
                _arr = np.asarray(X, dtype=np.float64)
    else:
        _arr = np.asarray(X, dtype=np.float64)

    # sklearn transform = subtract mean_ + divide by scale_ (no leakage --
    # mean_/scale_ are TRAIN statistics already fit and stored on the scaler).
    _arr = scaler.transform(imputer.transform(_arr))

    if hasattr(X, "columns"):
        import pandas as pd
        X_clean: Any = pd.DataFrame(
            _arr, columns=list(X.columns),
            index=getattr(X, "index", None),
        )
    else:
        X_clean = _arr
    return np.asarray(fn(X_clean))


def _fit_persist_and_transform(
    model: Any, X: Any, fn: Callable, n_rows: int | None,
) -> np.ndarray:
    """Fit imputer + scaler on X, persist on ``model``, then transform.

    Only invoked when the model has no pre-fitted persisted stats. The
    persistence side-effect means the SECOND call -- even at predict --
    reuses the train statistics and avoids leakage. The FIRST call is
    intrinsically leak-prone unless the caller hands in a training-tail
    frame; the warning above flags that case.

    Polars fast-path: when X is a fully-numeric polars DataFrame we
    impute + standardise entirely inside polars (fill_nan with column
    mean, then (x - mean) / std), then bridge the result to pandas via
    the Arrow split-blocks helper. Equivalent statistics to
    SimpleImputer(strategy='mean') + StandardScaler bit-for-bit
    (polars .mean()/.std(ddof=0) on NaN-converted-to-null produces the
    same column stats as numpy nanmean/nanstd which sklearn uses
    internally). Saves the polars->numpy->sklearn->pandas round-trip
    that bottlenecked predict_nan_guard on large polars inputs (~5x
    speedup measured on 100k rows). We STILL build the sklearn
    Imputer/Scaler objects post-hoc from the polars-computed stats so
    the persist step provides a transform-only fastpath for next call.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    try:
        import polars as pl
    except ImportError:
        pl = None  # type: ignore[assignment]

    # Cupy GPU acceleration was evaluated for this fastpath (D-Arch-6 audit, 2026-05-19);
    # bench at ``_benchmarks/bench_arch_d.bench_predict_guards_cupy`` measured 0.66x-0.79x
    # (cupy SLOWER than polars) on both n_rows=500k / n_cols=50 (25M cells) and n_rows=500k /
    # n_cols=200 (100M cells). The polars fastpath below is already a vectorised single-pass
    # plan; the H2D + D2H copy dominates the modest per-column arithmetic the GPU can offer.
    # Skipping per the >5% gate: not worth adding a cupy code path that would be slower at
    # every measured shape.
    if pl is not None and isinstance(X, pl.DataFrame) and all(dt.is_numeric() for dt in X.dtypes):
        from .utils import get_pandas_view_of_polars_df
        cols = X.columns
        # First pass: impute (fill_nan with drop_nans mean). Imputed mean == drop_nans mean so the imputer
        # statistics are computed on raw input. SECOND pass: compute std on the IMPUTED frame (not the raw)
        # so the result matches sklearn StandardScaler(SimpleImputer.fit_transform(X)) bit-for-bit. The
        # post-imputation std is strictly LESS than the drop_nans std (mean-filling reduces variance), and
        # sklearn uses the post-imputation values.
        # Fill BOTH null and NaN: a polars float column may carry nulls (not
        # just NaN). ``fill_nan`` alone leaves nulls untouched, so they would
        # survive to the bridged pandas frame and crash the NaN-intolerant
        # model the guard exists to protect. ``drop_nans().mean()`` already
        # ignores both null and NaN, matching sklearn SimpleImputer's nanmean.
        df_imp = X.with_columns([pl.col(c).fill_nan(pl.col(c).drop_nans().mean()).fill_null(pl.col(c).drop_nans().mean()) for c in cols])
        # All-null / all-NaN column: ``drop_nans().mean()`` is null, so the fill
        # above leaves null/NaN in place. sklearn SimpleImputer(keep_empty_features=
        # True) fills such a column with 0.0; mirror that so no null/NaN reaches
        # the model and the persisted imputer statistics stay 0 for empty columns.
        df_imp = df_imp.with_columns([pl.col(c).fill_nan(0.0).fill_null(0.0) for c in cols])
        # Compute post-imputation per-column means and stds in one pass for
        # persistence. We then re-use these scalars - via pl.lit constants -
        # to do the (x - mean) / std rewrite. The previous implementation
        # called ``pl.col(c).std(ddof=0)`` twice in the standardize step AND
        # ``pl.col(c).mean()`` again, so each column triggered three extra
        # full-column aggregations on top of the persistence scan. Broadcasting
        # pre-computed scalars cuts the aggregations from 4N to N per fit.
        _stats = df_imp.select([pl.col(c).mean().alias(f"_mean_{c}") for c in cols] + [pl.col(c).std(ddof=0).alias(f"_std_{c}") for c in cols]).row(0)
        _means_post = np.asarray(_stats[: len(cols)], dtype=np.float64)
        _stds_post = np.asarray(_stats[len(cols) :], dtype=np.float64)
        _stds_safe = np.where(_stds_post == 0.0, 1.0, _stds_post)
        df_std = df_imp.with_columns([(pl.col(c) - pl.lit(float(_means_post[_i]))) / pl.lit(float(_stds_safe[_i])) for _i, c in enumerate(cols)])
        # Persist as sklearn-compatible objects so the NEXT call hits the transform-only fastpath.
        # The imputer's statistics_ holds the imputation mean (== drop_nans mean == post-fill mean).
        try:
            imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
            imputer.statistics_ = _means_post.copy()
            imputer.n_features_in_ = len(cols)
            imputer.feature_names_in_ = np.array(list(cols), dtype=object)
            scaler = StandardScaler()
            scaler.mean_ = _means_post.copy()
            scaler.scale_ = _stds_safe.copy()
            scaler.var_ = _stds_post.copy() ** 2
            scaler.n_features_in_ = len(cols)
            scaler.feature_names_in_ = np.array(list(cols), dtype=object)
            scaler.n_samples_seen_ = int(len(X))
            model._mlframe_nan_imputer = imputer
            model._mlframe_nan_scaler = scaler
        except (AttributeError, TypeError):
            pass
        # Audit D P1-5 (2026-05-18): this is a 1-hop savings, not zero-copy end-to-end. The
        # polars-side mean/std + standardisation eliminate the polars->numpy->sklearn->pandas
        # round trip on the COMPUTE side, but ``fn`` downstream is a sklearn pipeline that takes
        # pandas and internally goes pandas->numpy. ``get_pandas_view_of_polars_df`` uses the
        # split-blocks Arrow bridge so numeric columns stay zero-copy at THIS hop; ``fn``'s
        # internal pandas->numpy is the remaining unavoidable copy until the sklearn estimator
        # accepts numpy/polars directly.
        X_clean: Any = get_pandas_view_of_polars_df(df_std)
        return np.asarray(fn(X_clean))

    if hasattr(X, "to_numpy"):
        # Polars: try the zero-copy Arrow bridge first; only fall back to the copy path on mixed-dtype frames where allow_copy=False raises.
        _is_polars = pl is not None and isinstance(X, pl.DataFrame)
        if _is_polars:
            try:
                _arr = X.to_numpy(allow_copy=False)
                if _arr.dtype != np.float64:
                    _arr = _arr.astype(np.float64, copy=False)
            except (TypeError, RuntimeError):
                _arr = X.to_numpy()
                if _arr.dtype != np.float64:
                    _arr = _arr.astype(np.float64, copy=False)
        else:
            try:
                _arr = X.to_numpy(dtype=np.float64)
            except TypeError:
                _arr = X.to_numpy().astype(np.float64, copy=False)
    else:
        _arr = np.asarray(X, dtype=np.float64)

    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    _arr_imp = imputer.fit_transform(_arr)
    scaler = StandardScaler()
    _arr_out = scaler.fit_transform(_arr_imp)

    try:
        model._mlframe_nan_imputer = imputer
        model._mlframe_nan_scaler = scaler
    except (AttributeError, TypeError):
        pass

    if hasattr(X, "columns"):
        import pandas as pd
        X_clean = pd.DataFrame(
            _arr_out, columns=list(X.columns),
            index=getattr(X, "index", None),
        )
    else:
        X_clean = _arr_out
    return np.asarray(fn(X_clean))


def prime_nan_guard_stats(
    model: Any, X_train: Any,
) -> None:
    """Public helper for training code to PRIME the NaN guard.

    Call once at fit time on the training-tail frame so subsequent
    predict-time invocations of ``_apply_nan_guard`` reuse the
    train-fit imputer + scaler instead of fitting on predict data.

    Primes the imputer + scaler unconditionally: even when the
    training frame is NaN-free, the persisted scaler's mean_/scale_
    are needed at predict if the predict frame DOES carry NaN (the
    guard transforms the post-imputation array with these stats).
    Skipping the prime on clean-train would mean predict-time
    ``_apply_nan_guard`` falls back to fitting on the predict frame
    -- the original leakage bug.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    if hasattr(X_train, "to_numpy"):
        # Symmetric polars zero-copy fastpath with the predict-time path so prime/predict don't disagree on bridge mechanics.
        try:
            import polars as _pl
            _is_polars = isinstance(X_train, _pl.DataFrame)
        except ImportError:
            _is_polars = False
        if _is_polars:
            try:
                _arr = X_train.to_numpy(allow_copy=False)
                if _arr.dtype != np.float64:
                    _arr = _arr.astype(np.float64, copy=False)
            except (TypeError, RuntimeError):
                _arr = X_train.to_numpy()
                if _arr.dtype != np.float64:
                    _arr = _arr.astype(np.float64, copy=False)
        else:
            try:
                _arr = X_train.to_numpy(dtype=np.float64)
            except TypeError:
                _arr = X_train.to_numpy().astype(np.float64, copy=False)
    else:
        _arr = np.asarray(X_train, dtype=np.float64)

    imputer = SimpleImputer(strategy="mean", keep_empty_features=True).fit(_arr)
    scaler = StandardScaler().fit(imputer.transform(_arr))
    try:
        model._mlframe_nan_imputer = imputer
        model._mlframe_nan_scaler = scaler
    except (AttributeError, TypeError):
        pass
