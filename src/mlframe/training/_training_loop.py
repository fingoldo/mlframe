"""Training loop helpers extracted from ``trainer.py``.

Core training path: model fitting with CatBoost/LGB/XGB fallbacks,
early stopping, OOM recovery, and post-hoc probability calibration.
"""

from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import Any

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


from mlframe.config import CATBOOST_MODEL_TYPES
from mlframe.core.helpers import get_model_best_iter

from ._eval_helpers import _align_xgb_cat_categories

# Refit helpers + their module-level constants moved to sibling
# ``_training_loop_refit.py`` to drop this file below the 1k-LOC
# monolith threshold; imported here so callers keep using
# ``from mlframe.training._training_loop import _maybe_refit_on_*``.
from ._training_loop_refit import (  # noqa: F401
    _maybe_refit_on_collapsed_predictions,
    _maybe_refit_on_degenerate_best_iter,
)
from .cb import (
    _maybe_get_or_build_cb_pool,
    _maybe_rewrite_eval_set_as_cb_pool,
    _polars_schema_diagnostic,
)
from .helpers import CB_DEFAULT_OCCURRENCE_LOWER_BOUND, compute_cb_text_processing
from .phases import phase
from .pipeline import prepare_df_for_catboost as _prep_cb
from .utils import get_pandas_view_of_polars_df
from .utils import maybe_clean_ram_adaptive as _maybe_clean_ram

logger = logging.getLogger(__name__)


def _in_interactive_notebook() -> bool:
    """True only inside an IPython/Jupyter kernel where CB's live plot makes sense."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and type(ip).__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _maybe_disable_cb_plot(model_type_name: str, fit_params: dict, verbose: bool) -> None:
    """Set ``plot=False`` for CatBoost fits outside an interactive notebook.

    Without it CB spawns+joins a MetricVisualizer/ipywidget plot thread (~1.5s/fit)
    even headless / ``verbose=0``. Pure-config: no numerics change. Respects an
    explicit user-supplied ``plot`` and never disables inside a live Jupyter kernel
    unless verbose is off.
    """
    if model_type_name not in CATBOOST_MODEL_TYPES or "plot" in fit_params:
        return
    if _in_interactive_notebook() and verbose:
        return
    fit_params["plot"] = False


def _ensure_cb_mtr_loss(model, train_target, pool=None) -> None:
    """When target is (N, K>=2) continuous but CatBoostRegressor lacks an
    MTR-compatible ``loss_function``, set ``loss_function='MultiRMSE'``
    pre-fit.

    CatBoost rejects 2-D continuous ``y`` with the default ``RMSE`` loss
    (``Currently only multi-regression, multilabel and survival
    objectives work with multidimensional target``). The dispatch parallels
    ``_ensure_cb_multilabel_loss`` for CatBoostClassifier+multilabel.
    """
    if model is None:
        return
    if type(model).__name__ != "CatBoostRegressor":
        return
    try:
        get = getattr(model, "get_param", None) or getattr(model, "get_params", None)
        params = get() if callable(get) else {}
    except Exception:
        params = {}
    _existing = params.get("loss_function")
    # Skip when the user already wired a multi-target-compatible loss.
    if _existing is not None and "multi" in str(_existing).lower():
        return
    label_arr = None
    if pool is not None:
        try:
            label_arr = np.asarray(pool.get_label())
        except Exception:
            label_arr = None
    if label_arr is None:
        label_arr = np.asarray(train_target) if train_target is not None else None
        if label_arr is not None and label_arr.dtype == object and label_arr.ndim == 1 and label_arr.shape[0] > 0:
            try:
                # np.array(<object-array>.tolist()) stacks the per-row label vectors
                # ~2.5x faster than the np.asarray-per-row listcomp at n=100k (object
                # tolist() yields the row arrays as-is, then np.array stacks them).
                # Bit-identical for uniform-width rows; ragged rows still raise here and
                # hit the except below exactly as the prior np.stack did.
                label_arr = np.array(label_arr.tolist())
            except Exception:
                label_arr = None
    if label_arr is None or label_arr.ndim != 2 or label_arr.shape[1] < 2:
        return
    # Only fires for continuous (float) labels; integer 2-D is multilabel
    # and handled by ``_ensure_cb_multilabel_loss`` instead.
    if label_arr.dtype.kind not in ("f",):
        return
    try:
        model.set_params(loss_function="MultiRMSE", eval_metric="MultiRMSE")
    except Exception:
        try:
            model._init_params["loss_function"] = "MultiRMSE"
            model._init_params["eval_metric"] = "MultiRMSE"
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in _training_loop.py:128: %s", e)
            pass


def _ensure_cb_multilabel_loss(model, train_target, pool=None) -> None:
    """When target is multilabel-shaped but CatBoost lacks loss_function,
    set loss_function='MultiLogloss' pre-fit."""
    if model is None:
        return
    if type(model).__name__ != "CatBoostClassifier":
        return
    try:
        get = getattr(model, "get_param", None) or getattr(model, "get_params", None)
        params = get() if callable(get) else {}
    except Exception:
        params = {}
    if params.get("loss_function") is not None:
        return
    label_arr = None
    if pool is not None:
        try:
            label_arr = np.asarray(pool.get_label())
        except Exception:
            label_arr = None
    if label_arr is None:
        label_arr = np.asarray(train_target) if train_target is not None else None
        if label_arr is not None and label_arr.dtype == object and label_arr.ndim == 1 and label_arr.shape[0] > 0:
            try:
                # np.array(<object-array>.tolist()) stacks the per-row label vectors
                # ~2.5x faster than the np.asarray-per-row listcomp at n=100k (object
                # tolist() yields the row arrays as-is, then np.array stacks them).
                # Bit-identical for uniform-width rows; ragged rows still raise here and
                # hit the except below exactly as the prior np.stack did.
                label_arr = np.array(label_arr.tolist())
            except Exception as _e_stack:
                # Stack failure -> label_arr stays None -> the function
                # returns without configuring MultiLogloss / HammingLoss,
                # so a multilabel CatBoost ends up training with the
                # single-label default loss. Caller wouldn't see why.
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "multilabel CB auto-config: failed to stack label rows "
                    "(%s); CatBoost will use the single-label default loss "
                    "instead of MultiLogloss. Pass a 2D label array to bypass.",
                    _e_stack,
                )
                label_arr = None
    if label_arr is None or label_arr.ndim != 2:
        return
    try:
        model.set_params(loss_function="MultiLogloss", eval_metric="HammingLoss")
    except Exception:
        try:
            model._init_params["loss_function"] = "MultiLogloss"
            model._init_params["eval_metric"] = "HammingLoss"
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in _training_loop.py:183: %s", e)
            pass


def _handle_oom_error(model_obj, model_type_name: str) -> bool:
    """Attempt to recover from an OOM error by clearing caches and
    returning True if the caller should retry the fit.
    """
    import gc
    gc.collect()
    # Clear LGB/XGB/CB internal caches if accessible.
    for _attr in ("_Booster", "_cached_train_features", "_cached_val_features"):
        if hasattr(model_obj, _attr):
            try:
                delattr(model_obj, _attr)
            except Exception:  # nosec B110 - exception already logged below, non-fatal by design
                pass
    logger.warning(
        "OOM during %s.fit; cleared caches and will retry once.",
        model_type_name,
    )
    return True


# Wave 94 (2026-05-21): post-hoc calibration wrappers
# (_SigmoidAdapter, _PostHocCalibratedModel, _PerClassIsotonicCalibrator,
# _PostHocMultiCalibratedModel, _maybe_apply_posthoc_calibration) moved
# to sibling file _calibration_models.py to drop this file below the
# 1k-line monolith threshold. Re-exported below so existing callers
# (`from ._training_loop import _PostHocCalibratedModel`, etc.) keep working.
from ._calibration_models import (  # noqa: F401, E402
    _maybe_apply_posthoc_calibration,
    _PerClassIsotonicCalibrator,
    _PostHocCalibratedModel,
    _PostHocMultiCalibratedModel,
    _SigmoidAdapter,
)


def _train_model_with_fallback(
    model: Any,
    model_obj: Any,
    model_type_name: str,
    train_df: pd.DataFrame | np.ndarray,
    train_target: pd.Series | np.ndarray,
    fit_params: dict[str, Any],
    verbose: bool = False,
) -> tuple[Any, int | None]:
    """Train model with automatic GPU->CPU fallback on OOM errors.

    Parameters
    ----------
    model : Any
        Model to train (may be a Pipeline).
    model_obj : Any
        The actual estimator object (extracted from Pipeline if needed).
    model_type_name : str
        Name of the model type (e.g., 'CatBoostClassifier').
    train_df : pd.DataFrame or np.ndarray
        Training features.
    train_target : pd.Series or np.ndarray
        Training target values.
    fit_params : dict
        Additional parameters for model.fit().
    verbose : bool, default=False
        Whether to log verbose output.

    Returns
    -------
    tuple
        (trained_model, best_iteration) where best_iteration may be None.
    """
    t0_fit = timer()
    # 0-feature train frame is unfittable: CatBoost raises ``CatBoostError: Input data must have at least one feature``,
    # XGBoost raises an opaque DMatrix IndexError, and the linear/sklearn estimators raise their own validate_data errors.
    # The suite-level guard at ``_trainer_train_and_evaluate`` already skips the common FS-empties-everything case, but
    # any column-dropping step between that check and this fit primitive (or a direct caller) can still arrive 0-feature.
    # Mirror the empty-FS warning + return ``(None, None)`` so the caller's ``if model is None: skip`` path handles it,
    # rather than letting the per-backend C++ crash abort the whole suite run.
    _n_feat = None
    if train_df is not None and hasattr(train_df, "shape") and len(getattr(train_df, "shape", ())) == 2:
        _n_feat = train_df.shape[1]
    if _n_feat == 0:
        logger.warning(
            "Skipping %s fit: train frame has 0 features (feature selection / column dropping removed every column). "
            "Nothing to fit -- the model is skipped instead of crashing the backend.",
            model_type_name,
        )
        return None, None
    # CB-only: reuse a single ``catboost.Pool`` across weight schemas
    # and same-target_type targets by mutating the Pool's label/weight
    # in place instead of letting the sklearn wrapper rebuild from X on
    # every fit. Gated on:
    #   * model is CatBoost-family;
    #   * installed CatBoost exposes ``Pool.set_label`` and
    #     ``Pool.set_weight`` (callable);
    #   * ``CatBoostClassifier.fit(X=Pool)`` is the idiomatic native path
    #     (short-circuits rebuild in ``_build_train_pool``).
    # XGB/LGB are not yet covered -- their sklearn wrappers don't accept
    # pre-built DMatrix/Dataset yet (upstream FRs drafted in
    # ``D:\Machine Learning\3rdParty\reproducers\upstream_feature_requests\``).
    # The per-build logging is what makes their rebuild cost visible.
    _cb_pool = _maybe_get_or_build_cb_pool(
        model_type_name=model_type_name,
        model=model,
        train_df=train_df,
        train_target=train_target,
        fit_params=fit_params,
    )
    # Also reuse the val Pool across fits.
    # Rewrites fit_params['eval_set'] from (val_df, val_target) to a
    # cached Pool so CB's sklearn wrapper short-circuits the val-side
    # rebuild too. Only fires when _cb_pool is active (train-side reuse
    # succeeded) -- otherwise the mixed-container path (train=df,
    # eval_set=pool) confuses CB's fit signature.
    if _cb_pool is not None and model_type_name in CATBOOST_MODEL_TYPES:
        _maybe_rewrite_eval_set_as_cb_pool(fit_params)
    # Diagnostic: log the type+module of train_df right before model.fit so
    # silent type drift is visible in the log (Polars vs pandas vs numpy).
    # Critical: type(pl.DataFrame).__name__ == "DataFrame" -- same as pandas --
    # so we log the module too, otherwise "DataFrame" can hide a Polars frame
    # that should have been converted upstream.
    _is_polars = isinstance(train_df, pl.DataFrame)
    _is_pandas = isinstance(train_df, pd.DataFrame)
    try:
        if _is_polars:
            _kind = "pl.DataFrame"
        elif _is_pandas:
            _kind = "pd.DataFrame"
        elif isinstance(train_df, np.ndarray):
            _kind = f"np.ndarray(dtype={train_df.dtype})"
        else:
            _kind = type(train_df).__name__
        if hasattr(train_df, "dtypes") and hasattr(train_df, "columns"):
            _dtype_summary = ", ".join(f"{c}={train_df[c].dtype}" for c in list(train_df.columns)[:5])
            if len(train_df.columns) > 5:
                _dtype_summary += f", ... ({len(train_df.columns)} cols total)"
        else:
            _dtype_summary = ""
        logger.info("  [pre-fit] train_df type=%s, %s", _kind, _dtype_summary)
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed in _training_loop.py:323: %s", e)
        pass

    # Polars-frame contract: only CatBoost, XGBoost, and HistGradientBoosting
    # accept a Polars frame natively at fit time -- their strategies carry
    # ``supports_polars=True``. Everyone else (LGB, sklearn, linear, ridge,
    # ...) MUST arrive with pandas; if a pl.DataFrame gets here for them, the
    # upstream lazy-conversion -> pipeline_cache -> process_model chain has a
    # leak. Previously the trainer silently ran a second polars->pandas
    # conversion as a "self-heal" -- which hid a regression where
    # ``pipeline_cache`` crossed streams between XGB (polars-native,
    # ``cache_key="tree" + tier(False,False)``) and LGB (same key) -- LGB kept
    # pulling XGB's polars frame out of cache and paying a duplicate 224 s
    # conversion. The pipeline_cache fix (container-kind in key, core.py) is
    # the real fix; this raise is the guard that ensures future leaks are
    # caught at the trainer boundary instead of being papered over.
    _POLARS_NATIVE_FIT_MODEL_PREFIXES = (
        "CatBoost",  # CatBoostClassifier / CatBoostRegressor / CatBoost
        "XGB",  # XGBClassifier / XGBRegressor / XGBRanker
        "HistGradient",  # HistGradientBoostingClassifier / Regressor
    )
    # Look through MultiOutputClassifier wrapper for the polars-native check.
    # The wrapper's `estimator` is the per-label base; if the base is polars-native
    # (e.g. HGB), each per-label fit will accept polars too.
    # _ChainEnsemble is the multilabel chain ensemble -- its inner is exposed
    # as `base_estimator` (cloned per chain at fit time).
    _effective_model_type_name = model_type_name
    if model_type_name in ("MultiOutputClassifier", "MultiOutputRegressor", "ClassifierChain"):
        inner = getattr(model, "estimator", None)
        if inner is not None:
            _effective_model_type_name = type(inner).__name__
    elif model_type_name == "_ChainEnsemble":
        inner = getattr(model, "base_estimator", None)
        if inner is not None:
            _effective_model_type_name = type(inner).__name__
    if _is_polars and not any(_effective_model_type_name.startswith(p) for p in _POLARS_NATIVE_FIT_MODEL_PREFIXES):
        raise RuntimeError(
            f"{model_type_name} received pl.DataFrame at fit time "
            f"(shape={train_df.shape}, id={id(train_df)}). Only Polars-native "
            f"strategies (CatBoost, XGBoost, HistGradientBoosting) may receive "
            f"polars -- everyone else needs pandas via the core.py lazy-"
            f"conversion path. Most likely cause: ``pipeline_cache`` returned "
            f"a polars frame cached by a polars-native strategy under a "
            f"``cache_key`` that collides with this strategy's key (see the "
            f"kind-suffix fix in core.py). Diagnose via "
            f"pipeline_cache keys + id() -- do NOT add another silent "
            f"self-heal."
        )

    # Defensive null-fill for pandas categorical features handed to CatBoost.
    # The polars-native path's ``_polars_fill_null_in_categorical`` plus
    # ``prepare_df_for_catboost`` cover most cases, but the multilabel
    # codepath (MultiOutputClassifier wrapping) re-slices the frame after
    # those fills run -- by the time the per-label CB fit lands here, val
    # / test rows may carry raw NaN in cat columns and CB raises ``Invalid
    # type for cat_feature ... =NaN`` (fuzz c0062). Mirror the polars
    # __MISSING__ sentinel for the pandas surface so the bug is patched
    # at the trainer boundary regardless of which upstream path led here.
    if _is_pandas and model_type_name in CATBOOST_MODEL_TYPES and isinstance(train_df, pd.DataFrame):
        # CatBoost Pool rejects category-dtype columns absent from cat_features
        # with "has dtype 'category' but is not in cat_features list". The
        # ordinal-encoding auto-flip path narrows cat_features when text/
        # embedding routing reclassifies a column without reverting its
        # category dtype on the frame. Reconcile here: widen cat_features
        # with any category-dtype column not already routed elsewhere, and
        # cast category-dtype columns routed to text/embedding back to object.
        _text_set = set(fit_params.get("text_features") or [])
        _emb_set = set(fit_params.get("embedding_features") or [])
        _cat_dtype_cols = [c for c, dt in zip(train_df.columns, train_df.dtypes) if isinstance(dt, pd.CategoricalDtype)]
        _explicit_cats = set(fit_params.get("cat_features") or [])
        _missing_cats = [c for c in _cat_dtype_cols if c not in _explicit_cats and c not in _text_set and c not in _emb_set]
        if _missing_cats:
            fit_params["cat_features"] = sorted(_explicit_cats | set(_missing_cats))
        _decategorise_for_text_or_emb = [c for c in _cat_dtype_cols if c in _text_set or c in _emb_set]
        if _decategorise_for_text_or_emb:
            train_df = train_df.copy() if not getattr(train_df, "_mlframe_filled", False) else train_df
            for _c in _decategorise_for_text_or_emb:
                # Use ``astype("string").fillna(sentinel)`` so OOV-null cells
                # (from the train+val joint Enum cast, strict=False on test)
                # don't surface as NaN -- CB rejects NaN in cat_features with
                # "Invalid type for cat_feature ... NaN: cat_features must be
                # integer or string". Sentinel matches the upstream Polars
                # __MISSING__ pattern.
                train_df[_c] = train_df[_c].astype("string").fillna("__MISSING__").astype(object)
            train_df._mlframe_filled = True
            # Mirror onto eval_set for early-stopping evaluation.
            _eval_set = fit_params.get("eval_set")
            if _eval_set:
                _new_eval_set = []
                for pair in _eval_set:
                    if isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], pd.DataFrame):
                        _eval_df, _eval_y = pair
                        _eval_df_filled = _eval_df
                        for _c in _decategorise_for_text_or_emb:
                            if _c in _eval_df_filled.columns:
                                if not getattr(_eval_df_filled, "_mlframe_filled", False):
                                    _eval_df_filled = _eval_df_filled.copy()
                                _eval_df_filled[_c] = _eval_df_filled[_c].astype("string").fillna("__MISSING__").astype(object)
                                _eval_df_filled._mlframe_filled = True
                        _new_eval_set.append((_eval_df_filled, _eval_y))
                    else:
                        _new_eval_set.append(pair)
                fit_params["eval_set"] = _new_eval_set

    if _is_pandas and model_type_name in CATBOOST_MODEL_TYPES and "cat_features" in fit_params and fit_params["cat_features"] and isinstance(train_df, pd.DataFrame):
        _cat_cols = [c for c in fit_params["cat_features"] if c in train_df.columns]
        if _cat_cols:
            for _c in _cat_cols:
                _s = train_df[_c]
                if _s.isna().any():
                    train_df = train_df.copy() if not getattr(train_df, "_mlframe_filled", False) else train_df
                    train_df[_c] = _s.astype("string").fillna("__MISSING__").astype("category")
                    train_df._mlframe_filled = True
            # Symmetric fill on eval_set so val/test slices don't trip the
            # same NaN check during early-stopping evaluation.
            _eval_set = fit_params.get("eval_set")
            if _eval_set:
                _new_eval_set = []
                for pair in _eval_set:
                    if isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], pd.DataFrame):
                        _eval_df, _eval_y = pair
                        _eval_df_filled = _eval_df
                        for _c in _cat_cols:
                            if _c in _eval_df_filled.columns and _eval_df_filled[_c].isna().any():
                                if not getattr(_eval_df_filled, "_mlframe_filled", False):
                                    _eval_df_filled = _eval_df_filled.copy()
                                _eval_df_filled[_c] = _eval_df_filled[_c].astype("string").fillna("__MISSING__").astype("category")
                                _eval_df_filled._mlframe_filled = True
                        _new_eval_set.append((_eval_df_filled, _eval_y))
                    else:
                        _new_eval_set.append(pair)
                fit_params["eval_set"] = _new_eval_set

    # Dynamic CB ``text_processing`` calibration: scale ``occurrence_lower_bound``
    # to the training-row count whenever this is a CatBoost fit with text
    # features. Default CB OLB=50 hangs on small folds (RFECV inner CV +
    # outlier-detection trim, fuzz c0056/c0070); ``compute_cb_text_processing``
    # returns None (no-op) when the row count is high enough to leave the
    # default in place. Skipped when the user has explicitly set
    # ``text_processing`` via cb_kwargs (already on the estimator's params).
    if model_type_name in CATBOOST_MODEL_TYPES:
        _has_text = bool(fit_params.get("text_features")) or (_cb_pool is not None and bool(getattr(_cb_pool, "_mlframe_text_features", None)))
        if _has_text:
            _cb_n_rows = train_df.shape[0] if hasattr(train_df, "shape") else (len(_cb_pool) if _cb_pool is not None and hasattr(_cb_pool, "__len__") else None)
            _user_text_proc = None
            if hasattr(model, "get_params"):
                try:
                    _user_text_proc = model.get_params().get("text_processing")
                except Exception:
                    _user_text_proc = None
            if _user_text_proc is None:
                _tp = compute_cb_text_processing(_cb_n_rows) if _cb_n_rows is not None else None
                if _tp is not None and hasattr(model, "set_params"):
                    try:
                        model.set_params(text_processing=_tp)
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "[%s] scaled CB text_processing.occurrence_lower_bound to %s " "(n_train=%s, default would be %s).",
                                model_type_name,
                                _tp["dictionaries"][0]["occurrence_lower_bound"],
                                _cb_n_rows,
                                CB_DEFAULT_OCCURRENCE_LOWER_BOUND,
                            )
                    except Exception as _tp_exc:
                        # Non-fatal: if CB version rejects this shape we
                        # fall back to the post-fit "Dictionary size is 0"
                        # recovery path below.
                        logger.warning(
                            "[%s] failed to set scaled CB text_processing (%s); " "falling back to post-fit recovery.",
                            model_type_name,
                            _tp_exc,
                        )

    try:
        # Final cat alignment right before fit, by which point any
        # upstream polars->pandas conversion has run. Targets a flake
        # where a prior polars_nullable->pandas conversion leaves a
        # later case's pandas frame with a pd.CategoricalDtype whose
        # categories list disagrees between train and val/test.
        # Re-align here so XGB's stored cat index matches at predict.
        _eval_set_for_align = fit_params.get("eval_set")
        _val_df_from_eval = None
        if _eval_set_for_align:
            _pairs = _eval_set_for_align if isinstance(_eval_set_for_align, list) else [_eval_set_for_align]
            for _p in _pairs:
                if isinstance(_p, tuple) and len(_p) == 2 and isinstance(_p[0], pd.DataFrame):
                    _val_df_from_eval = _p[0]
                    break
        if isinstance(train_df, pd.DataFrame) and _val_df_from_eval is not None:
            train_df, _aligned_val, _ = _align_xgb_cat_categories(
                model_type_name,
                train_df,
                val_df=_val_df_from_eval,
                test_df=None,
            )
            # Refresh eval_set if val_df was modified (set_categories
            # returns a new Series; the eval_set tuple needs the new
            # frame reference).
            if _aligned_val is not None and _aligned_val is not _val_df_from_eval:
                if isinstance(_eval_set_for_align, list):
                    fit_params["eval_set"] = [
                        (_aligned_val, _p[1]) if isinstance(_p, tuple) and _p[0] is _val_df_from_eval else _p for _p in _eval_set_for_align
                    ]
                elif isinstance(_eval_set_for_align, tuple):
                    fit_params["eval_set"] = (_aligned_val, _eval_set_for_align[1])

        _maybe_disable_cb_plot(model_type_name, fit_params, verbose)
        with phase(
            "model.fit",
            model=model_type_name,
            n_rows=(train_df.shape[0] if hasattr(train_df, "shape") else None),
            n_cols=(train_df.shape[1] if hasattr(train_df, "shape") else None),
        ):
            if _cb_pool is not None:
                # Reuse path: X=Pool, y omitted (label already on the Pool).
                # fit_params still carries sample_weight, which CB's wrapper
                # ignores when X is a Pool (the Pool already has weight).
                # Filter it explicitly so downstream assertion paths don't
                # flag a "sample_weight with Pool" mismatch.
                _reuse_fit_params = {k: v for k, v in fit_params.items() if k not in ("sample_weight", "cat_features", "text_features", "embedding_features")}
                _ensure_cb_multilabel_loss(model, train_target, pool=_cb_pool)
                _ensure_cb_mtr_loss(model, train_target, pool=_cb_pool)
                _ensure_xgb_classification_objective(model, train_target)
                model = _maybe_wrap_for_2d_target(model, train_target)
                model.fit(_cb_pool, **_reuse_fit_params)
            else:
                _ensure_cb_multilabel_loss(model, train_target)
                _ensure_cb_mtr_loss(model, train_target)
                _ensure_xgb_classification_objective(model, train_target)
                _model_pre_wrap_type = type(model).__name__
                model = _maybe_wrap_for_2d_target(model, train_target)
                # When ``_maybe_wrap_for_2d_target`` introduced a
                # MultiOutputClassifier wrapper, strip ``eval_set`` from
                # fit_params - MOC doesn't slice eval_set per label, so the
                # inner estimator would see a 2-D val y and raise
                # ``y should be a 1d array``. The inner HGB / LGB / Linear
                # classifiers don't accept eval_set anyway. Surfaced by
                # 3-way fuzz cases (cb_hgb_lgb_linear*xgb /
                # multilabel + eval_set passed through).
                if type(model).__name__ == "MultiOutputClassifier" and _model_pre_wrap_type != "MultiOutputClassifier":
                    # Strip val-injected fit_params - MOC doesn't slice
                    # them per label, so the inner estimator's
                    # ``_validate_data`` chokes on the 2-D ``y_val``
                    # with ``y should be a 1d array``. Surfaced 3-way
                    # fuzz c0036 / c0041 / c0045 / c0056 (mixed-model
                    # multilabel suites where ``_setup_eval_set``
                    # injected ``X_val`` / ``y_val`` for inner
                    # gradient-boosting val-set support).
                    _strip_keys = ("eval_set", "X_val", "y_val", "validation_data")
                    fit_params = {k: v for k, v in fit_params.items() if k not in _strip_keys}
                model.fit(train_df, train_target, **fit_params)
    except Exception as e:
        try_again = False
        error_str = str(e)

        if "out of memory" in error_str:
            try_again = _handle_oom_error(model_obj, model_type_name)

        elif "User defined callbacks are not supported for GPU" in error_str:
            if "callbacks" in fit_params:
                logger.warning(e)
                try_again = True
                del fit_params["callbacks"]

        elif "CUDA Tree Learner" in error_str:
            logger.warning("CUDA is not enabled in this LightGBM build. Falling back to CPU.")
            model.set_params(device_type="cpu")
            try_again = True
        elif "pandas dtypes must be int, float or bool" in error_str:
            # Upstream feature-typing gap: a column reached the estimator with a
            # dtype it cannot consume (e.g. object/datetime/Categorical not cast
            # to numeric). A silent ``return None, None`` here hides the real
            # cause and surfaces downstream as an opaque "model produced no
            # predictions" failure. Surface the offending dtypes and re-raise so
            # the upstream type-detection / casting gap is visible and fixable.
            _dtypes = getattr(train_df, "dtypes", None)
            logger.error(
                "Model %s received a column with an unsupported pandas dtype "
                "(estimator requires int/float/bool). Offending frame dtypes: %s. "
                "Fix upstream feature typing / casting -- do not pass object / "
                "datetime / un-encoded categorical columns to this estimator.",
                model_type_name, _dtypes,
            )
            raise

        elif model_type_name in CATBOOST_MODEL_TYPES and "Dictionary size is 0" in error_str:
            # CatBoost's text feature estimator failed to build a TF-IDF
            # vocabulary -- the column's non-null samples, after the
            # occurrence_lower_bound filter, leave an empty dictionary.
            # Root cause: columns auto-promoted to text_features that
            # have >99.9% null rows (e.g.
            # _raw_countries, job_post_source with 6-20 non-null
            # strings out of 810_000). Proactive guard in
            # _auto_detect_feature_types now blocks these at promotion
            # time, but this is the defensive fallback: on the exact
            # CB error, drop text_features from fit_params and retry
            # without text processing. The columns stay in the frame
            # (CB will treat them as plain categorical-by-name or
            # ignore).
            text_feat = fit_params.get("text_features") or []
            if text_feat:
                logger.warning(
                    "CatBoost raised 'Dictionary size is 0' on text_features %s -- "
                    "the column(s) have too few non-null samples for CB's TF-IDF "
                    "estimator to build a vocabulary. Dropping text_features from "
                    "fit_params and retrying. Fix upstream: block promotion of "
                    "sparse columns in _auto_detect_feature_types (see the "
                    "min_non_null_for_text_promotion guard), or increase "
                    "non-null coverage of these columns in your feature "
                    "extraction.",
                    text_feat,
                )
                # Reroute the dropped text columns to cat_features so CB's
                # categorical handling still sees them (otherwise CB tries to
                # cast the string values to float on the retry and raises
                # ``Cannot convert 'X' to float``).
                _existing_cats = list(fit_params.get("cat_features") or [])
                _moved_to_cat = [c for c in text_feat if c not in _existing_cats]
                fit_params = {k: v for k, v in fit_params.items() if k != "text_features"}
                if _moved_to_cat:
                    fit_params["cat_features"] = _existing_cats + _moved_to_cat
                try_again = True
            else:
                # Raise -- same error without text_features in params is
                # an unexpected variant, not our problem.
                pass

        elif (
            model_type_name in CATBOOST_MODEL_TYPES
            and isinstance(train_df, pl.DataFrame)
            and (
                "No matching signature found" in error_str
                # Catch *both* "Categorical for a numerical feature column" and
                # "Categorical for a text feature column" (the latter surfaces
                # when a column auto-promoted from cat_features -> text_features
                # is still pl.Categorical in the df). Upstream fix casts those
                # columns to pl.String before CB.fit; this is a safety net for
                # any future variant of the same error family.
                or "Unsupported data type Categorical" in error_str
            )
        ):
            # CatBoost's native-Polars fastpath (_set_features_order_data_polars_*)
            # can reject certain categorical column layouts with opaque messages --
            # either "No matching signature found" (fused cpdef dispatch miss) or
            # the categorical/numeric type mismatch above. Fall back to the pandas
            # path: zero-copy Arrow view + `prepare_df_for_catboost` preserves
            # dtypes and CatBoost's pandas path accepts a wider
            # range of category backings.
            # Full last-line for the one-line message, plus a structured
            # schema dump so the NEXT occurrence is diagnosable from the
            # first log line (prev. we only had the truncated error str,
            # and for opaque dispatch misses that's useless).
            last_line = error_str.splitlines()[-1] if error_str else "<empty>"
            logger.warning(
                "CatBoost Polars fastpath rejected the data (%s); " "converting to pandas and retrying.",
                last_line[:240],
            )
            # Mark the model "Polars-broken" so subsequent predict_proba /
            # predict_log_proba calls via _predict_with_fallback go straight
            # to the pandas path -- avoids the same Cython dispatch miss on
            # every VAL/TEST/ensemble scoring (one WARN + one ~2s retry per
            # call saved). See the symmetric short-circuit in
            # _predict_with_fallback.
            try:
                model._mlframe_polars_fastpath_broken = True
            except Exception as _mark_broken_err:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                # Deliberately NOT named `e`: this is nested inside the outer `except Exception as e:` handler,
                # and Python implicitly `del`s the exception name at the end of its own except clause -- reusing
                # `e` here would delete the OUTER `e` too, breaking the `raise e` re-raise further down.
                logger.debug("suppressed in _training_loop.py:688: %s", _mark_broken_err)
                pass
            schema_dump = _polars_schema_diagnostic(
                train_df,
                cat_features=fit_params.get("cat_features"),
                text_features=fit_params.get("text_features"),
            )
            logger.warning("CB Polars fastpath failure -- schema context:\n%s", schema_dump)
            # ``get_pandas_view_of_polars_df`` and ``prepare_df_for_catboost`` (as _prep_cb) are now
            # imported at module top; re-importing them on every retry inside the exception handler
            # paid sys.modules lookup cost N times per training session.

            cat_feat = list(fit_params.get("cat_features") or [])
            text_feat = list(fit_params.get("text_features") or [])

            def _decategorize_text_cols(df):
                """CatBoost's pandas path rejects columns that are pd.Categorical
                but not in cat_features with "column 'X' has dtype 'category' but
                is not in cat_features list". Columns auto-promoted from
                cat_features -> text_features keep a pd.Categorical dtype after
                the Polars->pandas zero-copy conversion. Cast those to plain
                object to keep CB happy (and preserve the string content).
                """
                if not text_feat:
                    return df
                for col in text_feat:
                    if col in df.columns and isinstance(df[col].dtype, pd.CategoricalDtype):
                        df[col] = df[col].astype("object").fillna("")
                return df

            # Per-step timing for the fallback: a production run showed this
            # entire path consumed >1 hour on a 1M x 98 frame with 4
            # high-cardinality text columns -- without timing it was impossible
            # to tell which step (Polars->pandas vs. prep_cb vs. decategorize)
            # was responsible. The timer log writes to the trainer logger so
            # the lines interleave with surrounding INFO output.
            t0_fb = timer()
            shape_str = f"{train_df.shape[0]:_}x{train_df.shape[1]}" if hasattr(train_df, "shape") else "?"

            t0 = timer()
            train_df = get_pandas_view_of_polars_df(train_df)
            logger.info("  [fallback] polars->pandas(train) %s in %.1fs", shape_str, timer() - t0)

            # IMPORTANT: decategorize text columns BEFORE prepare_df_for_catboost.
            # Otherwise prep_cb hits the pd.Categorical text columns (auto-promoted
            # from cat to text earlier) and runs
            #   df[col].astype(str).fillna("").astype("category")
            # which on a high-cardinality column like skills_text (81k unique
            # values over 810k rows) takes many minutes per column -- the
            # production-reproduced hang that motivated this reorder.
            t0 = timer()
            train_df = _decategorize_text_cols(train_df)
            logger.info("  [fallback] decategorize text cols(train) in %.1fs", timer() - t0)

            t0 = timer()
            _prep_cb(train_df, cat_features=cat_feat)  # in-place; text_feat already decategorised above
            logger.info("  [fallback] prepare_df_for_catboost(train) in %.1fs", timer() - t0)

            # eval_set carries the val split for CB -- rewrite it too.
            eval_set = fit_params.get("eval_set")
            if eval_set is not None:
                t0_es = timer()
                pairs = eval_set if isinstance(eval_set, list) else [eval_set]
                new_pairs = []
                for pair in pairs:
                    X_val, y_val = pair
                    if isinstance(X_val, pl.DataFrame):
                        X_val = get_pandas_view_of_polars_df(X_val)
                        # ``get_pandas_view_of_polars_df`` keys each frame's
                        # pl.Categorical->pl.Enum remap on THAT frame's own
                        # unique values, so train and val (converted separately
                        # above) get DIVERGING pandas Categorical ``categories``
                        # lists -- the same string maps to different integer
                        # codes across the fit/eval boundary (a silent mis-encode
                        # for any code-consuming backend). Re-align to the
                        # train+val category union (leak-free: val already feeds
                        # early stopping) before prep so codes match.
                        train_df, X_val, _ = _align_xgb_cat_categories(
                            model_type_name, train_df, val_df=X_val, test_df=None,
                        )
                        # Decategorize BEFORE prep_cb (see train_df comment above).
                        X_val = _decategorize_text_cols(X_val)
                        _prep_cb(X_val, cat_features=cat_feat)  # in-place; text_feat already decategorised above
                    else:
                        X_val = _decategorize_text_cols(X_val) if isinstance(X_val, pd.DataFrame) else X_val
                    new_pairs.append((X_val, y_val))
                fit_params["eval_set"] = new_pairs if isinstance(eval_set, list) else new_pairs[0]
                logger.info("  [fallback] eval_set rewrite in %.1fs", timer() - t0_es)

            logger.info("  [fallback] total pandas prep for CB in %.1fs", timer() - t0_fb)
            try_again = True

        elif "unexpected keyword argument" in error_str and any(param in error_str for param in ("X_val", "y_val", "eval_set")):
            # Older sklearn versions don't support validation set in HistGradientBoosting
            val_params = ["X_val", "y_val", "eval_set"]
            removed = [p for p in val_params if p in fit_params]
            if removed:
                logger.warning(
                    f"This sklearn version doesn't support validation set parameters ({', '.join(removed)}) "
                    f"for {model_type_name}. Training without early stopping validation."
                )
                for param in val_params:
                    fit_params.pop(param, None)
                try_again = True

        if try_again:
            _maybe_clean_ram()
            with phase(
                "model.fit",
                model=model_type_name,
                n_rows=(train_df.shape[0] if hasattr(train_df, "shape") else None),
                n_cols=(train_df.shape[1] if hasattr(train_df, "shape") else None),
                retry=True,
            ):
                model.fit(train_df, train_target, **fit_params)
        else:
            raise e

    _maybe_clean_ram()
    fit_elapsed = timer() - t0_fit
    if verbose:
        shape_str = f"{train_df.shape[0]:_}x{train_df.shape[1]}" if hasattr(train_df, "shape") else ""
        logger.info("  model.fit(%s) done -- %s, %.1fs", model_type_name, shape_str, fit_elapsed)

    # Apply post-hoc isotonic calibration to binary classifiers that were
    # tagged with ``_mlframe_posthoc_calibrate=True``. Without this the
    # ``prefer_calibrated_classifiers=True`` flag is a no-op on tree
    # models.
    try:
        model = _maybe_apply_posthoc_calibration(model, fit_params, model_type_name, verbose=verbose)
    except Exception as _calib_err:
        logger.warning(f"Post-hoc calibration hook raised: {_calib_err}")

    best_iter = None
    if model is not None:
        try:
            best_iter = get_model_best_iter(model_obj)
            if best_iter and verbose:
                logger.info("es_best_iter: %d", best_iter)
        except (AttributeError, TypeError, ValueError):
            logger.warning("Could not get best iteration", exc_info=True)

    # Loss-fallback retry on degenerate early stopping (2026-05-26).
    # Heavy-kurt targets get Huber via ``_apply_loss_recommendation_-
    # in_place``; on EXTREME-kurt (observed +42.67 in prod) the
    # Huber gradient ``delta * sign(residual)`` collapses to ~ 0 when
    # most rows have residual ~ 0, ES fires at iter=0/1, model returns
    # the constant train-mean. Detect + refit with the RMSE-family
    # default. Logic carved into ``_maybe_refit_on_degenerate_best_iter``
    # for focused unit testing.
    if model is not None and best_iter is not None:
        _new_best_iter = _maybe_refit_on_degenerate_best_iter(
            model_obj=model_obj,
            model_type_name=model_type_name,
            best_iter=best_iter,
            train_df=train_df,
            train_target=train_target,
            fit_params=fit_params,
            logger_=logger,
        )
        if _new_best_iter is not None:
            best_iter = _new_best_iter

    # MLP / recurrent collapse detection (2026-05-26 followup): same
    # failure shape as the booster Huber-collapse path -- network
    # converges to a near-constant prediction (output saturation under
    # tanh_train_range + BN-less LeakyReLU, etc). Architecture-
    # agnostic detector via pred-variance ratio; refit with the
    # output bound removed when triggered.
    if model is not None:
        _maybe_refit_on_collapsed_predictions(
            model=model,
            model_obj=model_obj,
            model_type_name=model_type_name,
            train_df=train_df,
            train_target=train_target,
            fit_params=fit_params,
            logger_=logger,
        )

    return model, best_iter


# xgb-objective / 2d-target-wrap helpers carved to _training_loop_objectives.py (1k-LOC ceiling).
from ._training_loop_objectives import (  # noqa: E402, F401
    _ensure_xgb_classification_objective,
    _maybe_wrap_for_2d_target,
)
