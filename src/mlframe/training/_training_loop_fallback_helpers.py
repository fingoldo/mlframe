"""Helpers carved out of ``_training_loop`` to keep that module under its size budget."""

from __future__ import annotations

import logging
from timeit import default_timer as timer

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


from mlframe.training._hgb_polars_categorical import pin_hgb_categorical_features_for_polars
from mlframe.config import CATBOOST_MODEL_TYPES
from mlframe.core.helpers import get_model_best_iter

from ._eval_helpers import _align_xgb_cat_categories

# Refit helpers + their module-level constants moved to sibling
# ``_training_loop_refit.py`` to drop this file below the 1k-LOC
# monolith threshold; imported here so callers keep using
# ``from mlframe.training._training_loop import _maybe_refit_on_*``.
from ._training_loop_text_retry import retry_params_after_empty_text_dictionary
from ._training_loop_refit import (
    _maybe_refit_on_collapsed_predictions,
    _maybe_refit_on_best_iter_pathology,
)
from mlframe.training.cb.shared import (
    polars_schema_diagnostic as _polars_schema_diagnostic,
)
from .helpers import CB_DEFAULT_OCCURRENCE_LOWER_BOUND, compute_cb_text_processing
from .phases import phase
from .pipeline import prepare_df_for_catboost as _prep_cb
from .utils import get_pandas_view_of_polars_df
from .utils import maybe_clean_ram_adaptive as _maybe_clean_ram

logger = logging.getLogger(__name__)
from ._training_loop_cb_utils import (
    _ensure_cb_mtr_loss,
    _ensure_cb_multilabel_loss,
    _handle_oom_error,
    _maybe_disable_cb_plot,
    _suppress_catboost_noise,
)
from ._training_loop_objectives import _ensure_xgb_classification_objective
from ._calibration_models import _maybe_apply_posthoc_calibration
from ._training_loop_objectives import _maybe_wrap_for_2d_target



def _train_model_with_f_base_estimator_cloned_per(model_type_name, model, _effective_model_type_name, _is_polars, _POLARS_NATIVE_FIT_MODEL_PREFIXES, train_df, _is_pandas, fit_params, _cb_pool, verbose, train_target, model_obj, t0_fit):
    """Block of _train_model_with_fallback_unguarded starting at ``_effective_model_type_name = _train_model_with_f_base_estimator_cloned``."""
    _effective_model_type_name = _train_model_with_f_base_estimator_cloned_per_2(model_type_name, model, _effective_model_type_name)
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
    train_df = _train_model_with_f_trainer_boundary_regardless_which(_is_pandas, model_type_name, train_df, fit_params)

    train_df = _train_model_with_f_pandas_model_type_name(_is_pandas, model_type_name, fit_params, train_df)

    # Dynamic CB ``text_processing`` calibration: scale ``occurrence_lower_bound``
    # to the training-row count whenever this is a CatBoost fit with text
    # features. Default CB OLB=50 hangs on small folds (RFECV inner CV +
    # outlier-detection trim, fuzz c0056/c0070); ``compute_cb_text_processing``
    # returns None (no-op) when the row count is high enough to leave the
    # default in place. Skipped when the user has explicitly set
    # ``text_processing`` via cb_kwargs (already on the estimator's params).
    _train_model_with_f_text_processing_via_cb(model_type_name, fit_params, _cb_pool, train_df, model)

    try:
        # Final cat alignment right before fit, by which point any
        # upstream polars->pandas conversion has run. Targets a flake
        # where a prior polars_nullable->pandas conversion leaves a
        # later case's pandas frame with a pd.CategoricalDtype whose
        # categories list disagrees between train and val/test.
        # Re-align here so XGB's stored cat index matches at predict.
        _eval_set_for_align = fit_params.get("eval_set")
        _val_df_from_eval = None
        _val_df_from_eval = _train_model_with_f_re_align_here_xgb(_eval_set_for_align, _val_df_from_eval)
        train_df = _train_model_with_f_isinstance_train_df_pd(train_df, _val_df_from_eval, model_type_name, _eval_set_for_align, fit_params)

        _maybe_disable_cb_plot(model_type_name, fit_params, verbose)
        fit_params, model = _train_model_with_f_phase(model_type_name, train_df, _cb_pool, fit_params, model, train_target)
    except Exception as e:
        try_again = False
        error_str = str(e)

        if "KeyboardInterrupt" in error_str:
            # CatBoost catches whatever a python-side custom metric / callback raises and re-raises it as CatBoostError,
            # so a Ctrl+C during the fit arrived here as a model FAILURE: with continue_on_model_failure the suite went
            # on to the next model instead of stopping. KeyboardInterrupt is not an Exception, so re-raising it here
            # passes through every ``except Exception`` between this frame and the caller.
            raise KeyboardInterrupt("interrupted during model fit (CatBoost re-raised it as CatBoostError)") from e

        if "out of memory" in error_str:
            try_again = _handle_oom_error(model_obj, model_type_name)

        elif "User defined callbacks are not supported for GPU" in error_str and "callbacks" in fit_params:
            logger.warning("%s; retrying without callbacks (backstop: CatBoostGpuFitGuard normally strips them before fit)", e)
            try_again, _ = True, fit_params.pop("callbacks")

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
            # CatBoost's text estimator could not build a vocabulary for at least ONE text column, and the
            # raise names none of them -- so the old handler dropped EVERY text feature and retried, silently
            # training without any of the columns that were promoted on purpose.
            #
            # It also blamed the wrong thing ("too few non-null samples"). Measured against CatBoost 1.2.10, a
            # column with 400 non-null rows and 400 distinct tokens still raises, while one whose rows carry
            # three tokens each survives at a token frequency of 3 -- the condition is about token structure,
            # not row counts. ``_cb_text_probe`` asks the installed CatBoost per column instead of re-deriving
            # its vocabulary filter, so only the genuinely unusable columns are dropped.
            try_again, fit_params = retry_params_after_empty_text_dictionary(model, train_df, train_target, fit_params)

        elif (
            model_type_name in CATBOOST_MODEL_TYPES
            and pl is not None
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
                model._mlframe_polars_fastpath_miss_observed = True
            except Exception as _mark_broken_err:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                # Deliberately NOT named `e`: this is nested inside the outer `except Exception as e:` handler,
                # and Python implicitly `del`s the exception name at the end of its own except clause -- reusing
                # `e` here would delete the OUTER `e` too, breaking the `raise e` re-raise further down.
                logger.debug("suppressed: %s", _mark_broken_err)
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
                    if pl is not None and isinstance(X_val, pl.DataFrame):
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
                    "This sklearn version doesn't support validation set parameters (%s) " "for %s. Training without early stopping validation.",
                    ", ".join(removed),
                    model_type_name,
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
                with _suppress_catboost_noise():
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
    except Exception as _calib_err:  # best-effort: model stays uncalibrated, training continues
        logger.warning("Post-hoc calibration hook raised: %s", _calib_err)

    best_iter = None
    best_iter = _train_model_with_f_model_none(model, model_obj, verbose, best_iter)

    # Loss-fallback retry on degenerate early stopping.
    # Heavy-kurt targets get Huber via ``_apply_loss_recommendation_-
    # in_place``; on EXTREME-kurt (observed +42.67 in prod) the
    # Huber gradient ``delta * sign(residual)`` collapses to ~ 0 when
    # most rows have residual ~ 0, ES fires at iter=0/1, model returns
    # the constant train-mean. Detect + refit with the RMSE-family
    # default. Logic carved into ``_maybe_refit_on_degenerate_best_iter``
    # for focused unit testing.
    best_iter = _train_model_with_f_focused_unit_testing(model, best_iter, model_obj, model_type_name, train_df, train_target, fit_params)

    # MLP / recurrent collapse detection: same
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
    return best_iter, model


def _train_model_with_f_trainer_boundary_regardless_which(_is_pandas, model_type_name, train_df, fit_params):
    """Block of _train_model_with_fallback_unguarded starting at ``if _is_pandas and model_type_name in CATBOOST_MODEL_TYPES and isinstan``."""
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
            # Shallow copy: every site below reassigns whole columns (train_df[_c] = ...), never
            # mutates a column's buffer in place, so a deep copy of train/eval frames here is
            # unnecessary (frames here can be 100+GB).
            train_df = train_df.copy(deep=False) if not getattr(train_df, "_mlframe_filled", False) else train_df
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
                                    _eval_df_filled = _eval_df_filled.copy(deep=False)
                                _eval_df_filled[_c] = _eval_df_filled[_c].astype("string").fillna("__MISSING__").astype(object)
                                _eval_df_filled._mlframe_filled = True
                        _new_eval_set.append((_eval_df_filled, _eval_y))
                    else:
                        _new_eval_set.append(pair)
                fit_params["eval_set"] = _new_eval_set
    return train_df


def _train_model_with_f_should_have_been_converted(_is_polars, _is_pandas, train_df):
    """Block of _train_model_with_fallback_unguarded starting at ``try:``."""
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
        logger.debug("suppressed: %s", e)
        pass


def _train_model_with_f_base_estimator_cloned_per_2(model_type_name, model, _effective_model_type_name):
    """Block of _train_model_with_fallback_unguarded starting at ``if model_type_name in ("MultiOutputClassifier", "MultiOutputRegressor"``."""
    if model_type_name in ("MultiOutputClassifier", "MultiOutputRegressor", "ClassifierChain"):
        inner = getattr(model, "estimator", None)
        if inner is not None:
            _effective_model_type_name = type(inner).__name__
    elif model_type_name == "_ChainEnsemble":
        inner = getattr(model, "base_estimator", None)
        if inner is not None:
            _effective_model_type_name = type(inner).__name__
    return _effective_model_type_name


def _train_model_with_f_pandas_model_type_name(_is_pandas, model_type_name, fit_params, train_df):
    """Block of _train_model_with_fallback_unguarded starting at ``if _is_pandas and model_type_name in CATBOOST_MODEL_TYPES and "cat_fea``."""
    if _is_pandas and model_type_name in CATBOOST_MODEL_TYPES and "cat_features" in fit_params and fit_params["cat_features"] and isinstance(train_df, pd.DataFrame):
        _cat_cols = [c for c in fit_params["cat_features"] if c in train_df.columns]
        if _cat_cols:
            for _c in _cat_cols:
                _s = train_df[_c]
                if _s.isna().any():
                    train_df = train_df.copy(deep=False) if not getattr(train_df, "_mlframe_filled", False) else train_df
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
                                    _eval_df_filled = _eval_df_filled.copy(deep=False)
                                _eval_df_filled[_c] = _eval_df_filled[_c].astype("string").fillna("__MISSING__").astype("category")
                                _eval_df_filled._mlframe_filled = True
                        _new_eval_set.append((_eval_df_filled, _eval_y))
                    else:
                        _new_eval_set.append(pair)
                fit_params["eval_set"] = _new_eval_set
    return train_df


def _train_model_with_f_text_processing_via_cb(model_type_name, fit_params, _cb_pool, train_df, model):
    """Block of _train_model_with_fallback_unguarded starting at ``if model_type_name in CATBOOST_MODEL_TYPES:``."""
    if model_type_name in CATBOOST_MODEL_TYPES:
        _has_text = bool(fit_params.get("text_features")) or (_cb_pool is not None and bool(getattr(_cb_pool, "_mlframe_text_features", None)))
        if _has_text:
            # Head off the empty-bigram-dictionary abort. The recovery below only learns about it by FITTING,
            # so a production run paid a full 44.99s doomed fit on 2.4M rows before retrying with unigrams.
            # A whitespace scan over a sample answers the same question in milliseconds, and it only fires
            # when NOT ONE sampled row of a column has two tokens -- the case where a bigram cannot exist.
            try:
                from mlframe.training.cb import single_token_text_features, unigram_text_processing

                _decl_text = list(fit_params.get("text_features") or [])
                _single_tok = single_token_text_features(train_df, _decl_text) if _decl_text else []
                if _single_tok and len(_single_tok) == len(_decl_text) and hasattr(model, "set_params"):
                    if not (model.get_params().get("text_processing") if hasattr(model, "get_params") else None):
                        model.set_params(text_processing=unigram_text_processing())
                        logger.info(
                            "  [pre-fit] %d text feature(s) %s carry a single token per row, so the default "
                            "bigram dictionary would be empty and the fit would abort. Starting with a unigram "
                            "dictionary instead of discovering this after a failed fit.",
                            len(_single_tok), _single_tok,
                        )
            except Exception as _tok_exc:
                logger.debug("single-token pre-scan skipped (%s: %s)", type(_tok_exc).__name__, _tok_exc)
            _cb_n_rows = train_df.shape[0] if hasattr(train_df, "shape") else (len(_cb_pool) if _cb_pool is not None and hasattr(_cb_pool, "__len__") else None)
            _user_text_proc = None
            if hasattr(model, "get_params"):
                try:
                    _user_text_proc = model.get_params().get("text_processing")
                except Exception as e:
                    logger.debug("model.get_params() failed, treating text_processing as unset: %s", e)
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


def _train_model_with_f_re_align_here_xgb(_eval_set_for_align, _val_df_from_eval):
    """Block of _train_model_with_fallback_unguarded starting at ``if _eval_set_for_align:``."""
    if _eval_set_for_align:
        _pairs = _eval_set_for_align if isinstance(_eval_set_for_align, list) else [_eval_set_for_align]
        for _p in _pairs:
            if isinstance(_p, tuple) and len(_p) == 2 and isinstance(_p[0], pd.DataFrame):
                _val_df_from_eval = _p[0]
                break
    return _val_df_from_eval


def _train_model_with_f_isinstance_train_df_pd(train_df, _val_df_from_eval, model_type_name, _eval_set_for_align, fit_params):
    """Block of _train_model_with_fallback_unguarded starting at ``if isinstance(train_df, pd.DataFrame) and _val_df_from_eval is not Non``."""
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
                fit_params["eval_set"] = [(_aligned_val, _p[1]) if isinstance(_p, tuple) and _p[0] is _val_df_from_eval else _p for _p in _eval_set_for_align]
            elif isinstance(_eval_set_for_align, tuple):
                fit_params["eval_set"] = (_aligned_val, _eval_set_for_align[1])
    return train_df


def _train_model_with_f_phase(model_type_name, train_df, _cb_pool, fit_params, model, train_target):
    """Block of _train_model_with_fallback_unguarded starting at ``with phase(``."""
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
            pin_hgb_categorical_features_for_polars(model, train_df)
            with _suppress_catboost_noise():
                model.fit(train_df, train_target, **fit_params)
    return fit_params, model


def _train_model_with_f_model_none(model, model_obj, verbose, best_iter):
    """Block of _train_model_with_fallback_unguarded starting at ``if model is not None:``."""
    if model is not None:
        try:
            best_iter = get_model_best_iter(model_obj)
            if best_iter and verbose:
                logger.info("es_best_iter: %d", best_iter)
        except (AttributeError, TypeError, ValueError):
            logger.warning("Could not get best iteration", exc_info=True)
    return best_iter


def _train_model_with_f_focused_unit_testing(model, best_iter, model_obj, model_type_name, train_df, train_target, fit_params):
    """Block of _train_model_with_fallback_unguarded starting at ``if model is not None and best_iter is not None:``."""
    if model is not None and best_iter is not None:
        _new_best_iter = _maybe_refit_on_best_iter_pathology(
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
    return best_iter
