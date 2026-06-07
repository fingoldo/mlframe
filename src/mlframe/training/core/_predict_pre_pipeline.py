"""Carved out of ``mlframe.training.core.predict``.

Bound back into the parent's namespace via ``from ._predict_<name> import X``
at the parent's module bottom so historical
``from mlframe.training.core.predict import predict_from_models``
resolves transparently.
"""
from __future__ import annotations

import glob
import logging
import os
import pickle as _pickle
from collections import defaultdict
from copy import deepcopy
from os.path import exists, join

from scipy import stats
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import polars as pl
from pyutilz.strings import slugify

from ..configs import TargetTypes
from ..extractors import FeaturesAndTargetsExtractor
from ..io import load_mlframe_model
from ..pipeline import prepare_df_for_catboost
from .._cb_pool import _predict_with_fallback
from ..utils import drop_columns_from_dataframe, get_pandas_view_of_polars_df
from .utils import (
    DEFAULT_PROBABILITY_THRESHOLD,
    _drop_cols_df,
    _setup_model_directories,
    _validate_input_columns_against_metadata,
    _validate_trusted_path,
)

logger = logging.getLogger("mlframe.training.core.predict")


def _apply_extensions_pipeline(df: Any, ext_pipeline: Any, verbose: int = 0):
    """Apply the persisted ``extensions_pipeline`` to a predict frame.

    Two persistence shapes are supported (see ``apply_preprocessing_extensions``):
      * ``dict`` of ``{column_name: TfidfVectorizer}`` -- TF-IDF-only stage; replace each declared column with its tfidf expansion using the train-time vocabulary.
      * sklearn ``Pipeline`` -- numeric stack (scaler / KBins / polynomial / RBF / PCA / etc); replace the input columns with the transformed frame and preserve fit-time column names via ``get_feature_names_out``.

    The frame is converted to pandas first (extensions pipelines are always sklearn-typed and fit on pandas). Returns the augmented pandas frame; non-finite input pipelines (``None``) are a no-op."""
    if ext_pipeline is None:
        return df
    if isinstance(df, pl.DataFrame):
        # Extensions pipelines are sklearn-typed and were fit on pandas -- materialise once for the transform.
        df = get_pandas_view_of_polars_df(df)
    # Bundle dispatch: PySR -> TFIDF -> sklearn, mirroring training order so
    # predict replay sees the same column-add sequence.
    try:
        from mlframe.training.pipeline import PreprocessingExtensionsBundle as _PEB
    except ImportError:
        _PEB = None
    if _PEB is not None and isinstance(ext_pipeline, _PEB):
        if ext_pipeline.pysr is not None:
            df = ext_pipeline.pysr.transform(df)
        if ext_pipeline.tfidf is not None:
            df = _apply_extensions_pipeline(df, ext_pipeline.tfidf, verbose=verbose)
        if ext_pipeline.sklearn_pipe is not None:
            df = _apply_extensions_pipeline(df, ext_pipeline.sklearn_pipe, verbose=verbose)
        return df
    if isinstance(ext_pipeline, dict):
        # TF-IDF-only shape -- replicate the training-time per-column replacement using the saved vectorizers.
        # Train hard-failed on missing tfidf-source cols (apply_preprocessing_extensions raises KeyError before
        # fit). Symmetric behaviour at predict: collect every missing source up front and raise once with the
        # full list. The previous WARN-and-skip silently produced models whose feature_names_in_ included tfidf
        # cols that never appeared at predict, then crashed at model.predict_proba with an opaque "missing
        # features" error. Loud-fail beats silent skip.
        _missing_tfidf = [_col for _col in ext_pipeline if _col not in df.columns]
        if _missing_tfidf:
            raise KeyError(
                f"[extensions_pipeline] tfidf source columns missing from predict frame: "
                f"{_missing_tfidf}. Training pipeline saw these cols at fit time; predict must too. "
                "Restore the upstream extraction or retrain on the current schema."
            )
        for _col, _vec in ext_pipeline.items():
            _text = df[_col].fillna("").astype(str).values
            _spmat = _vec.transform(_text)
            try:
                _n_feats = len(_vec.get_feature_names_out())
            except Exception:
                _n_feats = _spmat.shape[1]
            _new_cols = [f"{_col}__tfidf_{i}" for i in range(_n_feats)]
            from ..pipeline import sparse_df_from_spmatrix
            _new_df = sparse_df_from_spmatrix(_spmat, _new_cols, df.index)
            df = df.drop(columns=[_col]).join(_new_df)
        return df
    # Pipeline shape -- run .transform on the full frame; reuse fit-time output column names when available.
    # Subset the predict frame to the fit-time feature set before calling
    # .transform. The training-side apply_preprocessing_extensions drops
    # non-numeric cols (iter-43) and all-null cols (iter-44) BEFORE the
    # sklearn-bridge pipeline gets fit. The fitted pipeline's
    # feature_names_in_ records the surviving columns; at predict the raw
    # input frame still carries the pre-filter cat / text / all-null cols
    # that sklearn's strict feature-name check then rejects with "The
    # feature names should match those that were passed during fit",
    # logged + swallowed below -- the downstream model then sees the raw
    # frame minus the expected pca0..pcaN extension cols and crashes with
    # "expects features missing from input: ['pca0', ...]". Surfaced by
    # iter-49 300k seed=13 cb-regression where dim_reducer=PCA was on.
    _fit_feature_names = getattr(ext_pipeline, "feature_names_in_", None)
    if _fit_feature_names is not None and isinstance(df, pd.DataFrame):
        _fit_list = [str(_c) for _c in _fit_feature_names]
        _fit_set = set(_fit_list)
        _df_cols = [str(_c) for _c in df.columns]
        _present_set = _fit_set & set(_df_cols)
        if len(_present_set) < len(_fit_set):
            _missing = [c for c in _fit_list if c not in _present_set]
            logger.warning(
                "[extensions_pipeline] predict frame is missing %d "
                "fit-time column(s): %s. Pipeline.transform will likely "
                "fail; downstream model will see raw input. Restore "
                "upstream extraction or retrain on the current schema.",
                len(_missing), _missing[:10],
            )
        elif _df_cols != _fit_list:
            # All fit-time cols present but ORDER (and/or extras like
            # cat_low/cat_mid/x_null carried over from the pre-filter
            # frame) doesn't match -- sklearn's strict feature-name
            # check rejects both. Subset + reorder to the fit-time list.
            df = df.loc[:, _fit_list]
    try:
        _arr = ext_pipeline.transform(df)
    except Exception as _exc:
        # Hard-fail: returning the raw frame here would silently serve predictions on un-transformed columns (the model was trained on the post-extension feature space), producing wrong outputs with no error. Re-raise so the caller sees the failure instead of getting nonsense predictions. Soft-fail is gated behind the explicit MLFRAME_EXTENSIONS_SOFT_FAIL escape hatch, defaulting OFF.
        if os.environ.get("MLFRAME_EXTENSIONS_SOFT_FAIL", "").lower() in ("1", "true", "yes"):
            logger.error("[extensions_pipeline] transform failed: %s. MLFRAME_EXTENSIONS_SOFT_FAIL is set -> returning RAW frame; downstream model will see un-transformed columns and almost certainly produce nonsense.", _exc)
            return df
        raise RuntimeError(
            f"[extensions_pipeline] transform failed at predict time: {_exc}. "
            "The model was trained on the post-extension feature space; serving raw columns would produce wrong predictions. "
            "Restore the saved pipeline / retrain, or set MLFRAME_EXTENSIONS_SOFT_FAIL=1 to (unsafely) fall back to the raw frame."
        ) from _exc
    try:
        _names = list(ext_pipeline.get_feature_names_out())
    except (AttributeError, ValueError, NotImplementedError):
        _names = [f"ext_{i}" for i in range(_arr.shape[1])]
    try:
        import scipy.sparse as _sp
        _is_sparse = _sp.issparse(_arr)
    except ImportError:
        _is_sparse = False
    if _is_sparse:
        try:
            from ..pipeline import sparse_df_from_spmatrix
            return sparse_df_from_spmatrix(_arr, _names, df.index)
        except Exception:
            _arr = _arr.toarray()
    return pd.DataFrame(_arr, columns=_names, index=df.index)

def _try_predict_with_pp_fallback(
    fn,
    primary,
    fallback,
    *,
    model,
    expected_list,
    pandas_view_cache: dict,
    model_name: str,
    verbose: int = 0,
):
    """Call ``fn(primary)`` with two fallback paths for predict-time dtype mismatches.

    Wave 88 (2026-05-21): extracted from the 525-line predict_from_models per-model
    try-block body. Logic identical to the prior nested closure; named-arg surface
    now explicit, lifetime no longer scoped to a single for-loop iteration.

    Behaviour contract:

      1. Primary attempt routes through ``_predict_with_fallback`` when fn is
         ``predict`` / ``predict_proba`` so the CB val Pool cache + NaN guard +
         LGBM polars auto-convert kick in (was the SKEW-CB-POOL-CACHE site that
         re-paid 50-70s on every predict).
      2. On TypeError matching the "isnan on strings" / "'<' not supported
         between float and str" pattern, retry on ``fallback`` (the pre-pipeline
         frame) -- the model's internal OrdinalEncoder was fit on raw strings.
      3. On polars-input rejection (older CB / sklearn-wrapped XGB), retry via
         a cached pandas view (no re-conversion).
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .predict import _ensure_pandas_view
    _method = getattr(fn, "__name__", None)
    _initial_exc: Optional[BaseException] = None
    if _method in ("predict", "predict_proba"):
        try:
            return _predict_with_fallback(model, primary, method=_method, verbose=bool(verbose))
        except TypeError as _te:
            # Fall through to the in-line handler below so the encoder-mismatch retry path is preserved.
            _initial_exc = _te
        except (ValueError, AttributeError) as _ve:
            _initial_exc = _ve
    try:
        if _initial_exc is None:
            return fn(primary)
        raise _initial_exc
    except TypeError as _te:
        _msg = str(_te)
        # Same root cause -- HGB's internal OrdinalEncoder fit on raw strings
        # called at predict with numeric input. Two equivalent symptoms:
        #   1. "ufunc 'isnan' not supported" (isnan probe on str vocabulary)
        #   2. "'<' not supported between instances of 'float' and 'str'"
        # Both retry on the pre-main-pipeline frame where cat cols are strings.
        _is_encoder_mismatch = (
            ("isnan" in _msg and "supported" in _msg)
            or ("'<' not supported" in _msg and "'float'" in _msg and "'str'" in _msg)
        )
        if _is_encoder_mismatch and fallback is not None:
            logger.warning(
                "predict_from_models: %s.%s on post-pipeline "
                "frame tripped encoder dtype mismatch (%s); "
                "retrying on pre-pipeline frame (the model's "
                "internal OrdinalEncoder was fit on raw strings).",
                model_name, fn.__name__,
                _msg.splitlines()[0][:120],
            )
            _fb = fallback
            if expected_list is not None and hasattr(_fb, "columns"):
                _drop_fb = [c for c in _fb.columns if c not in expected_list]
                if _drop_fb:
                    if isinstance(_fb, pl.DataFrame):
                        _fb = _fb.drop(_drop_fb)
                    else:
                        _fb = _fb.drop(columns=_drop_fb)
                # Normalise column ORDER to fit-time expectation: LGB/XGB
                # silently accept all-required-cols + wrong-order then produce
                # nonsense predictions (feature_names_in_ consulted only at fit
                # time). Cheap view-only reorder.
                _fb_have = {str(c) for c in _fb.columns}
                _order = [c for c in expected_list if str(c) in _fb_have]
                if _order and list(map(str, _fb.columns)) != list(map(str, _order)):
                    if isinstance(_fb, pl.DataFrame):
                        _fb = _fb.select(_order)
                    else:
                        _fb = _fb.loc[:, _order]
            return fn(_fb)
        # Polars-input rejection (older CB / sklearn-wrapped XGB): retry via shared pandas view.
        if isinstance(primary, pl.DataFrame):
            logger.warning(
                "predict_from_models: %s.%s rejected polars frame (%s); retrying on cached pandas view.",
                model_name, fn.__name__, _msg.splitlines()[0][:160],
            )
            return fn(_ensure_pandas_view(primary, pandas_view_cache))
        raise
    except (ValueError, AttributeError) as _exc:
        # Polars-input rejection paths in older library versions raise ValueError /
        # AttributeError; fall back to the pandas view rather than dropping the model.
        if isinstance(primary, pl.DataFrame):
            logger.warning(
                "predict_from_models: %s.%s rejected polars frame (%s: %s); retrying on cached pandas view.",
                model_name, fn.__name__, type(_exc).__name__,
                str(_exc).splitlines()[0][:160],
            )
            return fn(_ensure_pandas_view(primary, pandas_view_cache))
        # Linear / sklearn feature-name mismatch (e.g. text_col reaches a Ridge
        # whose fit-time features were only the numeric ones). Subset the input
        # to ``expected_list`` (pre-computed from feature_names_in_ /
        # feature_names_) and retry. This is the iter#64 family fix: the linear
        # pre_pipeline carries an imputer + scaler whose feature_names_in_ is
        # the numeric-only subset, so we drop the text columns BEFORE calling
        # model.predict rather than letting sklearn's validate_data reject the
        # whole call.
        _msg = str(_exc)
        if (
            isinstance(_exc, ValueError)
            and expected_list is not None
            and hasattr(primary, "columns")
            and "feature names" in _msg.lower()
        ):
            _drop = [c for c in primary.columns if c not in expected_list]
            if _drop:
                logger.warning(
                    "predict_from_models: %s.%s tripped feature-names mismatch "
                    "(%s); dropping %d unseen column(s) %s and retrying.",
                    model_name, fn.__name__,
                    _msg.splitlines()[0][:120],
                    len(_drop), _drop[:5],
                )
                _have = {str(c) for c in primary.columns}
                _order = [c for c in expected_list if str(c) in _have]
                if isinstance(primary, pl.DataFrame):
                    _subset = primary.select(_order) if _order else primary.drop(_drop)
                else:
                    _subset = primary.loc[:, _order] if _order else primary.drop(columns=_drop)
                return fn(_subset)
        raise

def _apply_pre_pipeline_with_passthrough(
    input_for_model,
    *,
    model,
    model_obj,
    pipeline,
    df,
    df_pre_pipeline,
    metadata,
    model_name,
    verbose,
):
    """Apply ``model_obj.pre_pipeline.transform`` with text/embedding passthrough stash + feature-subset fallback.

    Wave 90 (2026-05-21): extracted from the predict.py:1372 mega-try body
    (was a 192-line nested ``if hasattr(model_obj, "pre_pipeline") ...``).
    Behaviour preserved bit-for-bit. Returns the (possibly transformed,
    possibly subsetted) input_for_model.

    No-op paths (returns input unchanged):
      - model_obj has no ``pre_pipeline`` attribute or it is None;
      - ``model_obj.pre_pipeline is pipeline`` (suite-level pipeline already applied);
      - ``check_is_fitted`` says the per-model pre_pipeline is unfitted
        (tree-model placeholder Pipelines carried for sklearn-compat).

    Active path:
      1. Stash text+embedding passthrough cols (from ``metadata``) by reading
         them off the broadest source frame -- prefer df_pre_pipeline, fall
         back to df, then input_for_model.
      2. Call ``pre_pipeline.transform(input_for_model)``.
      3. Re-attach stashed cols to the post-transform pandas frame so the
         downstream model sees the same frame width as at fit time.
      4. On transform failure (e.g. NotFittedError on a cloned-not-refit
         MRMR/RFECV/BorutaShap), fall back to subsetting input_for_model
         to the inner model's ``feature_names_in_`` / ``feature_names_``.
      5. Every narrow-except path WARN-logs which (col, error_type) pair
         failed so silent corruption is impossible.
    """
    if not hasattr(model_obj, "pre_pipeline") or model_obj.pre_pipeline is None:
        return input_for_model
    if model_obj.pre_pipeline == pipeline:
        return input_for_model

    from sklearn.utils.validation import check_is_fitted
    try:
        check_is_fitted(model_obj.pre_pipeline)
        _pp_fitted = True
    except Exception:
        _pp_fitted = False

    if not _pp_fitted:
        if verbose:
            logger.debug(
                "predict_from_models: %s has unfitted pre_pipeline; "
                "skipping .transform (main pipeline already prepared the frame)",
                model_name,
            )
        return input_for_model

    _meta_text = list(metadata.get("text_features") or [])
    _meta_emb = list(metadata.get("embedding_features") or [])
    _passthrough_cols = _meta_text + _meta_emb
    _stashed_passthrough: dict[str, Any] = {}
    if _passthrough_cols:
        _src_for_stash = None
        for _candidate in (df_pre_pipeline, df, input_for_model):
            if _candidate is None or not hasattr(_candidate, "columns"):
                continue
            if all(c in _candidate.columns for c in _passthrough_cols):
                _src_for_stash = _candidate
                break
        if _src_for_stash is not None:
            for _pc in _passthrough_cols:
                try:
                    if isinstance(_src_for_stash, pl.DataFrame):
                        _stashed_passthrough[_pc] = _src_for_stash.get_column(_pc).to_pandas()
                    else:
                        _stashed_passthrough[_pc] = _src_for_stash[_pc].reset_index(drop=True)
                except (KeyError, AttributeError, ValueError, TypeError) as _stash_err:
                    logger.warning(
                        "predict_from_models: %s passthrough col %r failed to stash "
                        "(%s: %s); downstream model will receive a frame missing this "
                        "column. Predictions on rows whose pre-fit pipeline depended on "
                        "it may be wrong.",
                        model_name, _pc, type(_stash_err).__name__, _stash_err,
                    )

    try:
        input_for_model = model_obj.pre_pipeline.transform(input_for_model)
        if _stashed_passthrough and isinstance(input_for_model, pd.DataFrame):
            for _pc, _vals in _stashed_passthrough.items():
                if _pc in input_for_model.columns:
                    continue
                try:
                    _vals_aligned = _vals
                    if hasattr(_vals_aligned, "reset_index"):
                        _vals_aligned = _vals_aligned.reset_index(drop=True)
                    input_for_model = input_for_model.reset_index(drop=True)
                    input_for_model[_pc] = _vals_aligned
                except (KeyError, ValueError, TypeError) as _reattach_err:
                    logger.warning(
                        "predict_from_models: %s passthrough col %r stashed but "
                        "failed to re-attach after pre_pipeline.transform "
                        "(%s: %s); model will see a frame missing this column.",
                        model_name, _pc, type(_reattach_err).__name__, _reattach_err,
                    )
    except Exception as _pp_exc:
        _inner_feat_names = getattr(model, "feature_names_in_", None)
        if _inner_feat_names is None:
            _inner_feat_names = getattr(model, "feature_names_", None)
        if _inner_feat_names is not None and hasattr(input_for_model, "columns"):
            try:
                _inner_list = [str(c) for c in _inner_feat_names]
                _have = {str(c) for c in input_for_model.columns}
                if all(c in _have for c in _inner_list):
                    if isinstance(input_for_model, pl.DataFrame):
                        input_for_model = input_for_model.select(_inner_list)
                    else:
                        input_for_model = input_for_model.loc[:, _inner_list]
            except (KeyError, ValueError, TypeError, AttributeError) as _subset_err:
                logger.warning(
                    "predict_from_models: %s feature_names_in_ subset failed "
                    "(%s: %s); falling through to model.predict on unsubset frame, "
                    "which will likely raise a shape mismatch downstream.",
                    model_name, type(_subset_err).__name__, _subset_err,
                )
        logger.warning(
            "predict_from_models: %s pre_pipeline.transform "
            "raised %s: %s. Skipping pre_pipeline (main "
            "pipeline already encoded cat_features); subsetted "
            "input to inner model's feature_names_in_ when "
            "available, then passing to model.predict.",
            model_name, type(_pp_exc).__name__,
            str(_pp_exc).splitlines()[0][:160],
        )

    return input_for_model
