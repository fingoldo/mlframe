"""Prediction + load entry points for mlframe trained suites."""
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

logger = logging.getLogger(__name__)


def _polars_native_class_names() -> tuple[str, ...]:
    """Bare class-name allowlist for the polars fastpath: CatBoost and XGBoost sklearn-API estimators (and the
    DMatrix-reuse shim subclasses).

    Returns bare names (not class objects) so we don't pay the import cost when the user has neither library installed and so a class-name match still works across the entire MRO -- the XGB shims (``XGBRegressorWithDMatrixReuse``, ``XGBClassifierWithDMatrixReuse``) inherit from XGBRegressor / XGBClassifier so the MRO check below catches them, but listing the shim names too keeps the heuristic explicit."""
    return (
        "CatBoostClassifier", "CatBoostRegressor", "CatBoostRanker",
        "XGBClassifier", "XGBRegressor", "XGBRanker",
        "XGBClassifierWithDMatrixReuse", "XGBRegressorWithDMatrixReuse",
    )


def _is_polars_native_model(model_obj: Any) -> bool:
    """Return True if ``model_obj.model`` is a CB or XGB sklearn-API class that accepts a polars DataFrame directly in ``predict`` / ``predict_proba``. MRO-aware so DMatrix-reuse shims inheriting from XGBRegressor / XGBClassifier are recognised even though their bare class names differ."""
    if model_obj is None:
        return False
    _m = getattr(model_obj, "model", model_obj)
    if _m is None:
        return False
    _allowed = _polars_native_class_names()
    return any(cls.__name__ in _allowed for cls in type(_m).__mro__)


def _ensure_pandas_view(df: Any, view_cache: dict) -> Any:
    """Return a pandas view of ``df``. If ``df`` is already pandas, returned as-is. If polars, the converted view is cached in ``view_cache`` keyed by ``id(df)`` so repeated calls on the same source frame within one predict call pay one conversion."""
    if not isinstance(df, pl.DataFrame):
        return df
    _src_id = id(df)
    _view = view_cache.get(_src_id)
    if _view is None:
        _view = get_pandas_view_of_polars_df(df)
        view_cache[_src_id] = _view
    return _view


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
            _new_df = pd.DataFrame.sparse.from_spmatrix(_spmat, columns=_new_cols, index=df.index)
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
        logger.error("[extensions_pipeline] transform failed: %s. Predict frame returned unchanged; downstream model will see RAW columns and almost certainly produce nonsense -- retrain or restore the saved pipeline.", _exc)
        return df
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
            return pd.DataFrame.sparse.from_spmatrix(_arr, columns=_names, index=df.index)
        except Exception:
            _arr = _arr.toarray()
    return pd.DataFrame(_arr, columns=_names, index=df.index)


def _load_ct_ensemble_entries(models_path: str, slug_to_original_target_type: dict, slug_to_original_target_name: dict, trusted_root: str | None = None) -> dict:
    """Scan ``models_path`` for cross-target ensemble dumps stored under ``<target_type_slug>/_CT_ENSEMBLE__<original_target>/CT_ENSEMBLE.dump`` and return a nested dict keyed ``{target_type: {_CT_ENSEMBLE__<orig>: [entry]}}`` so callers can merge it into the loaded ``models`` structure."""
    out: dict = defaultdict(lambda: defaultdict(list))
    _ct_files = glob.glob(join(models_path, "**", "_CT_ENSEMBLE__*", "*.dump"), recursive=True)
    for _f in _ct_files:
        _validate_trusted_path(_f, trusted_root or os.path.abspath(models_path))
        _rel = os.path.relpath(_f, models_path)
        _parts = _rel.split(os.sep)
        if len(_parts) < 3:
            continue
        _tt_slug = _parts[0]
        _tname_slug = _parts[1]
        if not _tname_slug.startswith("_CT_ENSEMBLE__"):
            continue
        _tt = slug_to_original_target_type.get(_tt_slug, _tt_slug)
        # The _CT_ENSEMBLE__<orig> directory name is the in-memory dict key, so reuse it verbatim (no slugify round-trip).
        _ct_key = _tname_slug
        _entry = load_mlframe_model(_f)
        if _entry is not None:
            out[_tt][_ct_key].append(_entry)
    return out


def _combine_probs(
    all_probs: list,
    flavour: str | None,
    *,
    quantile_alphas: Sequence[float] | None = None,
    quantile_mode: str = "sort",
    rrf_k: int = 60,
    ensure_prob_limits: bool = True,
    is_calibrated_per_model: Sequence[bool] | None = None,
    metadata: dict | None = None,
    target_label: str | None = None,
) -> np.ndarray:
    """Combine per-model prediction probabilities using the training-time-selected ``flavour``.

    Delegates to the canonical ``combine_probs`` helper in ``mlframe.models.ensembling`` to guarantee
    byte-identical (within fp64 tolerance) replay against the train-stamped output. ``rrf_k`` defaults
    to 60 but the caller (predict pipeline) should plumb the persisted ``metadata["rrf_k"]`` /
    ``metadata["ensembles_chosen_params"]`` so a non-default-k train run replays correctly.

    When ``quantile_alphas`` is provided the aggregated output is treated as a quantile-regression matrix
    ``(N, K=len(alphas))`` and ``fix_quantile_crossing`` is applied after the ensemble step -- arithmetic /
    harmonic / quadratic aggregations can break per-row monotonicity even when each member's own quantiles
    are monotone. Without this the SKEW-QUANTILE-ENS bug surfaced.

    Arch-4: ``is_calibrated_per_model`` carries one bool per member of ``all_probs`` indicating whether
    that model's probabilities have been post-hoc calibrated. A mix (some True, some False) produces a
    ``logger.warning`` because blending calibrated with uncalibrated probs degrades calibration of the
    ensemble; the function does NOT refuse (user policy: WARN, NOT REFUSE). When ``metadata`` is
    supplied, ``metadata["ensembles_calibrated"]`` is stamped as a bool reflecting "all members
    calibrated" (True only when every flag is True; False otherwise).
    """
    from mlframe.models.ensembling import combine_probs as _shared_combine_probs

    if is_calibrated_per_model is not None:
        _flags = [bool(f) for f in is_calibrated_per_model]
        if _flags and any(_flags) and not all(_flags):
            _n_cal = sum(_flags)
            logger.warning(
                "[_combine_probs] mixing calibrated and uncalibrated probabilities for ensemble"
                "%s: %d/%d members are calibrated. Blending the two degrades the ensemble's "
                "calibration; the predict path still proceeds (Arch-4 WARN, not REFUSE) but the "
                "deployed reliability diagram will diverge from each member's individual one.",
                f" target={target_label!r}" if target_label else "",
                _n_cal, len(_flags),
            )
        if isinstance(metadata, dict) and _flags:
            metadata["ensembles_calibrated"] = bool(all(_flags))

    stacked = np.stack(all_probs)
    combined = _shared_combine_probs(
        stacked, flavour or "arithm",
        rrf_k=int(rrf_k),
        ensure_prob_limits=ensure_prob_limits,
    )

    if quantile_alphas is not None and combined.ndim == 2:
        # P2-6: when the aggregated matrix's K dim does not match the alpha tuple, log + skip
        # rather than silently letting an aligned-by-coincidence call apply the wrong fix.
        if combined.shape[1] == len(quantile_alphas):
            try:
                from ..quantile_postproc import fix_quantile_crossing
                combined = fix_quantile_crossing(combined, quantile_alphas, mode=quantile_mode)
            except Exception as _qe:
                logger.warning("[_combine_probs] fix_quantile_crossing(post-aggregation) failed: %s", _qe)
        else:
            logger.warning(
                "[_combine_probs] quantile_alphas has %d entries but aggregated matrix has %d cols; "
                "skipping fix_quantile_crossing to avoid mis-aligned column rewrites.",
                len(quantile_alphas), combined.shape[1],
            )
    return combined


def _is_post_hoc_calibrated_model(model_obj: Any) -> bool:
    """Detect whether ``model_obj`` is a post-hoc calibrated wrapper.

    Class-name matching keeps the heavy training-module imports out of the predict path. Recognises
    the two wrappers stamped by mlframe.training: ``_PostHocCalibratedModel`` (single-output) and
    ``_PostHocMultiCalibratedModel`` (multi-output). Anything else is treated as uncalibrated for
    Arch-4 mix detection.
    """
    if model_obj is None:
        return False
    try:
        _name = type(model_obj).__name__
    except Exception:
        return False
    return _name in ("_PostHocCalibratedModel", "_PostHocMultiCalibratedModel")


def _resolve_quantile_alphas(metadata: dict, target_type: Any, target_name: Any, model_obj: Any = None) -> Sequence[float] | None:
    """Resolve quantile alphas for a (target_type, target_name) pair.

    Tries metadata-stored keys first; then falls back to model-object introspection
    (CatBoost MultiQuantile / XGB ``quantile_alpha`` / sklearn ``QuantileRegressor.alphas_``).
    Returns ``None`` when the target type isn't quantile or alphas can't be recovered.
    """
    if target_type not in ("quantile_regression", "regression_quantile") and str(target_type) != "quantile_regression":
        return None
    # Metadata-stored alphas: support a few shapes that have been used historically.
    if isinstance(metadata, dict):
        _qa = metadata.get("quantile_alphas")
        if isinstance(_qa, dict):
            _by_tt = _qa.get(target_type) or _qa.get(str(target_type))
            if isinstance(_by_tt, dict):
                _alphas = _by_tt.get(target_name) or _by_tt.get(str(target_name))
                if _alphas:
                    return list(_alphas)
            elif isinstance(_by_tt, (list, tuple)):
                return list(_by_tt)
        elif isinstance(_qa, (list, tuple)):
            return list(_qa)
    # Model-object introspection: a single member usually carries the alpha list (CB MultiQuantile, XGB quantile_alpha).
    if model_obj is not None:
        _m = getattr(model_obj, "model", model_obj)
        for _attr in ("quantile_alpha", "alphas_", "alphas", "_quantile_alphas"):
            _val = getattr(_m, _attr, None)
            if _val is not None and hasattr(_val, "__iter__"):
                try:
                    _alphas = [float(a) for a in _val]
                    if _alphas:
                        return _alphas
                except (TypeError, ValueError):
                    continue
    return None


def _resolve_chosen_ensemble_params(metadata: dict, target_type: Any = None, target_name: Any = None) -> dict:
    """Look up persisted ensemble replay params (rrf_k, etc.) for ``(target_type, target_name)``.

    C-P1-11: predict-side blend math must use the SAME rrf_k the train side recorded; pre-fix the
    predict path hard-coded k=60 which silently drifted when a user changed it at train time.
    ``metadata["ensembles_chosen_params"][tt][tname] = {"rrf_k": ...}`` is written by the per-target
    stamper in ``_phase_train_one_target``; this helper tolerates the absence (legacy models) and
    returns ``{}`` -- callers should treat that as "use defaults".
    """
    _ep = metadata.get("ensembles_chosen_params") if isinstance(metadata, dict) else None
    if not isinstance(_ep, dict):
        return {}
    if target_type is not None:
        _by_tt = _ep.get(target_type) or _ep.get(str(target_type))
        if isinstance(_by_tt, dict):
            if target_name is not None:
                _params = _by_tt.get(target_name) or _by_tt.get(str(target_name))
                if isinstance(_params, dict):
                    return _params
            # single-entry fallback
            if len(_by_tt) == 1:
                _only = next(iter(_by_tt.values()))
                if isinstance(_only, dict):
                    return _only
    # suite-wide fallback when only one (tt, tname) entry exists across the map
    _all_params: list[dict] = []
    for _v in _ep.values():
        if isinstance(_v, dict):
            for _vv in _v.values():
                if isinstance(_vv, dict):
                    _all_params.append(_vv)
    if len(_all_params) == 1:
        return _all_params[0]
    return {}


def _resolve_chosen_flavour(metadata: dict, target_type: Any = None, target_name: Any = None) -> str | None:
    """Look up the persisted chosen ensemble flavour for ``(target_type, target_name)``.

    Arch-3: ``metadata['ensembles_chosen']`` is sub-keyed per ensemble family --
    ``{"simple": {tt: {tname: flavour}}, "cross_target": {tt: {tname: flavour}}}``.
    ``_CT_ENSEMBLE__*``-prefixed target names dispatch to the ``cross_target`` bucket; everything
    else reads the ``simple`` bucket.
    """
    _ec = metadata.get("ensembles_chosen") if isinstance(metadata, dict) else None
    if _ec is None:
        return None
    if isinstance(_ec, str):
        return _ec
    if not isinstance(_ec, dict):
        return None
    # Pick bucket by target-name prefix. CT_ENSEMBLE keys live under "cross_target"; everything
    # else under "simple". Falls back to whichever bucket has the entry when only one is populated.
    _is_ct = isinstance(target_name, str) and target_name.startswith("_CT_ENSEMBLE__")
    _bucket = _ec.get("cross_target" if _is_ct else "simple")
    if not isinstance(_bucket, dict):
        _bucket = None
    if _bucket is not None and target_type is not None:
        _by_tt = _bucket.get(target_type) or _bucket.get(str(target_type))
        if isinstance(_by_tt, dict):
            if target_name is not None:
                _fl = _by_tt.get(target_name) or _by_tt.get(str(target_name))
                if _fl:
                    return _fl
            # Single-target fallback -- if there's only one entry, use it.
            if len(_by_tt) == 1:
                return next(iter(_by_tt.values()))
        elif isinstance(_by_tt, str):
            return _by_tt
    # No (tt, tname) match in the dispatched bucket -- if the whole map has a single flavour
    # everywhere across BOTH buckets, use it.
    _all = []
    for _fam in _ec.values():
        if isinstance(_fam, dict):
            for _v in _fam.values():
                if isinstance(_v, dict):
                    _all.extend(_v.values())
                elif isinstance(_v, str):
                    _all.append(_v)
        elif isinstance(_fam, str):
            _all.append(_fam)
    _all_set = {f for f in _all if f}
    if len(_all_set) == 1:
        # Set with cardinality 1 is deterministic, but sort-then-pick documents the intent and survives the
        # accidental len==N>1 case if a future caller widens the guard.
        return sorted(_all_set)[0]
    return None


def _replay_suite_datetime_decomposition(df, metadata, verbose: int = 0):
    """Replay the training-side ``create_date_features`` calls that the SUITE owns and drop any FTE-owned datetime source columns (FTE-handled cols are re-derived by the FTE's own ``transform`` on predict input, but FTE leaves the raw source via ``delete_original_cols=False`` -- the suite-side decomposition phase dropped those at training time, so predict must too or the raw datetime64 col will reach LGB / XGB / sklearn and fail dtype promotion).

    The suite persists ``metadata["datetime_methods"] = {src_col: {accessor: dtype_name}}``; this helper re-applies the same expansion to the predict-time frame so derived columns are byte-identical to training. When a source col is missing (already FTE-derived or absent), the corresponding entry is skipped.
    """
    from mlframe.feature_engineering.basic import create_date_features
    import numpy as _np

    _fte_emitted_map = metadata.get("ftextractor_emitted_columns") or {}
    if _fte_emitted_map and hasattr(df, "columns"):
        _drop = [c for c in _fte_emitted_map.keys() if c in df.columns]
        if _drop:
            if isinstance(df, pl.DataFrame):
                df = df.drop(_drop)
            else:
                df = df.drop(columns=_drop)

    _methods_map = metadata.get("datetime_methods") or {}
    if not _methods_map:
        return df

    _dtype_resolvers = {"int8": _np.int8, "int16": _np.int16, "int32": _np.int32, "int64": _np.int64,
                        "uint8": _np.uint8, "uint16": _np.uint16, "uint32": _np.uint32, "uint64": _np.uint64,
                        "float32": _np.float32, "float64": _np.float64}
    _present_sources = [c for c in _methods_map.keys() if c in df.columns]
    if not _present_sources:
        return df
    # Group sources by their methods dict so we batch-decompose cols sharing the same expansion (typical case: every suite-owned datetime col uses the same configured methods).
    _by_methods: Dict[Tuple[Tuple[str, str], ...], List[str]] = {}
    for _src in _present_sources:
        _m = _methods_map[_src]
        _key = tuple(sorted((str(k), str(v)) for k, v in _m.items()))
        _by_methods.setdefault(_key, []).append(_src)
    for _methods_key, _srcs in _by_methods.items():
        _resolved_methods = {}
        for _accessor, _dtype_name in _methods_key:
            _resolved_methods[_accessor] = _dtype_resolvers.get(_dtype_name, _np.int8)
        if verbose:
            logger.info("Replaying datetime decomposition (%s) on %d source col(s): %s",
                        "/".join(sorted(_resolved_methods.keys())), len(_srcs), _srcs)
        df = create_date_features(df, cols=_srcs, delete_original_cols=True, methods=_resolved_methods)
    return df


def _slice_frame(df: Any, start: int, length: int) -> Any:
    """Polars/pandas-agnostic slice. Returns rows [start, start+length). Out-of-range tail truncates safely."""
    if isinstance(df, pl.DataFrame):
        return df.slice(start, length)
    if isinstance(df, pd.DataFrame):
        return df.iloc[start: start + length]
    raise TypeError(f"_slice_frame: unsupported type {type(df).__name__}")


def _concat_probs_dicts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate per-batch probability dicts along axis 0 by model_name. Missing keys in any part are skipped
    (a model that crashed on one batch shouldn't poison the others; the caller's warn-and-continue handler
    already logged the failure)."""
    if not parts:
        return {}
    out: dict[str, np.ndarray] = {}
    keys: set[str] = set()
    for part in parts:
        keys.update(part.keys())
    for key in keys:
        chunks = [part[key] for part in parts if key in part and part[key] is not None]
        if chunks:
            out[key] = np.concatenate(chunks, axis=0)
    return out


def _run_batched(
    entry_fn: Callable,
    df: Any,
    predict_batch_rows: int,
    *args, **kwargs,
) -> dict[str, Any]:
    """Iterate ``entry_fn(_slice_frame(df, start, predict_batch_rows), *args, **kwargs)`` over batches and
    concatenate per-key arrays. Used by predict_mlframe_models_suite + predict_from_models when the caller sets
    ``predict_batch_rows`` to bound peak RSS on multi-million-row inference. Per-model probabilities /
    predictions / ensemble outputs are all concatenated row-wise; metadata + models_used + per_target dicts come
    from the FIRST batch (they're row-count-invariant)."""
    n = len(df)
    if n == 0:
        return entry_fn(df, *args, **kwargs)
    batch_outs: list[dict[str, Any]] = []
    _start = 0
    while _start < n:
        _length = min(predict_batch_rows, n - _start)
        _slice = _slice_frame(df, _start, _length)
        _out = entry_fn(_slice, *args, **kwargs)
        batch_outs.append(_out)
        _start += _length

    if len(batch_outs) == 1:
        return batch_outs[0]

    merged: dict[str, Any] = dict(batch_outs[0])  # carry metadata / models_used etc. from batch-0
    for _key in ("predictions", "probabilities", "per_target_probabilities", "per_target_predictions"):
        if _key in merged and isinstance(merged[_key], dict):
            merged[_key] = _concat_probs_dicts([b.get(_key, {}) or {} for b in batch_outs])
    for _key in ("ensemble_predictions", "ensemble_probabilities"):
        _parts = [b.get(_key) for b in batch_outs if b.get(_key) is not None]
        if _parts:
            try:
                merged[_key] = np.concatenate(_parts, axis=0)
            except ValueError:
                merged[_key] = _parts[0]
    # input_df concatenated row-wise so consumers reading the post-extensions frame from the result still see all rows.
    _input_parts = [b.get("input_df") for b in batch_outs if b.get("input_df") is not None]
    if _input_parts:
        try:
            if isinstance(_input_parts[0], pl.DataFrame):
                merged["input_df"] = pl.concat(_input_parts, how="vertical")
            else:
                merged["input_df"] = pd.concat(_input_parts, axis=0)
        except (TypeError, ValueError):
            merged["input_df"] = _input_parts[0]
    return merged


def predict_mlframe_models_suite(
    df: pl.DataFrame | pd.DataFrame,
    models_path: str,
    features_and_targets_extractor: FeaturesAndTargetsExtractor | None = None,
    model_names: list[str] | None = None,
    return_probabilities: bool = True,
    verbose: int = 1,
    trusted_root: str | None = None,
    predict_batch_rows: Optional[int] = None,
) -> dict[str, Any]:
    """
    Generate predictions using a trained mlframe models suite.

    Loads the trained suite from disk and applies all required transformations
    to raw input data before generating predictions.

    Args:
        df: Input DataFrame (raw data, same format as training input)
        models_path: Path to the models directory (e.g., "data/models/target_name/model_name")
        features_and_targets_extractor: Optional extractor to preprocess input (same as training)
        model_names: Optional list of specific model names to use (None = all models)
        return_probabilities: If True, return probabilities; if False, return class predictions
        verbose: Verbosity level

    Returns:
        Dict with:
            - "predictions": Dict[model_name, predictions array]
            - "probabilities": Dict[model_name, probabilities array] (if return_probabilities)
            - "ensemble_predictions": Combined ensemble predictions (if multiple models)
            - "metadata": Loaded metadata dict
    """
    # Validate inputs
    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError(f"df must be pandas or polars DataFrame, got {type(df).__name__}")
    if len(df) == 0:
        raise ValueError("df cannot be empty")
    if not isinstance(models_path, str) or not os.path.isdir(models_path):
        raise ValueError(f"models_path must be a valid directory, got: {models_path}")

    if predict_batch_rows is not None and predict_batch_rows > 0 and len(df) > predict_batch_rows:
        # Dispatch to the batched-runner; the batched-runner calls this same function recursively per slice
        # with predict_batch_rows=None so the legacy single-pass code path runs unchanged for each batch.
        return _run_batched(
            lambda _d: predict_mlframe_models_suite(
                _d, models_path,
                features_and_targets_extractor=features_and_targets_extractor,
                model_names=model_names,
                return_probabilities=return_probabilities,
                verbose=verbose,
                trusted_root=trusted_root,
                predict_batch_rows=None,
            ),
            df, predict_batch_rows,
        )

    results = {
        "predictions": {},
        "probabilities": {},
        "ensemble_predictions": None,
        "ensemble_probabilities": None,
        "metadata": None,
        "input_df": None,
    }

    # Prefer pickle-proto5 + zstd; fall back to legacy metadata.joblib for older saves.
    metadata_file_new = join(models_path, "metadata.pkl.zst")
    metadata_file_uncompressed = join(models_path, "metadata.pkl")
    metadata_file_legacy = join(models_path, "metadata.joblib")
    if exists(metadata_file_new):
        metadata_file = metadata_file_new
        loader_kind = "pkl.zst"
    elif exists(metadata_file_uncompressed):
        metadata_file = metadata_file_uncompressed
        loader_kind = "pkl"
    elif exists(metadata_file_legacy):
        metadata_file = metadata_file_legacy
        loader_kind = "joblib"
    else:
        raise FileNotFoundError(
            f"Metadata file not found in {models_path}; expected one of "
            f"metadata.pkl.zst, metadata.pkl, metadata.joblib"
        )

    if verbose:
        logger.info("Loading metadata from %s...", metadata_file)
    _root = trusted_root if trusted_root is not None else os.path.abspath(models_path)
    _validate_trusted_path(metadata_file, _root)
    if loader_kind == "pkl.zst":
        import pickle as _pickle
        import zstandard as _zstd
        _dctx = _zstd.ZstdDecompressor()
        with open(metadata_file, "rb") as _f:
            metadata = _pickle.loads(_dctx.decompress(_f.read()))
    elif loader_kind == "pkl":
        import pickle as _pickle
        with open(metadata_file, "rb") as _f:
            metadata = _pickle.load(_f)
    else:
        metadata = joblib.load(metadata_file)
    results["metadata"] = metadata

    pipeline = metadata.get("pipeline")
    extensions_pipeline = metadata.get("extensions_pipeline")
    metadata.get("columns", [])

    if verbose:
        logger.info("Preprocessing input data...")

    if features_and_targets_extractor is not None:
        df, _, _, _, _, _, columns_to_drop, _ = features_and_targets_extractor.transform(df)
        df = _drop_cols_df(df, columns_to_drop)

    # Polars fastpath: decide BEFORE the eager pandas materialisation. If every loaded model is CB / XGB sklearn-API
    # (polars-native) and the input is polars, keep the polars frame all the way through; non-native models pay a
    # lazy conversion via ``_pandas_view_cache`` so two non-native models on the same source polars df share one view.
    # Respect ``model_names`` filter before loading: the probe only needs to inspect models the user actually
    # requested. Loading every .dump in the directory wasted RSS for one-model-needed inference calls.
    _input_is_polars = isinstance(df, pl.DataFrame)
    _all_model_files = glob.glob(join(models_path, "**", "*.dump"), recursive=True)
    if model_names:
        _name_set = set(model_names)
        _model_files_for_native_probe = [
            _f for _f in _all_model_files
            if os.path.basename(_f).replace(".dump", "") in _name_set
        ]
        if not _model_files_for_native_probe:
            logger.warning(
                "predict_mlframe_models_suite: model_names=%r matched 0 of %d .dump files in %s; falling back "
                "to loading all models. Check for typos / slug mismatch.",
                model_names, len(_all_model_files), models_path,
            )
            _model_files_for_native_probe = _all_model_files
    else:
        _model_files_for_native_probe = _all_model_files
    _loaded_models_cache: dict[str, Any] = {}
    _all_polars_native = False
    if _input_is_polars and _model_files_for_native_probe:
        _probe_results = []
        for _f in _model_files_for_native_probe:
            _mo = load_mlframe_model(_f)
            _loaded_models_cache[_f] = _mo
            _probe_results.append(_is_polars_native_model(_mo))
        _all_polars_native = bool(_probe_results) and all(_probe_results)
    _pandas_view_cache: dict[int, pd.DataFrame] = {}

    if not _all_polars_native and isinstance(df, pl.DataFrame):
        df = get_pandas_view_of_polars_df(df)

    # Replay suite-owned datetime decomposition before validation/pipeline so the predict frame has the SAME derived columns as training; FTE already handled its own ts_field on the line above.
    df = _replay_suite_datetime_decomposition(df, metadata, verbose=verbose)

    df = _validate_input_columns_against_metadata(df, metadata, verbose=verbose)

    if pipeline is not None:
        if verbose:
            logger.info("Applying pipeline transformation...")
        df = pipeline.transform(df)

    # Extensions pipeline replay (PySR / TF-IDF / polynomial / scaler / KBins / RBF / PCA). MUST run AFTER the
    # main pipeline (same order as training in ``_phase_fit_pipeline`` -> ``apply_preprocessing_extensions``)
    # and BEFORE the per-model column subset, otherwise models trained with ``preprocessing_extensions`` see
    # raw columns at predict and produce garbage.
    if extensions_pipeline is not None:
        if verbose:
            logger.info("Applying extensions pipeline transformation...")
        df = _apply_extensions_pipeline(df, extensions_pipeline, verbose=verbose)

    # Storing the post-extensions frame in the returned dict pins a multi-GB frame in caller's reference graph
    # even when they don't read it. The legacy callers that DO read it relied on the key existing, so we keep
    # the slot but consider making it opt-out via the predict_batch_rows-style param if RSS pressure is
    # measured (deferred to a future PR; no measured speedup as long as caller drops the dict promptly).
    results["input_df"] = df

    if verbose:
        logger.info("Loading and running models...")

    model_files = _model_files_for_native_probe

    if not model_files:
        logger.warning(f"No model files found in {models_path}")
        return results

    all_probs = []
    all_preds = []
    # Per-target accumulator so Fix 3 can replay the chosen flavour separately for each (target_type, target_name).
    per_target_probs: dict[tuple[Any, Any], list[np.ndarray]] = {}
    per_target_preds: dict[tuple[Any, Any], list[np.ndarray]] = {}
    # Arch-4: per-member calibration flags so _combine_probs can WARN on mixed ensembles. One bool
    # per probability array, aligned with the prob accumulators above.
    all_calib_flags: list[bool] = []
    per_target_calib_flags: dict[tuple[Any, Any], list[bool]] = {}
    _slug_to_tt = metadata.get("slug_to_original_target_type", {}) or {}
    _slug_to_tn = metadata.get("slug_to_original_target_name", {}) or {}

    for model_file in model_files:
        model_name = os.path.basename(model_file).replace(".dump", "")

        if model_names and model_name not in model_names:
            continue

        # Recover (target_type, target_name) from the on-disk layout (mirrors load_mlframe_suite); used to key
        # per-target flavour replay below. Resolution to RAW target_type/target_name is the contract: keys leaking
        # back to slugs would diverge from predict_from_models (which iterates models[tt][tn] directly with the
        # raw tuple); we surface a WARN when the slug map is incomplete so the leak is visible.
        _rel = os.path.relpath(model_file, models_path)
        _parts = _rel.split(os.sep)
        if len(_parts) >= 3:
            _tt_slug, _tn_slug = _parts[0], _parts[1]
            _tt = _slug_to_tt.get(_tt_slug)
            _tn = _slug_to_tn.get(_tn_slug)
            if _tt is None:
                logger.warning(
                    "predict_mlframe_models_suite: slug_to_original_target_type missing entry for %r; "
                    "result keys for this model will use the slug verbatim, diverging from predict_from_models. "
                    "Re-save the suite with the current mlframe version to refresh the slug map.", _tt_slug,
                )
                _tt = _tt_slug
            if _tn is None:
                logger.warning(
                    "predict_mlframe_models_suite: slug_to_original_target_name missing entry for %r; "
                    "result keys for this model will use the slug verbatim, diverging from predict_from_models.", _tn_slug,
                )
                _tn = _tn_slug
        else:
            _tt, _tn = "unknown", "unknown"

        if verbose:
            logger.info("Loading model: %s", model_name)

        try:
            model_obj = _loaded_models_cache.get(model_file)
            if model_obj is None:
                model_obj = load_mlframe_model(model_file)
            if model_obj is None:
                logger.warning(f"Failed to load model: {model_file}")
                continue

            model = model_obj.model if hasattr(model_obj, "model") else model_obj

            input_for_model = df
            # Lazy polars->pandas only when this specific model is NOT polars-native (mirrors the training pattern
            # in ``_phase_train_one_target`` so two non-native models hit the cache after the first conversion).
            if isinstance(input_for_model, pl.DataFrame) and not _is_polars_native_model(model_obj):
                input_for_model = _ensure_pandas_view(input_for_model, _pandas_view_cache)
            if hasattr(model_obj, "pre_pipeline") and model_obj.pre_pipeline is not None:
                if model_obj.pre_pipeline != pipeline:
                    # Unified fitted-pipeline check (matches predict_from_models).
                    # Tree models with a polars-ds main pipeline often carry an
                    # UNFITTED placeholder sklearn Pipeline in pre_pipeline; calling
                    # transform on it raises NotFittedError.
                    from sklearn.utils.validation import check_is_fitted
                    from sklearn.exceptions import NotFittedError
                    try:
                        check_is_fitted(model_obj.pre_pipeline)
                        _pp_fitted = True
                    except Exception:
                        # ``Exception`` already subsumes NotFittedError; listing both is redundant. The
                        # broad catch is intentional - any check_is_fitted internal raise means "not safely
                        # fitted; skip transform".
                        _pp_fitted = False
                    if _pp_fitted:
                        try:
                            input_for_model = model_obj.pre_pipeline.transform(input_for_model)
                        except Exception as _pp_exc:
                            logger.warning(
                                "predict_mlframe_models_suite: %s pre_pipeline.transform "
                                "raised %s: %s. Skipping pre_pipeline.",
                                model_name, type(_pp_exc).__name__,
                                str(_pp_exc).splitlines()[0][:160],
                            )
                    elif verbose:
                        logger.debug(
                            "predict_mlframe_models_suite: %s has unfitted "
                            "pre_pipeline; skipping .transform.",
                            model_name,
                        )

            if return_probabilities and hasattr(model, "predict_proba"):
                # Route through _predict_with_fallback so the same predict-time guards used at training (CB val Pool
                # cache reuse, LGBM polars auto-convert, NaN safety net, CB polars dispatch-miss fallback) apply
                # uniformly at inference. Direct model.predict_proba bypassed the CB pool cache -- 50-70s/predict on
                # 7M rows.
                try:
                    probs = _predict_with_fallback(model, input_for_model, method="predict_proba", verbose=bool(verbose))
                except (TypeError, ValueError, AttributeError) as _polars_exc:
                    if isinstance(input_for_model, pl.DataFrame):
                        logger.warning("predict_proba on polars frame failed with %s: %s; retrying via pandas view.", type(_polars_exc).__name__, str(_polars_exc).splitlines()[0][:160])
                        input_for_model = _ensure_pandas_view(input_for_model, _pandas_view_cache)
                        probs = _predict_with_fallback(model, input_for_model, method="predict_proba", verbose=bool(verbose))
                    else:
                        raise
                results["probabilities"][model_name] = probs
                all_probs.append(probs)
                per_target_probs.setdefault((_tt, _tn), []).append(probs)
                _is_cal = _is_post_hoc_calibrated_model(model_obj)
                all_calib_flags.append(_is_cal)
                per_target_calib_flags.setdefault((_tt, _tn), []).append(_is_cal)

                # Shape-aware decision rule: 1-D = sigmoid threshold; (N,2) = threshold class-1;
                # (N,K>2) = argmax. Multilabel cannot be inferred from shape; caller must hold that contract.
                if probs.ndim == 2:
                    if probs.shape[1] == 2:
                        preds = (probs[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
                    else:
                        preds = np.argmax(probs, axis=1)
                else:
                    preds = (probs > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
                results["predictions"][model_name] = preds
                all_preds.append(preds)
                per_target_preds.setdefault((_tt, _tn), []).append(preds)
            else:
                try:
                    preds = _predict_with_fallback(model, input_for_model, method="predict", verbose=bool(verbose))
                except (TypeError, ValueError, AttributeError) as _polars_exc:
                    if isinstance(input_for_model, pl.DataFrame):
                        logger.warning("predict on polars frame failed with %s: %s; retrying via pandas view.", type(_polars_exc).__name__, str(_polars_exc).splitlines()[0][:160])
                        input_for_model = _ensure_pandas_view(input_for_model, _pandas_view_cache)
                        preds = _predict_with_fallback(model, input_for_model, method="predict", verbose=bool(verbose))
                    else:
                        raise
                results["predictions"][model_name] = preds
                all_preds.append(preds)
                per_target_preds.setdefault((_tt, _tn), []).append(preds)

        except KeyboardInterrupt:
            raise
        except (OSError, ValueError, RuntimeError) as e:
            # Surface the FULL traceback in addition to the one-line message; the prior log dropped the stack
            # trace and operators couldn't tell whether the failure was load-time (corrupt .dump on disk) or
            # predict-time (input schema mismatch / dispatch miss). exc_info=True keeps the warn-and-continue
            # semantics but adds the diagnostic.
            logger.error(
                "Error loading/predicting with model %s: %s", model_file, e,
                exc_info=True,
            )
            continue

    if len(all_probs) > 1:
        if verbose:
            logger.info("Computing ensemble predictions...")

        # Per-target chosen flavour replay (Fix 3). For each (tt, tname) that contributed >=1 model run we combine
        # using the flavour persisted at finalize time, exposing the result under ``per_target_probabilities``.
        # For quantile_regression targets we ALSO apply fix_quantile_crossing post-aggregation so consumers never
        # see crossings the ensemble step introduced.
        results.setdefault("per_target_probabilities", {})
        results.setdefault("per_target_predictions", {})
        for (_tt_k, _tn_k), _probs_list in per_target_probs.items():
            if not _probs_list:
                continue
            _flavour = _resolve_chosen_flavour(metadata, _tt_k, _tn_k)
            _ens_params = _resolve_chosen_ensemble_params(metadata, _tt_k, _tn_k)
            _rrf_k_replay = int(_ens_params.get("rrf_k", 60))
            _key = f"{_tt_k}_{_tn_k}"
            _q_alphas = _resolve_quantile_alphas(metadata, _tt_k, _tn_k)
            if len(_probs_list) > 1:
                _combined = _combine_probs(
                    _probs_list, _flavour, quantile_alphas=_q_alphas, rrf_k=_rrf_k_replay,
                    is_calibrated_per_model=per_target_calib_flags.get((_tt_k, _tn_k)),
                    metadata=metadata,
                    target_label=f"{_tt_k}/{_tn_k}",
                )
            else:
                _combined = _probs_list[0]
                if _q_alphas is not None and _combined.ndim == 2 and _combined.shape[1] == len(_q_alphas):
                    try:
                        from ..quantile_postproc import fix_quantile_crossing
                        _combined = fix_quantile_crossing(_combined, _q_alphas, mode="sort")
                    except Exception as _qe:
                        logger.warning("predict_mlframe_models_suite: fix_quantile_crossing failed: %s", _qe)
            results["per_target_probabilities"][_key] = _combined
            if _combined.ndim == 2:
                if _combined.shape[1] == 2:
                    _t_preds = (_combined[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
                else:
                    _t_preds = np.argmax(_combined, axis=1)
            else:
                _t_preds = (_combined > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
            results["per_target_predictions"][_key] = _t_preds

        # Suite-wide ensemble: resolve a single flavour when one was chosen across the suite, else arithmetic mean.
        _suite_flavour = _resolve_chosen_flavour(metadata)
        avg_probs = _combine_probs(
            all_probs, _suite_flavour,
            is_calibrated_per_model=all_calib_flags or None,
            metadata=metadata,
            target_label="suite",
        )
        results["ensemble_probabilities"] = avg_probs
        # Expose the ensemble inside the per-model probabilities dict under the
        # canonical "ensemble" key so consumers can iterate one dict.
        if isinstance(results.get("probabilities"), dict):
            results["probabilities"]["ensemble"] = avg_probs

        if avg_probs.ndim == 2:
            if avg_probs.shape[1] == 2:
                ensemble_preds = (avg_probs[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
            else:
                ensemble_preds = np.argmax(avg_probs, axis=1)
        else:
            ensemble_preds = (avg_probs > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
        results["ensemble_predictions"] = ensemble_preds
        if isinstance(results.get("predictions"), dict):
            results["predictions"]["ensemble"] = ensemble_preds

    elif len(all_preds) > 1:
        # Ensemble fallback when no probabilities exist. Dispatch by
        # prediction dtype: float -> arithmetic mean (regression /
        # quantile heads); int -> scipy.stats.mode majority voting
        # (classification predict() without predict_proba). Pre-fix the
        # branch unconditionally called stats.mode on the stacked preds,
        # which (a) returns statistically meaningless output for
        # continuous regression predictions (mode of 2 float arrays
        # picks the first by tie-break ~always) and (b) was 50-80x
        # slower than np.mean on 1M-row inputs (0.72s vs 0.01s --
        # surfaced by the 1M regression x lgb predict profile).
        _stacked = np.stack(all_preds)
        if np.issubdtype(_stacked.dtype, np.floating):
            results["ensemble_predictions"] = _stacked.mean(axis=0)
        else:
            ensemble_preds, _ = stats.mode(_stacked, axis=0)
            results["ensemble_predictions"] = ensemble_preds.flatten()

    elif len(all_preds) == 1:
        results["ensemble_predictions"] = all_preds[0]
        if all_probs:
            results["ensemble_probabilities"] = all_probs[0]

    if verbose:
        logger.info("Generated predictions for %d models", len(results['predictions']))

    return results


def predict_from_models(
    df: pl.DataFrame | pd.DataFrame,
    models: dict,
    metadata: dict,
    features_and_targets_extractor: FeaturesAndTargetsExtractor | None = None,
    return_probabilities: bool = True,
    verbose: int = 1,
    predict_batch_rows: Optional[int] = None,
) -> dict[str, Any]:
    """
    Generate predictions using in-memory models from train_mlframe_models_suite.

    This function works with models already in memory, avoiding disk I/O.
    Use this when you have the models dict returned by train_mlframe_models_suite.

    Args:
        df: Input DataFrame (raw data, same format as training input)
        models: Models dict returned by train_mlframe_models_suite.
            Structure: models[target_type][target_name] = [model_obj, ...]
        metadata: Metadata dict returned by train_mlframe_models_suite
        features_and_targets_extractor: Optional extractor to preprocess input (same as training)
        return_probabilities: If True, return probabilities; if False, return class predictions
        verbose: Verbosity level

    Returns:
        Dict with:
            - "predictions": Dict[model_name, predictions array]
            - "probabilities": Dict[model_name, probabilities array] (if return_probabilities)
            - "ensemble_predictions": Combined ensemble predictions (if multiple models)
            - "ensemble_probabilities": Averaged probabilities (if multiple models)
            - "models_used": List of model names that were used

    Example:
        ```python
        models, metadata = train_mlframe_models_suite(...)

        # Later, predict on new data
        results = predict_from_models(
            df=new_data,
            models=models,
            metadata=metadata,
            features_and_targets_extractor=ft_extractor,
        )
        print(results["ensemble_probabilities"])
        ```
    """
    # Validate inputs
    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError(f"df must be pandas or polars DataFrame, got {type(df).__name__}")
    if len(df) == 0:
        raise ValueError("df cannot be empty")

    if predict_batch_rows is not None and predict_batch_rows > 0 and len(df) > predict_batch_rows:
        return _run_batched(
            lambda _d: predict_from_models(
                _d, models, metadata,
                features_and_targets_extractor=features_and_targets_extractor,
                return_probabilities=return_probabilities,
                verbose=verbose,
                predict_batch_rows=None,
            ),
            df, predict_batch_rows,
        )

    results = {
        "predictions": {},
        "probabilities": {},
        "ensemble_predictions": None,
        "ensemble_probabilities": None,
        "models_used": [],
    }
    # Track per-model failures so we can raise a single aggregated
    # RuntimeError when EVERY supplied model failed at predict. Without
    # this, the per-model ``except Exception: continue`` swallow at the
    # bottom of the loop hides the root cause behind an empty
    # ``results["predictions"]`` dict -- surfaced by iter-45 500k
    # cb-regression run where the only model errored and the harness
    # only saw "predict_from_models returned empty predictions+
    # probabilities (per_target_probs keys: [])".
    _predict_errors: list[tuple[str, str]] = []
    _models_attempted = 0

    if verbose:
        logger.info("Preprocessing input data...")

    if features_and_targets_extractor is not None:
        df, _, _, _, _, _, columns_to_drop, _ = features_and_targets_extractor.transform(df)
        df = _drop_cols_df(df, columns_to_drop)

    # Polars fastpath probe (Fix 1). If every in-memory model is CB / XGB sklearn-API AND the input is polars,
    # keep polars all the way; non-native models on the same source frame share one cached pandas view.
    _input_is_polars = isinstance(df, pl.DataFrame)
    _all_polars_native_inmem = False
    _pandas_view_cache: dict[int, pd.DataFrame] = {}
    if _input_is_polars:
        _all_in_mem = []
        for _by_name in (models or {}).values():
            if not isinstance(_by_name, dict):
                continue
            for _entries in _by_name.values():
                if not isinstance(_entries, list):
                    continue
                for _e in _entries:
                    _all_in_mem.append(_e)
        # Cross-target ensemble entries (``_CT_ENSEMBLE__*``) are not polars-native; their inclusion forces
        # the lazy conversion path. Same for any non-CB / non-XGB model in the suite.
        if _all_in_mem:
            _all_polars_native_inmem = all(_is_polars_native_model(_e) for _e in _all_in_mem)

    # Replay suite-owned datetime decomposition before validation/pipeline so the predict frame has the SAME derived columns as training; FTE already handled its own ts_field on the line above.
    df = _replay_suite_datetime_decomposition(df, metadata, verbose=verbose)

    df = _validate_input_columns_against_metadata(df, metadata, verbose=verbose)

    # Preserve the pre-main-pipeline frame as a fallback for models whose
    # internal categorical handling (sklearn HGB's auto-detected
    # OrdinalEncoder, fit on string categories) crashes on the post-
    # pipeline encoded form. Surfaced by fuzz iter#80: lgb+hgb mixed on
    # Polars+cat - the main polars-ds pipeline encodes cat_low to Float64
    # for the LGB-side, but HGB's internal OrdinalEncoder built at fit
    # time tries to compare Float64 input against its string vocabulary
    # via ``xp.isnan(known_values)`` which trips on strings.
    df_pre_pipeline = df

    pipeline = metadata.get("pipeline")
    extensions_pipeline = metadata.get("extensions_pipeline")
    if pipeline is not None:
        if verbose:
            logger.info("Applying pipeline transformation...")
        # Polars-ds pipelines (saved when prefer_polarsds=True at fit
        # time) call ``.lazy()`` on the input -- they require a Polars
        # DataFrame. Sklearn pipelines accept pandas. Previously the
        # pandas conversion happened BEFORE pipeline.transform, which
        # crashed PdsPipeline with "AttributeError: 'DataFrame' object
        # has no attribute 'lazy'" on every Polars-input predict call.
        # Surfaced by fuzz iter#53 (binary lgb + Polars frame). Defer
        # the conversion until AFTER pipeline.transform so each pipeline
        # type sees the format it was fitted on.
        df = pipeline.transform(df)

    # Extensions pipeline replay (Fix 2). PySR / TF-IDF / polynomial / scaler / KBins / RBF / PCA stack, applied AFTER
    # the main pipeline (same order as training); without this models trained with ``preprocessing_extensions`` see
    # raw columns at predict and produce garbage.
    if extensions_pipeline is not None:
        if verbose:
            logger.info("Applying extensions pipeline transformation...")
        df = _apply_extensions_pipeline(df, extensions_pipeline, verbose=verbose)

    # Polars fastpath: keep polars when every model is CB / XGB; otherwise pay one shared pandas conversion now
    # so downstream models don't each pay their own. Extensions pipeline always returns pandas so by here the
    # "all-native" path is only live when extensions_pipeline was None.
    if isinstance(df, pl.DataFrame) and not _all_polars_native_inmem:
        df = _ensure_pandas_view(df, _pandas_view_cache)
    if isinstance(df_pre_pipeline, pl.DataFrame) and not _all_polars_native_inmem:
        df_pre_pipeline = _ensure_pandas_view(df_pre_pipeline, _pandas_view_cache)

    # Back-merge extensions output onto df_pre_pipeline so the raw pre-pipeline frame
    # carries the same union (raw cols + extensions cols) that ``_phase_helpers._phase_fit_pipeline``
    # creates at training. Polars-fastpath / pandas-bypass models (LGB strategy converts polars->pandas
    # internally without applying the main pipeline) are fit on this hybrid frame: model.feature_names_in_
    # therefore lists raw + extension cols, neither of which the post-main-pipeline ``df`` alone carries
    # (the main pipeline encoded/expanded the raw cols out; extensions then dropped the encoded form).
    # Without this merge the per-model column subset below cannot find the expected raw cols and raises
    # ``expects features missing from input``. Surfaced by fuzz iter#189 (binary x lgb x onehot x
    # dim_reducer=TruncatedSVD).
    if (
        extensions_pipeline is not None
        and isinstance(df, pd.DataFrame)
        and isinstance(df_pre_pipeline, pd.DataFrame)
    ):
        _ext_new_cols = [c for c in df.columns if c not in set(df_pre_pipeline.columns)]
        if _ext_new_cols:
            _ext_only = df[_ext_new_cols]
            # Align index defensively: extensions pipelines preserve the input index, but a
            # post-main-pipeline reset_index (sklearn ColumnTransformer occasionally drops it)
            # could surface non-matching indexes.
            if not df_pre_pipeline.index.equals(_ext_only.index):
                _ext_only = _ext_only.set_axis(df_pre_pipeline.index)
            df_pre_pipeline = pd.concat([df_pre_pipeline, _ext_only], axis=1)
    elif (
        # iter-157: polars-fastpath suites (``_all_polars_native_inmem=True``)
        # keep df_pre_pipeline as polars (line 996 skips the pandas
        # conversion). The pandas-only back-merge above skips, leaving
        # df_pre_pipeline as raw-cols-only polars. Models fit at train
        # time saw raw+extension cols via the polars-pre back-merge in
        # ``_phase_helpers``, so model.feature_names_in_ contains BOTH
        # raw and ext cols. The per-model column subset below then
        # raises ``expects features missing from input: [x0, x1, ...,
        # cat_mid]`` because raw cols are missing from df (post-ext)
        # AND ext cols are missing from df_pre_pipeline (raw-only
        # polars). Surfaced by iter-110 (polars + TruncatedSVD + MRMR
        # losing every raw col) and iter-79/105/118/126/146/147/155/156.
        extensions_pipeline is not None
        and isinstance(df, pd.DataFrame)
        and isinstance(df_pre_pipeline, pl.DataFrame)
    ):
        _ext_new_cols = [c for c in df.columns if c not in set(df_pre_pipeline.columns)]
        if _ext_new_cols and df.shape[0] == df_pre_pipeline.shape[0]:
            try:
                _ext_only_pl = pl.from_pandas(df[_ext_new_cols])
                df_pre_pipeline = df_pre_pipeline.hstack(_ext_only_pl)
            except Exception as _bm_err:
                logger.warning(
                    "[predict back-merge] polars hstack of extension cols "
                    "%s failed: %s. Models trained on the polars-pre + "
                    "back-merged frame will fall back to raw-only and "
                    "likely report missing features.",
                    _ext_new_cols[:5], _bm_err,
                )

    # Cat dtype coercion is PER-MODEL (in the loop below), not global.
    # Different model types need different dtypes for the same cat_low
    # column after the main pipeline ran:
    #   - LGB (no per-model pre_pipeline): needs pandas ``category`` so
    #     its predict-time auto-detection of categorical_feature matches
    #     the fit-time spec stored on the booster. Anything else (object
    #     or float64) -> "train and valid dataset categorical_feature do
    #     not match" (iter#55).
    #   - sklearn HGB (CatBoostEncoder pre_pipeline + native cat support)
    #     / linear-family (imputer + scaler pre_pipeline): need OBJECT
    #     or numeric. Both the pre_pipeline's encoder and HGB's own
    #     isnan-based input check reject categorical dtype outright
    #     (iter#80 lgb+hgb mixed mode + iter#80 hgb standalone).
    #   - CB: doesn't care - Pool built from any dtype with cat_features.
    # The cast happens in the per-model loop below.
    _cat_features = metadata.get("cat_features") or []

    if verbose:
        logger.info("Running predictions on in-memory models...")

    all_probs = []
    all_preds = []
    # Per-target accumulator so Fix 3 can replay the chosen ensemble flavour separately for each
    # (target_type, target_name) -- the suite-wide np.mean previously blended every model's prediction across
    # targets, which silently ignored the per-target flavour choice.
    per_target_probs: dict[tuple[Any, Any], list[np.ndarray]] = {}
    per_target_preds: dict[tuple[Any, Any], list[np.ndarray]] = {}
    # Arch-4: per-member calibration flags for mix detection in _combine_probs.
    all_calib_flags: list[bool] = []
    per_target_calib_flags: dict[tuple[Any, Any], list[bool]] = {}

    # Caches keyed by (id(df), id(_expected)) and (id(df), id(_expected)) -> precomputed drop / order lists.
    # Many models in a suite carry IDENTICAL feature_names_in_ (same family fit on same FE step); recomputing
    # the set-difference per model wasted N x len(cols) hash work. id-keyed because expected lists are stable
    # references on a model object across the lifetime of one predict call.
    _col_diff_cache: dict[tuple[int, int], dict[str, Any]] = {}

    for target_type, targets in models.items():
        for target_name, model_list in targets.items():
            for model_obj in model_list:
                if model_obj is None or not hasattr(model_obj, "model") or model_obj.model is None:
                    continue
                _models_attempted += 1

                model_name = f"{target_type}_{target_name}"
                if hasattr(model_obj, "pre_pipeline") and model_obj.pre_pipeline is not None:
                    pipeline_name = type(model_obj.pre_pipeline).__name__
                    model_name = f"{model_name}_{pipeline_name}"

                # Disambiguate when multiple models in this target share a name.
                base_name = model_name
                counter = 1
                while model_name in results["predictions"]:
                    model_name = f"{base_name}_{counter}"
                    counter += 1

                if verbose:
                    logger.info("Predicting with model: %s", model_name)

                try:
                    model = model_obj.model

                    input_for_model = df
                    # Lazy polars->pandas when this specific model is NOT polars-native; mirrors the training
                    # _phase_train_one_target pattern (cached view shared across models on the same source frame).
                    if isinstance(input_for_model, pl.DataFrame) and not _is_polars_native_model(model_obj):
                        input_for_model = _ensure_pandas_view(input_for_model, _pandas_view_cache)

                    # Subset to the per-model expected feature list BEFORE
                    # routing through pre_pipeline (sklearn pipelines for
                    # linear / ridge / sgd reject text/embedding columns at
                    # transform time with "Feature names unseen at fit
                    # time"). The shared metadata["columns"] keeps every
                    # column the suite saw at fit time (including text /
                    # embedding cols routed to CB-only text_features /
                    # embedding_features), but each fitted model exposes
                    # its own ``feature_names_in_`` (sklearn API used by
                    # XGB / LGB / linear) or ``feature_names_`` (CatBoost).
                    # Prefer the pre_pipeline's feature_names_in_ when a
                    # pre_pipeline is present (it sees raw input before
                    # the imputer / scaler reshape feature names); fall
                    # back to the model's own feature_names_in_ otherwise.
                    # Surfaced by fuzz iter#52 (xgb + text_col) and iter#64
                    # (linear + text_col -> pre_pipeline.transform crash).
                    _expected = None
                    if hasattr(model_obj, "pre_pipeline") and model_obj.pre_pipeline is not None:
                        _expected = getattr(model_obj.pre_pipeline, "feature_names_in_", None)
                    if _expected is None:
                        _expected = getattr(model, "feature_names_in_", None)
                    if _expected is None:
                        _expected = getattr(model, "feature_names_", None)
                    if _expected is not None and hasattr(input_for_model, "columns"):
                        # Cached per-(input_for_model, _expected) set-diff. Multiple models in one suite often
                        # carry identical feature_names_in_; this reuses the computed missing / drop lists.
                        _cache_key = (id(input_for_model), id(_expected))
                        # Normalise column name dtypes to plain ``str`` so numpy.str_
                        # (the dtype LGBMClassifier.feature_names_in_ carries when fit on a
                        # pandas DataFrame) compares equal to the Python str a polars/pandas
                        # DataFrame surfaces via ``.columns``. ``np.str_`` IS a ``str``
                        # subclass so this is mostly a defence-in-depth normalisation.
                        # Cache _expected_list per id(_expected) so the ``str(c) for c in _expected`` loop runs
                        # once per unique feature_names_in_ object rather than once per model. Many strategies
                        # share the same _expected list across multiple models in one suite (xgb shim + raw xgb
                        # both carry the same fit-time names); this re-uses the prior normalisation.
                        _cached_diff = _col_diff_cache.get(_cache_key)
                        if _cached_diff is not None:
                            _expected_list = _cached_diff["expected_list"]
                            _have = _cached_diff["have"]
                            _missing = list(_cached_diff["missing"])  # caller mutates so give a fresh list
                        else:
                            _expected_list = [str(c) for c in _expected]
                            _have = {str(c) for c in input_for_model.columns}
                            _missing = [c for c in _expected_list if c not in _have]
                            _col_diff_cache[_cache_key] = {
                                "expected_list": _expected_list,
                                "have": _have,
                                "missing": tuple(_missing),
                            }
                        if _missing:
                            # Models trained on the polars-native fastpath (LGB / CB / XGB on
                            # polars input with prefer_polarsds=True) carry feature_names_in_
                            # from the RAW pre-pipeline frame, not the post-pipeline /
                            # post-extensions frame. When the main pipeline or extensions stage
                            # changes column names (sklearn one-hot expansion, dim_reducer
                            # output like truncatedsvd0..N, TF-IDF), every expected raw column
                            # is "missing" from the post-everything ``df``. Fall back to
                            # df_pre_pipeline (the raw user frame) before raising. Surfaced by
                            # fuzz iter#189 (binary x lgb x cat_enc=onehot x
                            # dim_reducer=TruncatedSVD).
                            _fb = df_pre_pipeline
                            if _fb is not None and hasattr(_fb, "columns"):
                                _fb_have = {str(c) for c in _fb.columns}
                                _fb_still_missing = [c for c in _expected_list if c not in _fb_have]
                                if not _fb_still_missing:
                                    if isinstance(_fb, pl.DataFrame) and not _is_polars_native_model(model_obj):
                                        _fb = _ensure_pandas_view(_fb, _pandas_view_cache)
                                    if verbose:
                                        logger.info(
                                            "predict_from_models: %s post-pipeline df missing %d expected col(s); "
                                            "falling back to raw pre-pipeline frame which has all expected cols.",
                                            model_name, len(_missing),
                                        )
                                    input_for_model = _fb
                                    _have = {str(c) for c in input_for_model.columns}
                                    _missing = []
                        if _missing:
                            raise ValueError(
                                f"Model {model_name} expects features missing "
                                f"from input: {_missing}. Restore the upstream "
                                f"extraction or retrain on the current schema."
                            )
                        _drop_extra = [c for c in input_for_model.columns if c not in _expected_list]
                        if _drop_extra:
                            # Pandas accepts ``drop(columns=...)``; polars 1.x
                            # accepts ``drop(list)`` positionally and raises
                            # TypeError on ``columns=`` kwarg. Surfaced by
                            # fuzz iter#145 (xgb-only polars path keeps df
                            # as polars via the ``_all_polars_native`` gate).
                            if isinstance(input_for_model, pl.DataFrame):
                                input_for_model = input_for_model.drop(_drop_extra)
                            else:
                                input_for_model = input_for_model.drop(columns=_drop_extra)

                    # Per-model pre_pipeline (sklearn Pipeline of imputer +
                    # scaler for linear / ridge / sgd; identity for tree
                    # models that handle NaN natively). Apply AFTER the
                    # per-model column subset so text/embedding columns
                    # don't reach the pre_pipeline's input-feature checker.
                    if hasattr(model_obj, "pre_pipeline") and model_obj.pre_pipeline is not None:
                        if model_obj.pre_pipeline != pipeline:
                            # Tree models (hgb / lgb / xgb / cb) with a
                            # polars-ds main pipeline often carry an
                            # UNFITTED placeholder sklearn Pipeline in
                            # ``pre_pipeline`` (the strategy set it up
                            # for sklearn-compat introspection but never
                            # called ``.fit()`` because the main pipeline
                            # did all preprocessing). Calling .transform
                            # on it raises ``NotFittedError``. Skip the
                            # transform when the pipeline has no fitted
                            # steps; the main pipeline already prepared
                            # the frame. Surfaced by fuzz iter#79
                            # (regression x lgb,hgb x polars).
                            from sklearn.utils.validation import check_is_fitted
                            try:
                                check_is_fitted(model_obj.pre_pipeline)
                                _pp_fitted = True
                            except Exception:
                                # ``Exception`` subsumes NotFittedError; broad catch is intentional - any
                                # check_is_fitted internal raise means "not safely fitted; skip transform".
                                _pp_fitted = False
                            if _pp_fitted:
                                # Double-encoding guard. When the main polars-ds
                                # pipeline already encoded cat_features (cat_low
                                # -> Float64 etc.) AND the model strategy attached
                                # its own pre_pipeline with a category encoder
                                # (sklearn HGB carries category_encoders.
                                # CatBoostEncoder with requires_encoding=True),
                                # the encoder receives numeric input it cannot
                                # match against its string-fit vocabulary and
                                # raises "ufunc 'isnan' not supported" or
                                # similar dtype errors. Plus the iter#55 global
                                # cat-cast at the top of this function wraps
                                # the column in pandas Categorical, which the
                                # encoder also rejects. Skip with a WARN: the
                                # main pipeline already did the encoder's job;
                                # the model's raw .predict on the post-main-
                                # pipeline frame produces equivalent output.
                                # Surfaced by fuzz iter#80 (lgb+hgb on Polars+
                                # cat: hgb's CatBoostEncoder crashed).
                                #
                                # iter-59 (cb-only + MRMR + text_features):
                                # the saved pre_pipeline's MRMR selector drops
                                # non-numeric cols (text/embedding) at
                                # transform-time, but at fit time the trainer
                                # wrapped pre_pipeline.fit_transform with
                                # ``_passthrough_cols_fit_transform`` so text/
                                # embedding cols survived the selector and
                                # reached CB.fit. CB then stored the integer
                                # column index for ``text_features`` against
                                # the re-attached frame. Without the same
                                # re-attach at predict, CB sees a narrower
                                # frame and raises ``Invalid text_features[0]
                                # = N value: index must be < N``.
                                #
                                # The fix has to stash the passthrough col
                                # VALUES from the pre-subset frame (because
                                # iter-49's expected-cols subset above may
                                # have dropped them when pre_pipeline.
                                # feature_names_in_ doesn't include them),
                                # then re-attach AFTER pre_pipeline.transform.
                                _meta_text = list(metadata.get("text_features") or [])
                                _meta_emb = list(metadata.get("embedding_features") or [])
                                _passthrough_cols = (_meta_text + _meta_emb)
                                # Source frame for stashing: prefer the raw
                                # pre-pipeline frame (df_pre_pipeline) which
                                # carries the un-encoded text/embedding cols
                                # untouched by any prior column reduction.
                                _stashed_passthrough: dict[str, Any] = {}
                                if _passthrough_cols:
                                    _src_for_stash = None
                                    for _candidate in (df_pre_pipeline, df, input_for_model):
                                        if _candidate is None or not hasattr(_candidate, "columns"):
                                            continue
                                        _have_all = all(c in _candidate.columns for c in _passthrough_cols)
                                        if _have_all:
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
                                                # Narrow: only the failures that mean "col missing / wrong shape / wrong type".
                                                # Pre-fix swallowed bare Exception silently; the missing col then never
                                                # made it into _stashed_passthrough, so the re-attach loop at L1446 simply
                                                # skipped it, and the downstream model was fed a frame MISSING a passthrough
                                                # column it was TRAINED on -- wrong predictions on production traffic with
                                                # no log line. Log at WARNING so the operator sees which passthrough cols
                                                # failed to round-trip.
                                                logger.warning(
                                                    "predict_from_models: %s passthrough col %r failed to stash "
                                                    "(%s: %s); downstream model will receive a frame missing this "
                                                    "column. Predictions on rows whose pre-fit pipeline depended on "
                                                    "it may be wrong.",
                                                    model_name, _pc, type(_stash_err).__name__, _stash_err,
                                                )
                                try:
                                    input_for_model = model_obj.pre_pipeline.transform(input_for_model)
                                    # Re-attach stashed passthrough cols so the
                                    # downstream model sees the same frame
                                    # width as at fit time (iter-59).
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
                                                # Narrow: shape / dtype mismatch on the re-attach assignment. Pre-fix
                                                # silently swallowed; the col never made it back, model.predict ran on
                                                # an incomplete frame, predictions silently wrong. Log at WARNING so
                                                # operator can see WHICH cols failed to round-trip vs which were stashed.
                                                logger.warning(
                                                    "predict_from_models: %s passthrough col %r stashed but "
                                                    "failed to re-attach after pre_pipeline.transform "
                                                    "(%s: %s); model will see a frame missing this column.",
                                                    model_name, _pc, type(_reattach_err).__name__, _reattach_err,
                                                )
                                except Exception as _pp_exc:
                                    # When pre_pipeline.transform fails (most
                                    # commonly NotFittedError on a cloned-not-
                                    # refitted MRMR / RFECV / BorutaShap),
                                    # recover by subsetting ``input_for_model``
                                    # to the inner model's own feature names
                                    # (LightGBM / XGB / sklearn:
                                    # ``feature_names_in_``; CatBoost:
                                    # ``feature_names_``). The inner model was
                                    # fit on the SELECTOR-output column subset,
                                    # so passing the un-subset full feature
                                    # set raises "data (8) vs training (5)"
                                    # in LGB and a similar shape error in
                                    # CB / XGB. Pre-fix iter-59 / iter-301
                                    # path: predict skipped pre_pipeline and
                                    # blindly passed the full frame to the
                                    # inner model -> LightGBMError; with this
                                    # fallback subset the inner model sees
                                    # the expected K columns and predict
                                    # succeeds. Surfaced by fuzz iter-301
                                    # (lgb binary MRMR) / iter-326 (lgb+xgb
                                    # regression MRMR).
                                    _inner_feat_names = getattr(model, "feature_names_in_", None)
                                    if _inner_feat_names is None:
                                        _inner_feat_names = getattr(model, "feature_names_", None)
                                    if (
                                        _inner_feat_names is not None
                                        and hasattr(input_for_model, "columns")
                                    ):
                                        try:
                                            _inner_list = [str(c) for c in _inner_feat_names]
                                            _have = {str(c) for c in input_for_model.columns}
                                            if all(c in _have for c in _inner_list):
                                                if isinstance(input_for_model, pl.DataFrame):
                                                    input_for_model = input_for_model.select(_inner_list)
                                                else:
                                                    input_for_model = input_for_model.loc[:, _inner_list]
                                        except (KeyError, ValueError, TypeError, AttributeError) as _subset_err:
                                            # Narrow: subset failures (missing col, dtype mismatch, polars/pandas mismatch).
                                            # Pre-fix swallow let the fallback silently SKIP the inner-model feature
                                            # subset, then model.predict raised a cryptic shape error downstream.
                                            # Log so the cascade is debuggable instead of producing two unrelated-looking
                                            # errors.
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
                            elif verbose:
                                logger.debug(
                                    "predict_from_models: %s has unfitted "
                                    "pre_pipeline; skipping .transform (main "
                                    "pipeline already prepared the frame)",
                                    model_name,
                                )

                    # LGB-only cat dtype coercion. LightGBM auto-detects
                    # categorical_feature from input dtype at predict and
                    # compares against the fit-time spec; object / float64
                    # cat_low triggers "categorical_feature do not match".
                    # Cast to pandas ``category`` only for LGB so we don't
                    # break sklearn HGB / linear models that reject
                    # categorical dtype (iter#80). Casts ANY non-category
                    # dtype because LGB checks dtype, not category levels.
                    _model_module = type(model).__module__ or ""
                    _is_lgb = (
                        _model_module.startswith("lightgbm")
                        or _model_module.endswith("lgb_shim")
                        or "LGBM" in type(model).__name__
                    )
                    if _is_lgb and _cat_features and hasattr(input_for_model, "columns"):
                        # Collect only the cols that need a cat-cast, then build the cast Series objects via
                        # a single ``DataFrame.assign(**{col: cast_series})``. ``assign`` returns a new frame
                        # sharing memory for un-cast columns (BlockManager-level reuse on pandas >=2.0); the
                        # prior implementation's ``input_for_model.copy()`` allocated a fresh copy of EVERY
                        # column even when only 1-2 cat cols needed casting -- biggest single allocation on
                        # wide frames (8GB savings on 7M x 200 in prod).
                        _to_cast: dict[str, Any] = {}
                        for _cf in _cat_features:
                            if _cf in input_for_model.columns:
                                try:
                                    if input_for_model[_cf].dtype.name != "category":
                                        _to_cast[_cf] = input_for_model[_cf].astype("category")
                                except Exception as _exc:
                                    logger.debug(
                                        "predict_from_models: LGB cat-cast for %r failed "
                                        "(%s); leaving as-is",
                                        _cf, type(_exc).__name__,
                                    )
                        if _to_cast:
                            input_for_model = input_for_model.assign(**_to_cast)

                    # XGB cat dtype coercion (iter-101 family). XGB's
                    # QuantileDMatrix builder rejects object / pl.String /
                    # pl.LargeString cat cols with ``Invalid columns:
                    # cat_low: object`` (pandas path) or
                    # ``KeyError: DataType(large_string)`` (polars path).
                    # Cast to pandas ``category`` so the downstream
                    # ``enable_categorical=True``-fit XGB Booster accepts
                    # them. Pre-fix this surfaced on every XGB predict
                    # whose cat cols were stored as raw strings in the
                    # serving frame (iter-228 / iter-243 / iter-258 /
                    # iter-264 / iter-275 / iter-307 / iter-313 /
                    # iter-322 / iter-326).
                    _is_xgb = (
                        _model_module.startswith("xgboost")
                        or _model_module.endswith("xgb_shim")
                        or "XGB" in type(model).__name__
                    )
                    if _is_xgb and _cat_features and hasattr(input_for_model, "columns"):
                        if isinstance(input_for_model, pl.DataFrame):
                            _pl_cast_exprs = []
                            for _cf in _cat_features:
                                if _cf not in input_for_model.columns:
                                    continue
                                _dt = input_for_model.schema.get(_cf)
                                if _dt in (pl.String, pl.Utf8) or str(_dt).startswith("LargeString"):
                                    _pl_cast_exprs.append(pl.col(_cf).cast(pl.Categorical))
                            if _pl_cast_exprs:
                                try:
                                    input_for_model = input_for_model.with_columns(_pl_cast_exprs)
                                except Exception as _exc:
                                    logger.debug(
                                        "predict_from_models: XGB polars cat-cast failed (%s); "
                                        "leaving as-is",
                                        type(_exc).__name__,
                                    )
                        else:
                            _to_cast_xgb: dict[str, Any] = {}
                            for _cf in _cat_features:
                                if _cf in input_for_model.columns:
                                    try:
                                        _col_dtype = input_for_model[_cf].dtype
                                        if getattr(_col_dtype, "name", "") != "category":
                                            _to_cast_xgb[_cf] = input_for_model[_cf].astype("category")
                                    except Exception as _exc:
                                        logger.debug(
                                            "predict_from_models: XGB cat-cast for %r failed "
                                            "(%s); leaving as-is",
                                            _cf, type(_exc).__name__,
                                        )
                            if _to_cast_xgb:
                                input_for_model = input_for_model.assign(**_to_cast_xgb)

                    # Pre-main-pipeline fallback. Models with internal
                    # categorical handling fitted on string categories
                    # (sklearn HGB built an OrdinalEncoder over the raw
                    # cat_low strings via ``categorical_features='from_dtype'``)
                    # crash on the encoded post-pipeline frame: the encoder
                    # tries ``xp.isnan(known_values)`` on its string
                    # vocabulary and TypeErrors. Try post-pipeline first
                    # (correct for LGB / linear / etc.) and fall back to
                    # pre-pipeline on the specific isnan-on-strings TypeError.
                    # Surfaced by fuzz iter#80 (lgb+hgb on Polars+cat).
                    _exp_list_for_fallback = list(_expected) if _expected is not None else None

                    def _try_predict(fn, primary, fallback):
                        # Route the primary attempt through _predict_with_fallback so the CB val Pool cache /
                        # NaN guard / LGBM polars auto-convert kick in at predict (the direct fn(primary)
                        # bypass was the SKEW-CB-POOL-CACHE site that re-paid 50-70s on every predict).
                        _method = getattr(fn, "__name__", None)
                        if _method in ("predict", "predict_proba"):
                            try:
                                return _predict_with_fallback(model, primary, method=_method, verbose=bool(verbose))
                            except TypeError as _te:
                                # Fall through to the in-line handler below so the encoder-mismatch retry path is preserved.
                                _initial_exc: BaseException = _te
                            except (ValueError, AttributeError) as _ve:
                                _initial_exc = _ve
                            else:
                                _initial_exc = None  # unreachable; here for type-narrowing
                        else:
                            _initial_exc = None
                        try:
                            if _initial_exc is None:
                                return fn(primary)
                            raise _initial_exc
                        except TypeError as _te:
                            _msg = str(_te)
                            # Two related symptoms of the same root cause
                            # (HGB's internal OrdinalEncoder fit on raw
                            # strings, called at predict with numeric input):
                            #   1. ``ufunc 'isnan' not supported`` - the
                            #      encoder probes its string vocabulary
                            #      with isnan, which rejects strings.
                            #   2. ``'<' not supported between instances
                            #      of 'float' and 'str'`` - the encoder
                            #      tries to sort/compare the numeric input
                            #      against its string categories.
                            # Both indicate the same dtype mismatch and
                            # should retry on the pre-main-pipeline frame
                            # where cat columns are still strings.
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
                                if _exp_list_for_fallback is not None and hasattr(_fb, "columns"):
                                    _drop_fb = [c for c in _fb.columns if c not in _exp_list_for_fallback]
                                    if _drop_fb:
                                        # Polars-aware drop (iter#145 fix pattern)
                                        if isinstance(_fb, pl.DataFrame):
                                            _fb = _fb.drop(_drop_fb)
                                        else:
                                            _fb = _fb.drop(columns=_drop_fb)
                                    # Normalise column ORDER to match the model's fit-time expectation. LGB and
                                    # XGB silently accept a frame with all-required cols but wrong order, then
                                    # produce nonsense predictions because feature_names_in_ is consulted only at
                                    # fit time. Reordering to the fit-time list is cheap (view-only on pandas /
                                    # polars 1.x select) and the only safe way to avoid the silent miscompute.
                                    _fb_have = {str(c) for c in _fb.columns}
                                    _order = [c for c in _exp_list_for_fallback if str(c) in _fb_have]
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
                                return fn(_ensure_pandas_view(primary, _pandas_view_cache))
                            raise
                        except (ValueError, AttributeError) as _exc:
                            # Polars-input rejection paths in older library versions raise ValueError / AttributeError;
                            # fall back to the pandas view rather than dropping the model.
                            if isinstance(primary, pl.DataFrame):
                                logger.warning(
                                    "predict_from_models: %s.%s rejected polars frame (%s: %s); retrying on cached pandas view.",
                                    model_name, fn.__name__, type(_exc).__name__,
                                    str(_exc).splitlines()[0][:160],
                                )
                                return fn(_ensure_pandas_view(primary, _pandas_view_cache))
                            raise

                    if return_probabilities and hasattr(model, "predict_proba"):
                        probs = _try_predict(model.predict_proba, input_for_model, df_pre_pipeline)
                        results["probabilities"][model_name] = probs
                        all_probs.append(probs)
                        per_target_probs.setdefault((target_type, target_name), []).append(probs)
                        _is_cal = _is_post_hoc_calibrated_model(model_obj)
                        all_calib_flags.append(_is_cal)
                        per_target_calib_flags.setdefault((target_type, target_name), []).append(_is_cal)

                        if probs.ndim == 2:
                            if probs.shape[1] == 2:
                                preds = (probs[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
                            else:
                                preds = np.argmax(probs, axis=1)
                        else:
                            preds = (probs > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
                        results["predictions"][model_name] = preds
                        all_preds.append(preds)
                        per_target_preds.setdefault((target_type, target_name), []).append(preds)
                    else:
                        preds = _try_predict(model.predict, input_for_model, df_pre_pipeline)
                        results["predictions"][model_name] = preds
                        all_preds.append(preds)
                        per_target_preds.setdefault((target_type, target_name), []).append(preds)

                    results["models_used"].append(model_name)

                except Exception as e:
                    logger.error(f"Error predicting with model {model_name}: {e}")
                    _predict_errors.append((model_name, f"{type(e).__name__}: {e}"))
                    continue

    if len(all_probs) > 1:
        if verbose:
            logger.info("Computing ensemble predictions...")

        # Per-target chosen flavour replay (Fix 3). When the metadata records a flavour for any (tt, tname) that
        # contributed >=2 probability arrays we combine THAT target with the chosen flavour and merge results
        # into the per-target probabilities dict; the suite-wide "ensemble" stays as the arithmetic mean for
        # back-compat with consumers that read results["probabilities"]["ensemble"].
        results.setdefault("per_target_probabilities", {})
        results.setdefault("per_target_predictions", {})
        # In-memory equivalent of the suite-path resolution; quantile alphas can be discovered via metadata or by
        # introspecting the first member of models[tt][tname] (the alpha list lives on the inner estimator).
        for (_tt, _tname), _probs_list in per_target_probs.items():
            if not _probs_list:
                continue
            _flavour = _resolve_chosen_flavour(metadata, _tt, _tname)
            _key = f"{_tt}_{_tname}"
            _sample_model = None
            try:
                _members = (models or {}).get(_tt, {}).get(_tname, []) or []
                _sample_model = _members[0] if _members else None
            except (AttributeError, IndexError, TypeError):
                _sample_model = None
            _q_alphas = _resolve_quantile_alphas(metadata, _tt, _tname, _sample_model)
            if len(_probs_list) > 1:
                _combined = _combine_probs(
                    _probs_list, _flavour, quantile_alphas=_q_alphas,
                    is_calibrated_per_model=per_target_calib_flags.get((_tt, _tname)),
                    metadata=metadata,
                    target_label=f"{_tt}/{_tname}",
                )
            else:
                _combined = _probs_list[0]
                if _q_alphas is not None and _combined.ndim == 2 and _combined.shape[1] == len(_q_alphas):
                    try:
                        from ..quantile_postproc import fix_quantile_crossing
                        _combined = fix_quantile_crossing(_combined, _q_alphas, mode="sort")
                    except Exception as _qe:
                        logger.warning("predict_from_models: fix_quantile_crossing failed: %s", _qe)
            results["per_target_probabilities"][_key] = _combined
            if _combined.ndim == 2:
                if _combined.shape[1] == 2:
                    _t_preds = (_combined[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
                else:
                    _t_preds = np.argmax(_combined, axis=1)
            else:
                _t_preds = (_combined > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
            results["per_target_predictions"][_key] = _t_preds

        # Suite-wide ensemble: resolve a single flavour when one was chosen across the suite, else arithmetic mean.
        _suite_flavour = _resolve_chosen_flavour(metadata)
        avg_probs = _combine_probs(
            all_probs, _suite_flavour,
            is_calibrated_per_model=all_calib_flags or None,
            metadata=metadata,
            target_label="suite",
        )
        results["ensemble_probabilities"] = avg_probs
        if isinstance(results.get("probabilities"), dict):
            results["probabilities"]["ensemble"] = avg_probs

        if avg_probs.ndim == 2:
            if avg_probs.shape[1] == 2:
                ensemble_preds = (avg_probs[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
            else:
                ensemble_preds = np.argmax(avg_probs, axis=1)
        else:
            ensemble_preds = (avg_probs > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
        results["ensemble_predictions"] = ensemble_preds
        if isinstance(results.get("predictions"), dict):
            results["predictions"]["ensemble"] = ensemble_preds

    elif len(all_preds) > 1:
        # Dtype-aware ensemble fallback when no probabilities exist.
        # Float predictions (regression / quantile heads) -> arithmetic
        # mean; integer predictions (classification predict() without
        # predict_proba) -> scipy.stats.mode majority voting. Mirrors
        # the fix in ``predict_mlframe_models_suite`` -- this entry
        # point had the same regression-via-mode bug.
        _stacked = np.stack(all_preds)
        if np.issubdtype(_stacked.dtype, np.floating):
            results["ensemble_predictions"] = _stacked.mean(axis=0)
        else:
            ensemble_preds, _ = stats.mode(_stacked, axis=0)
            results["ensemble_predictions"] = ensemble_preds.flatten()

    elif len(all_preds) == 1:
        results["ensemble_predictions"] = all_preds[0]
        if all_probs:
            results["ensemble_probabilities"] = all_probs[0]

    if verbose:
        logger.info("Generated predictions for %d models", len(results['predictions']))

    # Escalate all-models-failed to a single aggregated RuntimeError.
    # Previously every per-model failure was logged and swallowed, leaving
    # the caller to detect "empty results" by hand. This hid the root
    # cause (the underlying per-model exception) behind a non-actionable
    # empty dict -- iter-45 500k cb-regression saw the predict swallow
    # the original error and the harness only got "per_target_probs
    # keys: []" with no clue what crashed.
    if (
        _models_attempted > 0
        and not results["predictions"]
        and not results["probabilities"]
        and _predict_errors
    ):
        _summary = "; ".join(
            f"{_mn}: {_err}" for _mn, _err in _predict_errors[:5]
        )
        if len(_predict_errors) > 5:
            _summary += f"; ... (+{len(_predict_errors) - 5} more)"
        raise RuntimeError(
            f"predict_from_models: all {_models_attempted} supplied "
            f"model(s) failed at predict; producing no predictions or "
            f"probabilities. Per-model errors: {_summary}"
        )

    return results


def load_mlframe_suite(models_path: str, trusted_root: str | None = None) -> tuple[dict, dict]:
    """
    Load a trained mlframe models suite from disk.

    Args:
        models_path: Path to the models directory (e.g., "data/models/target_name/model_name")

    Returns:
        Tuple of (models dict, metadata dict) in the same format as train_mlframe_models_suite:
        - models: Dict[target_type][target_name] = [model_obj, ...]
        - metadata: Dict with training configuration and artifacts
    """
    if not isinstance(models_path, str):
        raise TypeError(f"models_path must be string, got {type(models_path).__name__}")
    if not os.path.isdir(models_path):
        raise ValueError(f"models_path must be a valid directory, got: {models_path}")

    # Prefer pickle-proto5 + zstd; fall back to legacy metadata.joblib.
    metadata_file_new = join(models_path, "metadata.pkl.zst")
    metadata_file_uncompressed = join(models_path, "metadata.pkl")
    metadata_file_legacy = join(models_path, "metadata.joblib")
    if exists(metadata_file_new):
        metadata_file = metadata_file_new
        _kind = "pkl.zst"
    elif exists(metadata_file_uncompressed):
        metadata_file = metadata_file_uncompressed
        _kind = "pkl"
    elif exists(metadata_file_legacy):
        metadata_file = metadata_file_legacy
        _kind = "joblib"
    else:
        raise FileNotFoundError(
            f"Metadata file not found in {models_path}; expected one of "
            f"metadata.pkl.zst, metadata.pkl, metadata.joblib"
        )

    _root = trusted_root if trusted_root is not None else os.path.abspath(models_path)
    _validate_trusted_path(metadata_file, _root)
    if _kind == "pkl.zst":
        import pickle as _pickle
        import zstandard as _zstd
        _dctx = _zstd.ZstdDecompressor()
        with open(metadata_file, "rb") as _f:
            metadata = _pickle.loads(_dctx.decompress(_f.read()))
    elif _kind == "pkl":
        import pickle as _pickle
        with open(metadata_file, "rb") as _f:
            metadata = _pickle.load(_f)
    else:
        metadata = joblib.load(metadata_file)

    slug_to_original_target_type = metadata.get("slug_to_original_target_type", {})
    slug_to_original_target_name = metadata.get("slug_to_original_target_name", {})

    # Structure: models[target_type][target_name] = [model_obj, ...]
    # On-disk layout from _setup_model_directories: models_path/target_type/target_name/model.dump
    models = defaultdict(lambda: defaultdict(list))
    model_files = glob.glob(join(models_path, "**", "*.dump"), recursive=True)

    for model_file in model_files:
        rel_path = os.path.relpath(model_file, models_path)
        path_parts = rel_path.split(os.sep)

        if len(path_parts) >= 3:
            slugified_target_type = path_parts[0]
            slugified_target_name = path_parts[1]
            target_type = slug_to_original_target_type.get(slugified_target_type, slugified_target_type)
            target_name = slug_to_original_target_name.get(slugified_target_name, slugified_target_name)
        else:
            target_type = "unknown"
            target_name = "unknown"

        model_obj = load_mlframe_model(model_file)
        if model_obj is not None:
            models[target_type][target_name].append(model_obj)

    return dict(models), metadata


__all__ = [
    "predict_mlframe_models_suite",
    "predict_from_models",
    "load_mlframe_suite",
]
