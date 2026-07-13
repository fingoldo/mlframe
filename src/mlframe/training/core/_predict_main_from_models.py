"""``predict_from_models`` carved out of
``mlframe.training.core._predict_main`` for the 2026-05-22 sub-split that
brings _predict_main below 1k LOC.
"""
from __future__ import annotations

import logging

from scipy import stats
from typing import Any, Optional

import numpy as np
import pandas as pd
import polars as pl

from ..extractors import FeaturesAndTargetsExtractor
from .utils import (
    DEFAULT_PROBABILITY_THRESHOLD,
    get_decision_threshold,
    _drop_cols_df,
    _validate_input_columns_against_metadata,
)

logger = logging.getLogger("mlframe.training.core.predict")


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
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .predict import _apply_extensions_pipeline, _apply_pre_pipeline_with_passthrough, _coerce_cat_dtype_for_lgb_xgb, _combine_probs, _ensure_pandas_view, _is_polars_native_model, _is_post_hoc_calibrated_model, _replay_suite_datetime_decomposition, _resolve_chosen_ensemble_params, _resolve_chosen_flavour, _resolve_quantile_alphas, _run_batched, _try_predict_with_pp_fallback
    from .._classif_helpers import _canonical_predict_proba_shape
    from ..pipeline._categorical_composite_fe import replay_categorical_composite_fe
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

    results: dict[str, Any] = {
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
                _all_in_mem.extend(_entries)
        # Cross-target ensemble entries (``_CT_ENSEMBLE__*``) are not polars-native; their inclusion forces
        # the lazy conversion path. Same for any non-CB / non-XGB model in the suite.
        if _all_in_mem:
            _all_polars_native_inmem = all(_is_polars_native_model(_e) for _e in _all_in_mem)

    # Replay suite-owned datetime decomposition before validation/pipeline so the predict frame has the SAME derived columns as training; FTE already handled its own ts_field on the line above.
    df = _replay_suite_datetime_decomposition(df, metadata, verbose=verbose)
    df = replay_categorical_composite_fe(df, metadata, verbose=verbose)

    df = _validate_input_columns_against_metadata(df, metadata, verbose=bool(verbose))

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
    if extensions_pipeline is not None and isinstance(df, pd.DataFrame) and isinstance(df_pre_pipeline, pd.DataFrame):
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
                # ``pl.from_pandas(df[cols])`` pays a full pandas block consolidation copy through Arrow on the predict hot path. Building polars columns directly from per-column ``.to_numpy()`` views skips the pandas block manager round-trip; bench (100k x 30 mixed dtypes, 2026-05-24): 16.0ms -> 1.05ms (15x). ``rechunk=False`` on the from_pandas path showed no measurable gain in the same bench because the underlying copy is the consolidation, not the chunk merge.
                _ext_only_pl = pl.DataFrame({c: df[c].to_numpy() for c in _ext_new_cols})
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

                    # Wave 90 (2026-05-21): per-model pre_pipeline.transform
                    # (with text/embedding passthrough stashing + feature-
                    # subset fallback on NotFittedError) lifted to the
                    # module-level _apply_pre_pipeline_with_passthrough.
                    # Same 192 lines, one call -- behaviour preserved.
                    input_for_model = _apply_pre_pipeline_with_passthrough(
                        input_for_model,
                        model=model,
                        model_obj=model_obj,
                        pipeline=pipeline,
                        df=df,
                        df_pre_pipeline=df_pre_pipeline,
                        metadata=metadata,
                        model_name=model_name,
                        verbose=verbose,
                    )

                    # Wave 89 (2026-05-21): LGB + XGB cat dtype coercion lifted
                    # to module-level _coerce_cat_dtype_for_lgb_xgb. Same logic,
                    # one call instead of two adjacent ~40-line blocks.
                    # Thread persisted enum_domains (train-time Enum dictionaries) through so the polars XGB cat-cast lands on pl.Enum (per-Series, no global-string-cache widening). Out-of-domain values cast to null via strict=False (matches training treatment of truly-unseen test categories). Legacy bundles without enum_domains key fall back to pl.Categorical with WARN.
                    _enum_domains = metadata.get("enum_domains") if isinstance(metadata, dict) else None
                    input_for_model = _coerce_cat_dtype_for_lgb_xgb(
                        input_for_model, model=model, cat_features=_cat_features,
                        enum_domains=_enum_domains,
                    )

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
                    # Wave 88 (2026-05-21): the 90-line nested _try_predict closure
                    # was extracted to the module-level _try_predict_with_pp_fallback.
                    # Behaviour identical; the per-iteration def overhead is gone and
                    # the function is now unit-testable in isolation.
                    _exp_list_for_fallback = list(_expected) if _expected is not None else None

                    def _try_predict(fn, primary, fallback, model=model, _exp_list_for_fallback=_exp_list_for_fallback, model_name=model_name):
                        """Call ``fn`` (predict / predict_proba) on ``primary``, retrying on ``fallback`` when the post-pipeline frame trips a pre-pipeline-only estimator (e.g. an isnan-on-strings TypeError from a sklearn encoder)."""
                        return _try_predict_with_pp_fallback(
                            fn, primary, fallback,
                            model=model,
                            expected_list=_exp_list_for_fallback,
                            pandas_view_cache=_pandas_view_cache,
                            model_name=model_name,
                            verbose=verbose,
                        )

                    # CTE-RAW-X (2026-05-21): CompositeTargetEstimator's predict
                    # reads the base column directly from X to apply the transform's
                    # inverse (e.g. linear_residual: y = t_hat + alpha*base + beta).
                    # The alpha/beta were fit on the RAW base column at discovery
                    # time, but the default predict path passes the
                    # pre-pipeline-scaled X (input_for_model) where the base column
                    # is z-scored. Result: alpha*base_scaled ~ alpha*1 instead of
                    # alpha*base_raw ~ alpha*~10000, so the inverse degenerates to
                    # y_hat ~ t_hat (predictions stay in residual scale: mean~0,
                    # std=residual_std). This branch hands the wrapper the RAW
                    # frame; the inner non-composite estimators stay on the
                    # post-pipeline path.
                    from ..composite import CompositeTargetEstimator as _CTE_cls
                    _primary_for_model = df_pre_pipeline if isinstance(model, _CTE_cls) else input_for_model
                    if return_probabilities and hasattr(model, "predict_proba"):
                        probs = _try_predict(model.predict_proba, _primary_for_model, df_pre_pipeline)
                        # Keep the raw per-model output (multilabel MultiOutputClassifier returns a
                        # list of (N, 2) per-label arrays consumers rely on); canonicalize to (N, K)
                        # for the ensemble + shape-based decision logic so the list path does not crash
                        # at probs.ndim and the multilabel ensemble combine actually runs.
                        results["probabilities"][model_name] = probs
                        _ml_classes = getattr(model_obj, "classes_", None)
                        if _ml_classes is None:
                            _estimators = getattr(model_obj, "estimators_", None)
                            if _estimators is not None:
                                _ml_classes = [getattr(_e, "classes_", None) for _e in _estimators]
                        probs = _canonical_predict_proba_shape(probs, classes_=_ml_classes)
                        all_probs.append(probs)
                        per_target_probs.setdefault((target_type, target_name), []).append(probs)
                        _is_cal = _is_post_hoc_calibrated_model(model_obj)
                        all_calib_flags.append(_is_cal)
                        per_target_calib_flags.setdefault((target_type, target_name), []).append(_is_cal)

                        # Binary threshold from val/OOF-tuned metadata (never test); 0.5 fallback.
                        _bin_thr = get_decision_threshold(metadata, f"{target_type}|{target_name}", DEFAULT_PROBABILITY_THRESHOLD)
                        if probs.ndim == 2:
                            if probs.shape[1] == 2:
                                preds = (probs[:, 1] >= _bin_thr).astype(int)
                            else:
                                # Wave 21 P2: nan-safe argmax (second predict
                                # entry point; symmetric to L964 fix).
                                from ...utils.nan_safe import argmax_classes_safe
                                preds = argmax_classes_safe(
                                    probs, context=f"predict_from_models.{model_name}",
                                )
                        else:
                            preds = (probs >= _bin_thr).astype(int)
                        results["predictions"][model_name] = preds
                        all_preds.append(preds)
                        per_target_preds.setdefault((target_type, target_name), []).append(preds)
                    else:
                        # CTE-RAW-X: same rationale as the probs path above.
                        preds = _try_predict(model.predict, _primary_for_model, df_pre_pipeline)
                        results["predictions"][model_name] = preds
                        all_preds.append(preds)
                        per_target_preds.setdefault((target_type, target_name), []).append(preds)

                    results["models_used"].append(model_name)

                except Exception as e:
                    # Wave 41 (2026-05-20): twin path at line 995 already uses exc_info=True;
                    # this site was the asymmetric one - lost the traceback for downstream
                    # ensemble-member triage. Mirror the twin.
                    logger.exception("Error predicting with model %s", model_name)
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
            _ens_params = _resolve_chosen_ensemble_params(metadata, _tt, _tname)
            _rrf_k_replay = int(_ens_params.get("rrf_k", 60))
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
                    _probs_list, _flavour, quantile_alphas=_q_alphas, rrf_k=_rrf_k_replay,
                    is_calibrated_per_model=per_target_calib_flags.get((_tt, _tname)),
                    metadata=metadata,
                    target_label=f"{_tt}/{_tname}",
                    target_type=_tt,
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
            _ens_thr = get_decision_threshold(metadata, f"{_tt}|{_tname}", DEFAULT_PROBABILITY_THRESHOLD)
            if _combined.ndim == 2:
                if _combined.shape[1] == 2:
                    _t_preds = (_combined[:, 1] >= _ens_thr).astype(int)
                else:
                    # NaN-safe argmax: _combine_probs can emit NaN rows; plain
                    # np.argmax routes them silently to class 0.
                    from ...utils.nan_safe import argmax_classes_safe
                    _t_preds = argmax_classes_safe(
                        _combined, context=f"predict_from_models.per_target.{_key}",
                    )
            else:
                _t_preds = (_combined >= _ens_thr).astype(int)
            results["per_target_predictions"][_key] = _t_preds

        # Suite-wide ensemble: resolve a single flavour when one was chosen across the suite, else arithmetic mean.
        _suite_flavour = _resolve_chosen_flavour(metadata)
        # Pass the target_type only when the suite is homogeneously multiclass so the simplex renorm in
        # _combine_probs fires for the multiclass-only case without touching mixed / binary / multilabel suites.
        _suite_tts = {tt for (tt, _tn) in per_target_probs.keys()}
        _suite_tt = next(iter(_suite_tts)) if len(_suite_tts) == 1 else None
        avg_probs = _combine_probs(
            all_probs, _suite_flavour,
            is_calibrated_per_model=all_calib_flags or None,
            metadata=metadata,
            target_label="suite",
            target_type=_suite_tt,
        )
        results["ensemble_probabilities"] = avg_probs
        if isinstance(results.get("probabilities"), dict):
            results["probabilities"]["ensemble"] = avg_probs

        if avg_probs.ndim == 2:
            if avg_probs.shape[1] == 2:
                ensemble_preds = (avg_probs[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
            else:
                # NaN-safe argmax for the suite-wide ensemble row.
                from ...utils.nan_safe import argmax_classes_safe
                ensemble_preds = argmax_classes_safe(
                    avg_probs, context="predict_from_models.suite_ensemble",
                )
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
            from mlframe.models.ensembling import combine_float_predictions
            from ._predict_main_suite import _resolve_float_ensemble_flavour
            results["ensemble_predictions"] = combine_float_predictions(
                _stacked, flavour=_resolve_float_ensemble_flavour(metadata),
            )
        else:
            ensemble_preds, _ = stats.mode(_stacked, axis=0)
            results["ensemble_predictions"] = ensemble_preds.flatten()

    elif len(all_preds) == 1:
        results["ensemble_predictions"] = all_preds[0]
        if all_probs:
            results["ensemble_probabilities"] = all_probs[0]

    if verbose:
        logger.info("Generated predictions for %d models", len(results["predictions"]))

    # Escalate all-models-failed to a single aggregated RuntimeError.
    # Previously every per-model failure was logged and swallowed, leaving
    # the caller to detect "empty results" by hand. This hid the root
    # cause (the underlying per-model exception) behind a non-actionable
    # empty dict -- iter-45 500k cb-regression saw the predict swallow
    # the original error and the harness only got "per_target_probs
    # keys: []" with no clue what crashed.
    if _models_attempted > 0 and not results["predictions"] and not results["probabilities"] and _predict_errors:
        _summary = "; ".join(f"{_mn}: {_err}" for _mn, _err in _predict_errors[:5])
        if len(_predict_errors) > 5:
            _summary += f"; ... (+{len(_predict_errors) - 5} more)"
        raise RuntimeError(
            f"predict_from_models: all {_models_attempted} supplied "
            f"model(s) failed at predict; producing no predictions or "
            f"probabilities. Per-model errors: {_summary}"
        )

    return results
