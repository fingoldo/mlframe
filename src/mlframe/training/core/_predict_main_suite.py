"""``predict_mlframe_models_suite`` carved out of
``mlframe.training.core._predict_main`` for the 2026-05-22 sub-split that
brings _predict_main below 1k LOC.
"""
from __future__ import annotations

import glob
import logging
import os
from os.path import exists, join

from scipy import stats
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import polars as pl

from ..extractors import FeaturesAndTargetsExtractor
from ..io import load_mlframe_model
from ..cb import _predict_with_fallback
from ..utils import get_pandas_view_of_polars_df
from .utils import (
    DEFAULT_PROBABILITY_THRESHOLD,
    get_decision_threshold,
    _drop_cols_df,
    _validate_input_columns_against_metadata,
    _validate_trusted_path,
)

logger = logging.getLogger("mlframe.training.core.predict")


def _resolve_float_ensemble_flavour(metadata: Any) -> str:
    """Flavour for the float (regression / quantile) member aggregation. Defaults to ``"mean"`` (legacy raw
    average -- optimal when folds are clean). ``"robust"`` (MAD-gated mean) is available via a stamped
    metadata key for outlier-fold protection, but is NOT the default: at small K the 3.5-MAD gate over-fires
    on normal fold spread and costs ~6% RMSE in the clean regime, so robustness is opt-in per the bench."""
    if isinstance(metadata, dict):
        flavour = metadata.get("float_ensemble_flavour")
        if isinstance(flavour, str) and flavour:
            return flavour
    return "mean"


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
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .predict import _apply_extensions_pipeline, _combine_probs, _ensure_pandas_view, _is_polars_native_model, _is_post_hoc_calibrated_model, _replay_suite_datetime_decomposition, _resolve_chosen_ensemble_params, _resolve_chosen_flavour, _resolve_quantile_alphas, _run_batched, _validate_metadata_version_envelope
    # Validate inputs
    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError(f"df must be pandas or polars DataFrame, got {type(df).__name__}")
    if len(df) == 0:
        raise ValueError("df cannot be empty")
    if not isinstance(models_path, str):
        raise TypeError(f"models_path must be a str, got {type(models_path).__name__}")
    if not os.path.isdir(models_path):
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

    results: dict[str, Any] = {
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
        raise FileNotFoundError(f"Metadata file not found in {models_path}; expected one of " f"metadata.pkl.zst, metadata.pkl, metadata.joblib")

    if verbose:
        logger.info("Loading metadata from %s...", metadata_file)
    _root = trusted_root if trusted_root is not None else os.path.abspath(models_path)
    _validate_trusted_path(metadata_file, _root)
    if loader_kind == "pkl.zst":
        # ``pickle.loads`` on a zstd-decompressed in-memory buffer; the file path was
        # ``_validate_trusted_path``-checked above. We additionally verify the sha256 sidecar so a
        # tampered .pkl.zst is rejected before the loads(); legacy bundles without sidecar still
        # load through the trusted-path + version-envelope gates.
        from mlframe.utils.safe_pickle import verify_sidecar as _vsidecar
        import pickle as _pickle  # nosec B403 - pickle used only for trusted same-process/dev-local round-trips, see call sites in this file
        import zstandard as _zstd
        if not _vsidecar(metadata_file, allow_unverified=True):
            raise RuntimeError(f"predict_mlframe_models_suite: sha256 sidecar mismatch on {metadata_file!r}; refusing to load.")
        _dctx = _zstd.ZstdDecompressor()
        with open(metadata_file, "rb") as _f:
            metadata = _pickle.loads(_dctx.decompress(_f.read()))  # nosec B301 - BARE_PICKLE_OK: in-memory buffer, sidecar already verified above
    elif loader_kind == "pkl":
        from mlframe.utils.safe_pickle import safe_load as _sload
        metadata = _sload(metadata_file, allow_unverified=True)
    else:
        metadata = joblib.load(metadata_file)
    # Wave 19 P0 #2: validate the schema_version + composite_target_env_signature
    # fields that the WRITE side has populated since 2026-02 (see
    # _phase_config_setup.py:312 + _phase_helpers.py:253). The READ side never
    # checked them, so an artifact written by code path A could be silently
    # consumed by code path B that interprets the same field-set differently.
    # Validation is WARN-only on minor skew (lib versions, schema_version 1
    # vs 2) and HARD-FAIL on missing schema_version when the bundle claims
    # composite targets (those require schema_version >= 2 semantics).
    _validate_metadata_version_envelope(metadata, models_path)
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
        _model_files_for_native_probe = [_f for _f in _all_model_files if os.path.basename(_f).replace(".dump", "") in _name_set]
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
        for _model_file in _model_files_for_native_probe:
            _mo = load_mlframe_model(_model_file)
            _loaded_models_cache[_model_file] = _mo
            _probe_results.append(_is_polars_native_model(_mo))
        _all_polars_native = bool(_probe_results) and all(_probe_results)
    _pandas_view_cache: dict[int, pd.DataFrame] = {}

    if not _all_polars_native and isinstance(df, pl.DataFrame):
        df = get_pandas_view_of_polars_df(df)

    # Replay suite-owned datetime decomposition before validation/pipeline so the predict frame has the SAME derived columns as training; FTE already handled its own ts_field on the line above.
    df = _replay_suite_datetime_decomposition(df, metadata, verbose=verbose)

    df = _validate_input_columns_against_metadata(df, metadata, verbose=bool(verbose))

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
    # One disk-loaded model per target, so the ensemble replay can introspect the inner estimator for
    # quantile alphas (parity with the in-memory predict_from_models path) -- without it, disk-loaded
    # quantile bundles skipped fix_quantile_crossing and emitted crossed quantiles.
    per_target_sample_model: dict[tuple[Any, Any], Any] = {}
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
                                "predict_mlframe_models_suite: %s pre_pipeline.transform " "raised %s: %s. Skipping pre_pipeline.",
                                model_name,
                                type(_pp_exc).__name__,
                                str(_pp_exc).splitlines()[0][:160],
                            )
                    elif verbose:
                        logger.debug(
                            "predict_mlframe_models_suite: %s has unfitted " "pre_pipeline; skipping .transform.",
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
                per_target_sample_model.setdefault((_tt, _tn), model_obj)
                _is_cal = _is_post_hoc_calibrated_model(model_obj)
                all_calib_flags.append(_is_cal)
                per_target_calib_flags.setdefault((_tt, _tn), []).append(_is_cal)

                # Shape-aware decision rule: 1-D = sigmoid threshold; (N,2) = threshold class-1;
                # (N,K>2) = argmax. Multilabel cannot be inferred from shape; caller must hold that contract.
                # Binary threshold is the per-target tuned value stamped into metadata (val/OOF-tuned,
                # never test); falls back to 0.5 when no tuned threshold is present.
                _bin_thr = get_decision_threshold(metadata, f"{_tt}|{_tn}", DEFAULT_PROBABILITY_THRESHOLD)
                if probs.ndim == 2:
                    if probs.shape[1] == 2:
                        preds = (probs[:, 1] >= _bin_thr).astype(int)
                    else:
                        # Wave 21 P2: nan-safe argmax. Pre-fix np.argmax
                        # on a NaN-bearing proba row silently classified
                        # as class 0 -> confusion matrix + per-class
                        # P/R/F1 wrong with no upstream signal.
                        from ...utils.nan_safe import argmax_classes_safe
                        preds = argmax_classes_safe(
                            probs, context=f"predict.{model_name}",
                        )
                else:
                    preds = (probs >= _bin_thr).astype(int)
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
            _q_alphas = _resolve_quantile_alphas(metadata, _tt_k, _tn_k, per_target_sample_model.get((_tt_k, _tn_k)))
            if len(_probs_list) > 1:
                _combined = _combine_probs(
                    _probs_list, _flavour, quantile_alphas=_q_alphas, rrf_k=_rrf_k_replay,
                    is_calibrated_per_model=per_target_calib_flags.get((_tt_k, _tn_k)),
                    metadata=metadata,
                    target_label=f"{_tt_k}/{_tn_k}",
                    target_type=_tt_k,
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
            _ens_thr = get_decision_threshold(metadata, f"{_tt_k}|{_tn_k}", DEFAULT_PROBABILITY_THRESHOLD)
            if _combined.ndim == 2:
                if _combined.shape[1] == 2:
                    _t_preds = (_combined[:, 1] >= _ens_thr).astype(int)
                else:
                    # NaN-safe argmax: _combine_probs (RRF / geomean / harmonic) can emit
                    # NaN when a member row was NaN; plain np.argmax silently routes the
                    # row to class 0 and poisons the downstream confusion matrix.
                    from ...utils.nan_safe import argmax_classes_safe
                    _t_preds = argmax_classes_safe(
                        _combined, context=f"predict_mlframe_models_suite.per_target.{_key}",
                    )
            else:
                _t_preds = (_combined >= _ens_thr).astype(int)
            results["per_target_predictions"][_key] = _t_preds

        # Suite-wide ensemble: resolve a single flavour when one was chosen across the suite, else arithmetic mean.
        _suite_flavour = _resolve_chosen_flavour(metadata)
        # Renorm the suite simplex only when the suite is homogeneously multiclass (mixed / binary / multilabel
        # suites must not be renormalised -- see the gate rationale in _combine_probs).
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
        # Expose the ensemble inside the per-model probabilities dict under the
        # canonical "ensemble" key so consumers can iterate one dict.
        if isinstance(results.get("probabilities"), dict):
            results["probabilities"]["ensemble"] = avg_probs

        if avg_probs.ndim == 2:
            if avg_probs.shape[1] == 2:
                ensemble_preds = (avg_probs[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
            else:
                # NaN-safe argmax for the suite-wide ensemble row: same reasoning as the
                # per-target site above; plain np.argmax sent NaN rows to class 0.
                from ...utils.nan_safe import argmax_classes_safe
                ensemble_preds = argmax_classes_safe(
                    avg_probs, context="predict_mlframe_models_suite.suite_ensemble",
                )
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
            from mlframe.models.ensembling import combine_float_predictions
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

    return results
