"""Stages of ``process_model``."""

from __future__ import annotations

# --- imports (managed) ---
from mlframe.training._model_cache_fingerprint import training_fingerprint_mismatch
from mlframe.training.io import load_mlframe_model
from mlframe.training.train_eval import _composite_cache_mismatch
from mlframe.training.train_eval import _flag_catboost_polars_fastpath
from mlframe.training.train_eval import _validate_cached_model_schema
from mlframe.training.train_eval import logger
from mlframe.training.train_eval import optimize_model_for_storage
from pyutilz.system import get_own_memory_usage
from typing import Any
# --- end imports ---


def _load_cached_model_into_state(use_cached_model, fpath, verbose, common_params, fingerprint, model_obj, pre_pipeline):
    """Stage of ``process_model``."""
    if use_cached_model:
        assert fpath is not None  # guaranteed by the use_cached_model construction above
        if verbose:
            logger.info("Loading model from file %s", fpath)
        loaded_model: Any = load_mlframe_model(fpath)
        if loaded_model is None:
            # Load returned None (e.g. _SafeUnpickler rejected an unsafe class, file corrupted, version skew). The loader logs the root cause; we fall back to
            # retraining rather than attempting to use a half-loaded artifact and tripping AttributeError downstream on loaded_model.model.
            logger.warning(
                "Cached model load returned None at %s -- "
                "retraining. (Check earlier WARN for the real cause: "
                "unsafe class blocked by allowlist, corrupted file, etc.)",
                fpath,
            )
            use_cached_model = False
        else:
            if verbose:
                logger.info("Loaded.")
            mismatch = _validate_cached_model_schema(loaded_model, common_params.get("train_df"))
            if not mismatch:
                mismatch = _composite_cache_mismatch(loaded_model, common_params.get("composite_spec_digest"))
            if not mismatch:
                mismatch = training_fingerprint_mismatch(loaded_model, fingerprint)
            if mismatch:
                logger.warning("Invalidating stale cached model at %s: %s. Retraining.", fpath, mismatch)
                use_cached_model = False
            else:
                model_obj = loaded_model.model
                # A dump re-saved after composite wrapping holds the y-scale wrapper; evaluation here runs on T.
                if getattr(model_obj, "_routes_inner_input", False):
                    model_obj = model_obj.estimator_
                pre_pipeline = loaded_model.pre_pipeline
                # Restore the Polars-fastpath sticky flag.
                # CB's pickle/joblib serialization writes through CatBoost's
                # native ``save_model``, which doesn't preserve arbitrary
                # Python attributes set on the estimator (verified by a
                # prod log: ``cb_recency`` reload still hit the
                # ``predict_proba RAISED TypeError`` polars-fastpath miss
                # despite the original CB instance having had the flag set
                # at fit time). Re-derive it for any reloaded CB from the installed build's own probe: some CB 1.2.x builds
                # have polars dispatch gaps on nullable Categorical / Enum columns and some do not, so asserting it
                # unconditionally buys a pandas conversion on every VAL/TEST/ensemble call on a build that never needed one.
                # No-op on non-CB models (the attribute is never read for them).
                _model_cls_name = type(model_obj).__name__
                _flag_catboost_polars_fastpath(_model_cls_name, model_obj)
    return model_obj, pre_pipeline, use_cached_model


def _register_trained_model(fpath, model, optimize_storage, target_type, metadata_columns, pre_pipeline_name, model_file_name, models, cur_target_name, ens_models):
    """Stage of ``process_model``."""
    # Where this entry's dump lives, so composite post-processing can re-save it once it wraps the model.
    if fpath:
        try:
            model.model_file_path = fpath
        except AttributeError:
            logger.debug("could not stamp model_file_path on %s", type(model).__name__)

    # Optimize model for in-memory storage (after saving to disk to preserve full data in files)
    if optimize_storage:
        optimize_model_for_storage(model, target_type, metadata_columns)

    # A unique, stable identity for this entry among its target's models: feature pre-pipeline + model + weight schema
    # (``model_file_name`` already carries the weight, e.g. "cb_recency"). Metadata blocks key results by
    # ``entry.model_name``; without it they fell back to the class name / list index, so different weight schemas,
    # pre-pipelines and wrapped composite-target models overwrote each other's bootstrap CI, calibration, fairness...
    try:
        model.model_name = f"{(pre_pipeline_name or '').strip()} {model_file_name}".strip()
    except AttributeError:
        logger.debug("could not stamp model_name on %s", type(model).__name__)
    models.setdefault(target_type, {}).setdefault(cur_target_name, []).append(model)

    # ens_models can be None when not building ensembles
    if ens_models is not None:
        ens_models.append(model)


def _log_train_start(verbose, use_cached_model, pre_pipeline_name, model_obj, target_type, model_name):
    """Stage of ``process_model``."""
    if verbose and not use_cached_model:
        pipeline_label = pre_pipeline_name.strip() if pre_pipeline_name else ""
        from mlframe.training.reporting import display_estimator_name
        model_type_name = display_estimator_name(type(model_obj).__name__)
        _start_msg = (
            f"Starting train_and_evaluate {model_type_name} on {target_type} {pipeline_label} {model_name.strip()}"
            f", RAM usage {get_own_memory_usage():.1f}GBs...".replace("  ", " ")
        )
        logger.info("%s", _start_msg)


def _effective_common_params(common_params, model_name, cached_train_df, cached_val_df, cached_test_df):
    """Stage of ``process_model``."""
    effective_common_params = common_params.copy()
    effective_common_params["model_name"] = model_name
    if cached_train_df is not None:
        effective_common_params["train_df"] = cached_train_df
    if cached_val_df is not None:
        effective_common_params["val_df"] = cached_val_df
    if cached_test_df is not None:
        effective_common_params["test_df"] = cached_test_df

    # Remove parameters not accepted by train_and_evaluate_model
    for key in ["scaler", "imputer", "category_encoder", "rfecv_params", "model_path", "artifact_dir", "composite_spec_digest"]:
        effective_common_params.pop(key, None)
    return effective_common_params
