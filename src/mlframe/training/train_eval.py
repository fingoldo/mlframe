"""
Model training and evaluation functions.

This module contains the core functions for training and evaluating models,
including select_target, process_model, and related helpers.

Functions
---------
select_target
    Configure model parameters for a specific target variable.
process_model
    Process a single model: load from cache or train from scratch.
"""

from __future__ import annotations


import logging
from timeit import default_timer as timer
from os.path import join, exists
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import polars as pl

from pyutilz.system import get_own_memory_usage
from mlframe.training.utils import maybe_clean_ram_adaptive

from .configs import (
    TargetTypes,
    DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH,
    DEFAULT_RFECV_MAX_RUNTIME_MINS,
    DEFAULT_RFECV_CV_SPLITS,
    DEFAULT_RFECV_MAX_NOIMPROVING_ITERS,
)
from .io import load_mlframe_model, save_mlframe_model

logger = logging.getLogger(__name__)


def _extract_polars_cat_columns(df) -> List[str]:
    """Return column names whose dtype is `pl.Categorical` or `pl.Enum`.

    Used by cache-schema validation: if a column is Polars Categorical at
    predict time but was not registered as a cat_feature at fit time,
    CatBoost's native-Polars fastpath raises
    ``CatBoostError: Unsupported data type Categorical for a numerical feature column``.
    """
    if df is None or not isinstance(df, pl.DataFrame):
        return []
    out: List[str] = []
    for name, dtype in df.schema.items():
        if dtype == pl.Categorical or isinstance(dtype, pl.Enum):
            out.append(name)
    return out


def _validate_cached_model_schema(
    loaded_model,
    current_df,
) -> Optional[str]:
    """Return a reason string if the cached model's schema is incompatible
    with the current DataFrame, else None.

    Catches the common stale-cache scenarios where preprocessing or the
    feature set changed between the saved model and the current run:
        * the column list or order differs;
        * a column is Polars Categorical in `current_df` but was not
          registered as a cat_feature in the saved CatBoost model.

    The latter produces a cryptic
    ``CatBoostError: Unsupported data type Categorical for a numerical feature column``
    deep inside CatBoost's pyx layer -- this pre-flight check catches it.
    """
    m = getattr(loaded_model, "model", None)
    if m is None:
        return None

    # 1. Feature-name sequence check (CB / XGB / LGB / sklearn linear).
    saved_names: Optional[List[str]] = None
    for attr in ("feature_names_", "feature_names_in_"):
        if hasattr(m, attr):
            try:
                raw = getattr(m, attr)
                if raw is not None:
                    saved_names = list(raw)
                    break
            except Exception as e:
                logger.debug("swallowed exception in train_eval.py: %s", e)
                continue
    if saved_names is None and hasattr(m, "get_booster"):
        try:
            booster_names = m.get_booster().feature_names
            if booster_names:
                saved_names = list(booster_names)
        except Exception:
            saved_names = None

    if saved_names:
        current_names = list(current_df.columns) if current_df is not None else []
        if saved_names != current_names:
            diff = set(current_names) ^ set(saved_names)
            if diff:
                sample = sorted(diff)[:8]
                return f"feature-name mismatch (saved={len(saved_names)}, current={len(current_names)}); " f"symmetric diff sample={sample}"
            return "feature-name order differs between saved model and current df"

    # 2. CatBoost-specific cat_features check. Only meaningful if we have
    # saved feature names (to resolve indices back to names).
    if saved_names and hasattr(m, "_get_cat_feature_indices"):
        try:
            saved_cat_names = {saved_names[i] for i in m._get_cat_feature_indices() if 0 <= i < len(saved_names)}
        except Exception:
            saved_cat_names = None
        if saved_cat_names is not None:
            current_pl_cats = set(_extract_polars_cat_columns(current_df))
            missing = current_pl_cats - saved_cat_names
            if missing:
                return (
                    f"CatBoost cache mismatch: columns {sorted(missing)} are Polars Categorical in "
                    f"current df but were not trained as cat_features in the saved model"
                )
    return None


# =============================================================================
# Constants
# =============================================================================

# Constants now imported from configs.py (DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH, etc.)

# Import from trainer module
from .trainer import (
    _build_configs_from_params,
    train_and_evaluate_model,
)

logger = logging.getLogger(__name__)


def _n_classes_from_target(target, target_type: Optional[TargetTypes]) -> Optional[int]:
    """Derive K for per-strategy classification dispatch.

    MULTILABEL: K = number of label columns (target.shape[1]).
    MULTICLASS: K = number of unique values in 1-D target.
    BINARY/REGRESSION/None: returns None (caller leaves dispatch alone).
    """
    if target is None or target_type is None:
        return None
    if target_type == TargetTypes.MULTILABEL_CLASSIFICATION:
        arr = np.asarray(target)
        return int(arr.shape[1]) if arr.ndim == 2 else 1
    if target_type == TargetTypes.MULTICLASS_CLASSIFICATION:
        arr = np.asarray(target)
        if arr.ndim != 1:
            return None
        return len(np.unique(arr))
    return None


# =============================================================================
# Storage Optimization
# =============================================================================


def optimize_model_for_storage(
    model,
    target_type: TargetTypes,
    metadata_columns: Optional[List[str]] = None,
) -> None:
    """Optimize a model object for storage by removing redundant data.

    For classification models:
    - Removes train_preds, val_preds, test_preds (can be recreated from *_probs >= 0.5)

    For binary classification:
    - Keeps *_probs arrays in original (n, 2) shape for compatibility with
      load_mlframe_suite and ensembling code

    If metadata_columns is provided:
    - Removes model.columns if identical to metadata_columns

    Parameters
    ----------
    model : SimpleNamespace
        Model object with predictions and metadata attributes.
    target_type : TargetTypes
        Type of ML task (REGRESSION or BINARY_CLASSIFICATION).
    metadata_columns : list of str, optional
        Columns stored in metadata. If provided and identical to model.columns,
        model.columns will be set to None to save storage.

    Notes
    -----
    This function modifies the model object in-place.
    """
    is_classification = target_type == TargetTypes.BINARY_CLASSIFICATION

    if is_classification:
        # Remove *_preds for classification (can be recreated from *_probs >= 0.5)
        model.train_preds = None
        model.val_preds = None
        model.test_preds = None

        # NOTE: Do NOT squeeze probs from (n, 2) to (n,) here.
        # The in-memory models must keep 2D probs to match load_mlframe_suite output
        # and to remain compatible with ensembling code (ensemble_probabilistic_predictions
        # expects 2D arrays via pred.shape[1]).

    # Remove columns if identical to metadata columns.
    # ``list(model.columns)`` raises ``TypeError: iteration over a 0-d array``
    # when ``model.columns`` is a 0-d numpy scalar (single-column edge case;
    # reproduced on c0030_beb1dc9b @200k regression where the MLP path
    # surfaced this 376s into the fit, aborting the entire suite). Coerce
    # to a plain list defensively before the equality check.
    if metadata_columns is not None and hasattr(model, "columns") and model.columns is not None:
        if isinstance(model.columns, list):
            model_columns = model.columns
        elif isinstance(model.columns, np.ndarray) and model.columns.ndim == 0:
            model_columns = [model.columns.item()]
        else:
            model_columns = list(model.columns)
        if model_columns == metadata_columns:
            model.columns = None


# select_target carved to ``_train_eval_select_target``; re-exported below.
from .targets import select_target


def _call_train_evaluate_with_configs(
    model_obj: Optional[Any],
    model_params: Dict[str, Any],
    common_params: Dict[str, Any],
    pre_pipeline: Optional[Any],
    skip_pre_pipeline_transform: bool,
    skip_preprocessing: bool,
    model_name_prefix: str,
    just_evaluate: bool = False,
    verbose: bool = False,
    trainset_features_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Call train_and_evaluate_model with config objects.

    Helper function that merges model_params and common_params, builds
    configuration objects, and calls the config-based API.

    Parameters
    ----------
    model_obj : Any, optional
        The model object to train (sklearn estimator, Pipeline, etc.).
    model_params : dict
        Model-specific parameters (e.g., fit_params).
    common_params : dict
        Common parameters shared across models (e.g., data, targets, indices).
    pre_pipeline : Any, optional
        Preprocessing pipeline (sklearn TransformerMixin).
    skip_pre_pipeline_transform : bool
        Whether to skip the preprocessing pipeline transform.
    skip_preprocessing : bool
        Whether to skip only preprocessing (scaler/imputer/encoder) while still
        running feature selectors.
    model_name_prefix : str
        Prefix to add to the model name for reports.
    just_evaluate : bool, default=False
        If True, skip training and only evaluate cached predictions.
    verbose : bool, default=False
        Whether to print verbose output.
    trainset_features_stats : dict, optional
        Pre-computed feature statistics from training set.

    Returns
    -------
    tuple
        (model, train_df, val_df, test_df) where:
        - model: Trained model with results attached
        - train_df: Transformed training DataFrame (or None)
        - val_df: Transformed validation DataFrame (or None)
        - test_df: Transformed test DataFrame (or None)
    """
    # Merge all params into flat dict for _build_configs_from_params
    all_params = {**common_params, **model_params}

    # Extract params that go directly to v2 (not through _build_configs_from_params)
    all_params.pop("model", None)  # passed separately
    train_od_idx = all_params.pop("train_od_idx", None)
    val_od_idx = all_params.pop("val_od_idx", None)
    all_params.pop("trainset_features_stats", None)  # use function arg
    # OOF K-fold controls flow straight to the trainer, not through the config builder.
    oof_n_splits = all_params.pop("oof_n_splits", 0)
    oof_has_time = all_params.pop("oof_has_time", False)
    oof_random_seed = all_params.pop("oof_random_seed", 42)

    # Add control params
    all_params["pre_pipeline"] = pre_pipeline
    all_params["skip_pre_pipeline_transform"] = skip_pre_pipeline_transform
    all_params["skip_preprocessing"] = skip_preprocessing
    all_params["model_name_prefix"] = model_name_prefix
    all_params["just_evaluate"] = just_evaluate
    all_params["verbose"] = verbose

    # Build config objects
    data, control, metrics, reporting, naming, confidence, predictions, output = _build_configs_from_params(**all_params)

    # Call train_and_evaluate_model with config objects
    _result = train_and_evaluate_model(
        model=model_obj,
        data=data,
        control=control,
        metrics=metrics,
        reporting=reporting,
        naming=naming,
        output=output,
        confidence=confidence,
        predictions=predictions,
        train_od_idx=train_od_idx,
        val_od_idx=val_od_idx,
        trainset_features_stats=trainset_features_stats,
        oof_n_splits=oof_n_splits,
        oof_has_time=oof_has_time,
        oof_random_seed=oof_random_seed,
    )
    return cast(Tuple[Any, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]], _result)


def process_model(
    model_file: str,
    model_name: str,
    model_file_name: str,
    target_type: TargetTypes,
    pre_pipeline: Optional[Any],
    pre_pipeline_name: str,
    cur_target_name: str,
    trainset_features_stats: Optional[Dict[str, Any]],
    models: Dict[str, Dict[str, List[Any]]],
    model_params: Dict[str, Any],
    common_params: Dict[str, Any],
    ens_models: Optional[List[Any]],
    verbose: int,
    skip_pre_pipeline_transform: bool = False,
    skip_preprocessing: bool = False,
    cached_train_df: Optional[pd.DataFrame] = None,
    cached_val_df: Optional[pd.DataFrame] = None,
    cached_test_df: Optional[pd.DataFrame] = None,
    optimize_storage: bool = True,
    metadata_columns: Optional[List[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Any], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Process a single model: load from cache or train from scratch.

    Handles model loading from cache if available, otherwise trains the model
    and optionally saves it. Updates the models dict and ensemble list.

    Parameters
    ----------
    model_file : str
        Directory path for saving/loading model files.
    model_name : str
        Name identifier for the model.
    target_type : TargetTypes
        Type of ML task (REGRESSION or BINARY_CLASSIFICATION).
    pre_pipeline : Any, optional
        Preprocessing pipeline (sklearn TransformerMixin).
    pre_pipeline_name : str
        Name of the preprocessing pipeline for file naming.
    cur_target_name : str
        Current target column name being processed.
    trainset_features_stats : dict, optional
        Statistics computed on training features (e.g., means, stds).
    models : dict
        Dictionary to store trained models, keyed by target name and type.
    model_params : dict
        Model-specific parameters including the 'model' key.
    common_params : dict
        Common parameters for training (data, indices, etc.).
    ens_models : list, optional
        List to collect models for ensemble. Can be None if not using ensembles.
    verbose : int
        Verbosity level (0=silent, 1=info, 2=debug).
    skip_pre_pipeline_transform : bool, default=False
        Whether to skip the preprocessing pipeline transform.
    skip_preprocessing : bool, default=False
        Whether to skip only preprocessing (scaler/imputer/encoder) while still
        running feature selectors.
    cached_train_df : pd.DataFrame, optional
        Pre-transformed training DataFrame to reuse.
    cached_val_df : pd.DataFrame, optional
        Pre-transformed validation DataFrame to reuse.
    cached_test_df : pd.DataFrame, optional
        Pre-transformed test DataFrame to reuse.

    Returns
    -------
    tuple
        (trainset_features_stats, pre_pipeline, train_df, val_df, test_df)
        - trainset_features_stats: Updated feature statistics
        - pre_pipeline: The preprocessing pipeline (may be updated from cache)
        - train_df: Transformed training DataFrame
        - val_df: Transformed validation DataFrame
        - test_df: Transformed test DataFrame

    Raises
    ------
    KeyError
        If 'model' key is missing in model_params when not loading from cache.
    """
    # Build model file path
    fname = f"{model_file_name}.dump"
    if pre_pipeline_name:
        fname = pre_pipeline_name + " " + fname
    fpath = join(model_file, fname) if model_file else None

    # Prepare common_params with cached DataFrames if provided
    effective_common_params = common_params.copy()
    effective_common_params["model_name"] = model_name
    if cached_train_df is not None:
        effective_common_params["train_df"] = cached_train_df
    if cached_val_df is not None:
        effective_common_params["val_df"] = cached_val_df
    if cached_test_df is not None:
        effective_common_params["test_df"] = cached_test_df

    # Remove parameters not accepted by train_and_evaluate_model
    for key in ["scaler", "imputer", "category_encoder", "rfecv_params", "model_path", "artifact_dir"]:
        effective_common_params.pop(key, None)

    # Check if model exists in cache.
    #
    # Gating: historically this path loaded blindly whenever the .dump
    # existed. Preserve that as the default (suite-level cache is expected
    # to "just work"); callers force a retrain via
    # ``TrainingControlConfig(use_cache=False)`` (this flag is read off the
    # internal ``common_params`` dict that the suite assembles from the
    # typed configs).
    #
    # Schema validation: when the cache does load, validate the saved
    # model's feature list + cat_features against the current preprocessed
    # DataFrame. A mismatch usually means preprocessing or the feature set
    # changed between runs -- the classic symptom being a cryptic
    # ``Unsupported data type Categorical for a numerical feature column``
    # crash deep in CatBoost's Polars fastpath. Invalidate the stale cache
    # and retrain rather than bubble the opaque backend error.
    use_cache_flag = bool(common_params.get("use_cache", True))
    use_cached_model = use_cache_flag and bool(fpath and exists(fpath))
    if use_cached_model:
        assert fpath is not None  # guaranteed by the use_cached_model construction above
        if verbose:
            logger.info("Loading model from file %s", fpath)
        loaded_model: Any = load_mlframe_model(fpath)
        if loaded_model is None:
            # Load returned None (e.g. _SafeUnpickler rejected an unsafe class,
            # file corrupted, version skew). The loader logs the root cause;
            # we fall back to retraining rather than attempting to use a
            # half-loaded artifact and tripping AttributeError downstream on
            # loaded_model.model.
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
            if mismatch:
                logger.warning("Invalidating stale cached model at %s: %s. Retraining.", fpath, mismatch)
                use_cached_model = False
            else:
                model_obj = loaded_model.model
                pre_pipeline = loaded_model.pre_pipeline
                # Restore the Polars-fastpath sticky flag.
                # CB's pickle/joblib serialization writes through CatBoost's
                # native ``save_model``, which doesn't preserve arbitrary
                # Python attributes set on the estimator (verified by a
                # prod log: ``cb_recency`` reload still hit the
                # ``predict_proba RAISED TypeError`` polars-fastpath miss
                # despite the original CB instance having had the flag set
                # at fit time). Set it defensively for any reloaded CB --
                # we know CB 1.2.x's polars fastpath has dispatch gaps on
                # nullable Categorical / Enum columns, and a wasted retry
                # on every VAL/TEST/ensemble call burns a WARN + ~1-2 s.
                # No-op on non-CB models (the attribute is never read for
                # them).
                _model_cls_name = type(model_obj).__name__
                if _model_cls_name.startswith("CatBoost") and not getattr(model_obj, "_mlframe_polars_fastpath_broken", False):
                    try:
                        model_obj._mlframe_polars_fastpath_broken = True
                    except Exception:  # nosec B110 - non-trivial body
                        # CB Python class is permissive about attributes,
                        # but slot-restricted forks could refuse -- degrade
                        # to "pay one extra retry" rather than fail.
                        pass
    if not use_cached_model:
        if "model" not in model_params:
            raise KeyError(f"'model' key missing in model_params. Available keys: {list(model_params.keys())}")
        model_obj = model_params["model"]

    maybe_clean_ram_adaptive()

    # Train or evaluate the model
    start = timer()
    if verbose and not use_cached_model:
        pipeline_label = pre_pipeline_name.strip() if pre_pipeline_name else ""
        from mlframe.training.reporting import display_estimator_name
        model_type_name = display_estimator_name(type(model_obj).__name__)
        _start_msg = (
            f"Starting train_and_evaluate {model_type_name} on {target_type} {pipeline_label} {model_name.strip()}"
            f", RAM usage {get_own_memory_usage():.1f}GBs...".replace("  ", " ")
        )
        logger.info("%s", _start_msg)

    model, train_df_transformed, val_df_transformed, test_df_transformed = _call_train_evaluate_with_configs(
        model_obj=model_obj,
        model_params=model_params,
        common_params=effective_common_params,
        pre_pipeline=pre_pipeline,
        skip_pre_pipeline_transform=skip_pre_pipeline_transform,
        skip_preprocessing=skip_preprocessing,
        model_name_prefix=pre_pipeline_name,
        just_evaluate=use_cached_model,
        verbose=bool(verbose),
        trainset_features_stats=trainset_features_stats,
    )

    # Handle failed model - don't save or add to lists
    if model.model is None:
        logger.warning("Skipping failed model %s", model_name)
        return trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed

    if not use_cached_model:
        end = timer()
        if verbose:
            logger.info("Finished training, took %.1f min. RAM usage %.1fGBs...", (end - start) / 60, get_own_memory_usage())
        if fpath:
            # lean=True. The train-time save here is the inference-ready
            # bundle the harness / serving stack reads back; train_preds + train_target
            # (4M float32 each = ~32 MB on a large prod regression) + trainset_features_stats
            # ballooned MLP dumps to 135 MB in a prod run. The sibling
            # save at _phase_finalize.py:122 already used lean=True; this one missed it.
            # In-memory ``model`` is unchanged (lean affects the on-disk copy only), so
            # downstream in-process predict / metric computation continues to read the
            # original preds. Operators who need the forensic snapshot can re-save with
            # lean=False explicitly (rare; the metrics dict on the model object already
            # carries every train/val/test scalar score).
            save_mlframe_model(model, fpath, lean=True)

    # Optimize model for in-memory storage (after saving to disk to preserve full data in files)
    if optimize_storage:
        optimize_model_for_storage(model, target_type, metadata_columns)

    models.setdefault(target_type, {}).setdefault(cur_target_name, []).append(model)

    # ens_models can be None when not building ensembles
    if ens_models is not None:
        ens_models.append(model)

    if trainset_features_stats is None:
        trainset_features_stats = model.trainset_features_stats
        common_params["trainset_features_stats"] = trainset_features_stats

    maybe_clean_ram_adaptive()

    return trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed


__all__ = [
    # Constants
    "DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH",
    "DEFAULT_RFECV_MAX_RUNTIME_MINS",
    "DEFAULT_RFECV_CV_SPLITS",
    "DEFAULT_RFECV_MAX_NOIMPROVING_ITERS",
    # Functions
    "optimize_model_for_storage",
    "select_target",
    "process_model",
    "_call_train_evaluate_with_configs",
]
