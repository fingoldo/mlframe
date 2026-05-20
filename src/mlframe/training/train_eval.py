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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from pyutilz.system import clean_ram, get_own_memory_usage
from mlframe.training.utils import maybe_clean_ram_adaptive

from .configs import (
    TargetTypes,
    DEFAULT_CALIBRATION_BINS,
    DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH,
    DEFAULT_RFECV_MAX_RUNTIME_MINS,
    DEFAULT_RFECV_CV_SPLITS,
    DEFAULT_RFECV_MAX_NOIMPROVING_ITERS,
    LinearModelConfig,
    ModelHyperparamsConfig,
    TrainingBehaviorConfig,
    MultilabelDispatchConfig,
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
            except Exception:
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
                return (
                    f"feature-name mismatch (saved={len(saved_names)}, current={len(current_names)}); "
                    f"symmetric diff sample={sample}"
                )
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
    configure_training_params,
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
        return int(len(np.unique(arr)))
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

    # Remove columns if identical to metadata columns
    if metadata_columns is not None and hasattr(model, "columns") and model.columns is not None:
        model_columns = list(model.columns) if not isinstance(model.columns, list) else model.columns
        if model_columns == metadata_columns:
            model.columns = None


def select_target(
    model_name: str,
    target: Union[np.ndarray, pd.Series, pl.Series],
    target_type: TargetTypes,
    df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    train_idx: Optional[np.ndarray] = None,
    val_idx: Optional[np.ndarray] = None,
    test_idx: Optional[np.ndarray] = None,
    train_details: str = "",
    val_details: str = "",
    test_details: str = "",
    group_ids: Optional[np.ndarray] = None,
    cat_features: Optional[List[str]] = None,
    text_features: Optional[List[str]] = None,
    embedding_features: Optional[List[str]] = None,
    hyperparams_config: Optional[ModelHyperparamsConfig] = None,
    behavior_config: Optional[TrainingBehaviorConfig] = None,
    common_params: Optional[Dict[str, Any]] = None,
    sample_weight: Optional[np.ndarray] = None,
    mlframe_models: Optional[List[str]] = None,
    linear_model_config: Optional[LinearModelConfig] = None,
    train_df_size_bytes: Optional[float] = None,
    val_df_size_bytes: Optional[float] = None,
    multilabel_dispatch_config: Optional["MultilabelDispatchConfig"] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Configure model parameters for a specific target variable.

    From multiple possible targets in a dataframe, selects the required one
    and adjusts parameters for respective level 0 models.

    Parameters
    ----------
    model_name : str
        Base name for the model, will be augmented with target statistics.
    target : np.ndarray, pd.Series, or pl.Series
        Target values for training.
    target_type : TargetTypes
        Type of ML task (REGRESSION or BINARY_CLASSIFICATION).
    df : pd.DataFrame
        Full dataset containing features.
    train_df : pd.DataFrame, optional
        Pre-split training features DataFrame.
    test_df : pd.DataFrame, optional
        Pre-split test features DataFrame.
    val_df : pd.DataFrame, optional
        Pre-split validation features DataFrame.
    train_idx : np.ndarray, optional
        Indices for training rows in df.
    val_idx : np.ndarray, optional
        Indices for validation rows in df.
    test_idx : np.ndarray, optional
        Indices for test rows in df.
    train_details : str, default=""
        Additional details for training set report.
    val_details : str, default=""
        Additional details for validation set report.
    test_details : str, default=""
        Additional details for test set report.
    group_ids : np.ndarray, optional
        Group identifiers for per-group AUC computation.
    cat_features : list of str, optional
        Names of categorical feature columns.
    hyperparams_config : ModelHyperparamsConfig, optional
        Model hyperparameters (iterations, learning_rate, per-model kwargs).
    behavior_config : TrainingBehaviorConfig, optional
        Training behavior flags (GPU preference, calibration, fairness).
    common_params : dict, optional
        Common parameters passed to all models.
    sample_weight : np.ndarray, optional
        Sample weights for weighted training.
    mlframe_models : list of str, optional
        List of model types to create (e.g., ["cb", "lgb", "linear"]).
        If None, all models are created. Used for lazy model creation.
    linear_model_config : LinearModelConfig, optional
        Configuration for linear models. If provided, applies to all linear
        model types (linear, ridge, lasso, etc.).

    Returns
    -------
    tuple
        (common_params, models_params, rfecv_models_params, cpu_configs, gpu_configs)
        - common_params: Updated common parameters dict
        - models_params: Per-model parameter dicts
        - rfecv_models_params: RFECV configuration dicts
        - cpu_configs: CPU-specific model configurations
        - gpu_configs: GPU-specific model configurations

    Raises
    ------
    TypeError
        If target is not a supported type (np.ndarray, pd.Series, pl.Series).
    """
    # Per-split target summary. Previously the BT=
    # / MT= / ML= summary was computed on the FULL target (train+val+test),
    # which masked split-specific drift in forward-mode runs (one user's
    # log showed BT=74% as the only number, but actual splits were
    # train=74 / val=86 / test=83 - selection bias hiding in plain sight).
    #
    # New format: model_name carries ONLY the train-side rate as
    # ``BTTR=`` / ``MTTR=`` / ``MLTR=`` (TR for "train"). The per-split
    # report headers in ``_compute_split_metrics`` splice the matching
    # ``/BTV=`` / ``/BTTS=`` (V for "val", TS for "test") inline so
    # chart titles read e.g. ``BTTR/BTV=74%/86%`` (VAL) and
    # ``BTTR/BTTS=74%/83%`` (TEST). Drift is then visible in every
    # split's chart header.
    #
    # Falls back to the legacy whole-target ``BT=`` / ``MT=`` / ``ML=``
    # tag when no train_idx is supplied (back-compat for direct
    # unit-test callers that don't go through the suite).

    def _select(target_arr, idx):
        """Slice target by index array (np / pd / pl-aware).

        Wave 28 P1 fix (2026-05-20): pre-fix ``idx is False`` matched
        only the Python ``False`` singleton; ``numpy.False_`` from
        upstream caller (e.g. result of ``mask.any()`` style code)
        slipped past the guard and the function tried to slice with a
        scalar bool. Match against both Python bool and numpy scalar
        bool explicitly via ``isinstance``.
        """
        if idx is None:
            return None
        if isinstance(idx, (bool, np.bool_)) and not bool(idx):
            return None
        if hasattr(idx, "__len__") and len(idx) == 0:
            return None
        if isinstance(target_arr, (pl.Series, np.ndarray)):
            return target_arr[idx]
        # pd.Series — accept either positional indices (np.ndarray) or
        # a pre-aligned label index.
        if hasattr(idx, "dtype") and idx.dtype != bool:
            return target_arr.iloc[idx]
        return target_arr[idx]

    def _to_arr(t):
        if t is None:
            return None
        if isinstance(t, pl.Series):
            return t.to_numpy()
        if isinstance(t, pd.Series):
            return t.to_numpy()
        return np.asarray(t)

    train_t = _to_arr(_select(target, train_idx))

    if target_type == TargetTypes.REGRESSION:
        # Adaptive metric format (2 d.p. by default, widening for sub-1 values) instead of fixed :.4f.
        from ._format import format_metric as _fmt
        from .composite_transforms import is_composite_target_name
        # For composite targets the train mean is the T-scale residual
        # mean which is ~0 by OLS construction -- not informative.
        # Switch the label to ``MTRESID=`` to make the semantic
        # explicit. Detection covers every registered transform (incl.
        # monotonic/ewma/quantile/etc.), not just the original four.
        _is_composite = is_composite_target_name(model_name)
        _tag = "MTRESID" if _is_composite else "MTTR"
        if train_t is not None and train_t.size > 0:
            model_name += f" {_tag}={_fmt(train_t.mean())}"
        else:
            model_name += f" MT={_fmt(target.mean())}"
    elif target_type == TargetTypes.MULTILABEL_CLASSIFICATION:
        # Multilabel has 2-D target (N, K). Skip
        # the binary value_counts / positive-rate path (would raise
        # "Data must be 1-dimensional" on pandas.Series construction).
        target_arr = target if isinstance(target, np.ndarray) else np.asarray(target)
        if train_t is not None and train_t.ndim == 2 and train_t.shape[0] > 0:
            rates = train_t.mean(axis=0)
            summary = ",".join(f"{p*100:.0f}" for p in rates)
            model_name += f" MLTR={summary}%"
        elif target_arr.ndim == 2:
            per_label_pos = target_arr.mean(axis=0)
            summary = ",".join(f"{p*100:.0f}%" for p in per_label_pos)
            model_name += f" ML={summary}"
        else:
            model_name += f" ML=?"
    else:
        # Binary / multiclass — train rate on the model_name; per-split
        # contextual rates are appended downstream in _compute_split_metrics.

        def _binary_pos_rate(arr):
            """Robust binary-positive-rate computation.

            Hardened against object-arrays of arrays (some LTR / hybrid
            target shapes hand back
            ``np.array([np.array([...]), ...], dtype=object)`` so naive
            ``arr == 1`` comparisons broadcast per-cell into nested bool
            arrays, and the outer comparison trips
            ``ValueError: truth value of an array with more than one element
            is ambiguous``).
            """
            if arr is None:
                return None
            # Coerce to a 1-D numeric numpy array. ``ravel()`` flattens
            # any (n, 1) / (n,) / (n, K) -- for multilabel-like 2D
            # targets the rate is computed across all positions, which
            # is the closest reasonable analog to "binary pos rate" on
            # a non-strictly-binary target. The model_name suffix is
            # only used for display; an inexact rate on edge target
            # shapes is preferable to crashing the suite.
            try:
                if hasattr(arr, "to_numpy"):
                    arr_np = arr.to_numpy()
                else:
                    arr_np = np.asarray(arr)
                # If the array's dtype is object (e.g. nested arrays),
                # try unwrapping one level. ``np.concatenate`` of an
                # object-array of arrays gives a flat numeric array
                # when the elements are array-like.
                if arr_np.dtype == object and arr_np.size > 0 and isinstance(arr_np.flat[0], np.ndarray):
                    arr_np = np.concatenate([np.asarray(a).ravel() for a in arr_np.ravel()])
                arr_np = arr_np.ravel()
                size = arr_np.size
                if size == 0:
                    return None
                # Cast to numeric where possible; for non-numeric
                # targets (e.g. string labels) the comparison is fine
                # but ``==1`` returns an all-False array which is the
                # correct semantic ("no row equals integer 1").
                count = int(np.asarray(arr_np == 1, dtype=bool).sum())
                return float(count) / size
            except Exception:
                # Defensive: never crash the suite from a display-only
                # metric. Return None so the caller falls back.
                return None

        train_perc = _binary_pos_rate(train_t)

        if train_perc is not None:
            model_name += f" BTTR={train_perc*100:.0f}%"
            perc = train_perc
        else:
            # No train indices — fall back to whole-target rate (rare:
            # direct unit-test callers).
            if isinstance(target, (pl.Series, pd.Series)):
                vlcnts = target.value_counts(normalize=True)
            elif isinstance(target, np.ndarray):
                vlcnts = pd.Series(target).value_counts(normalize=True)
            else:
                raise TypeError(
                    f"target must be np.ndarray, pd.Series, or pl.Series, "
                    f"got {type(target).__name__}"
                )
            if isinstance(target, pl.Series):
                vlcnts = vlcnts.filter(pl.col(target.name) == 1)
                perc = vlcnts["proportion"][0] if len(vlcnts) > 0 else 0
            else:
                perc = vlcnts.loc[1] if 1 in vlcnts.index else 0
            model_name += f" BT={perc*100:.0f}%"

        # Degenerate-target guard: classification with a single class
        # (all 0s or all 1s after the target-building threshold) makes
        # ROC AUC / PR AUC undefined, the scorer returns NaN, and
        # early-stopping / ICE / RFECV all silently break in downstream
        # layers (some of which we already added NaN observability for).
        # Better to WARN loud right here so the operator can fix the
        # threshold or data selection. We do NOT abort -- in rare cases
        # (sanity runs, debugging) training a model on a single-class
        # target is intentional, and a hard raise would regress
        # legitimate callers.
        if 0.0 < perc < 1.0:
            # Also flag extreme imbalance (<0.1% or >99.9%) because
            # even with both classes present the signal is near-zero.
            if perc < 1e-3 or perc > (1.0 - 1e-3):
                logger.warning(
                    "select_target: extreme class imbalance for '%s' "
                    "(positive rate %.4f%%). Training may converge on "
                    "the majority class; AUC metrics will be noisy.",
                    model_name, perc * 100,
                )
        else:
            logger.warning(
                "select_target: degenerate classification target '%s' "
                "has only one class (positive rate=%.0f%%). ROC AUC / "
                "PR AUC are undefined; scorer will return NaN and "
                "early-stopping will stall. Fix the target threshold or "
                "pre-filter the data upstream.",
                model_name, perc * 100,
            )
    logger.debug("select_target: model_name=%s", model_name)

    # Ensure configs have defaults
    if hyperparams_config is None:
        hyperparams_config = ModelHyperparamsConfig()
    if behavior_config is None:
        behavior_config = TrainingBehaviorConfig()

    # Convert Pydantic configs to dicts for configure_training_params
    # exclude_none=True: downstream functions handle missing keys with their own defaults
    effective_config_params = hyperparams_config.model_dump(exclude_none=True)
    # Only include defined fields -- exclude any extra fields (e.g. _precomputed_fairness_subgroups).
    # Also exclude suite-level meta-flags that have no meaning to
    # configure_training_params (crash reporting + per-model
    # continue-on-failure are consumed directly by
    # train_mlframe_models_suite, not passed down the stack).
    defined_behavior_fields = set(TrainingBehaviorConfig.model_fields.keys())
    _SUITE_LEVEL_FLAGS = {
        "enable_crash_reporting",
        "continue_on_model_failure",
        "align_polars_categorical_dicts",
        # Save-time filename policy; not consumed by
        # configure_training_params, so mask from the downstream kwarg set.
        "model_file_hash_suffix",
        # Temporal-audit knobs are consumed by
        # train_mlframe_models_suite directly (in the per-target loop),
        # not by configure_training_params.
        "target_temporal_audit_column",
        "target_temporal_audit_granularity",
        "target_temporal_audit_save_plot",
        # Suite-level reporting knobs consumed directly by ``train_mlframe_models_suite`` (residual-audit thread-local override and the score_ensemble ``uncertainty_quantile`` arg) - never passed down to ``configure_training_params``.
        "report_residual_audit",
        "confidence_ensemble_quantile",
    }
    effective_behavior_params = {
        k: v for k, v in behavior_config.model_dump(exclude_none=True).items()
        if k in defined_behavior_fields and k not in _SUITE_LEVEL_FLAGS
    }
    # Pass _precomputed_fairness_subgroups explicitly if present (set by _build_common_params_for_target)
    precomputed_fairness = (behavior_config.model_extra or {}).get("_precomputed_fairness_subgroups")
    if precomputed_fairness is not None:
        effective_behavior_params["_precomputed_fairness_subgroups"] = precomputed_fairness

    (
        common_params,
        models_params,
        cb_rfecv,
        lgb_rfecv,
        xgb_rfecv,
        cpu_configs,
        gpu_configs,
    ) = configure_training_params(
        df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target=target,
        target_label_encoder=None,
        cat_features=cat_features,
        text_features=text_features,
        embedding_features=embedding_features,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        sample_weight=sample_weight,
        train_details=train_details,
        val_details=val_details,
        test_details=test_details,
        group_ids=group_ids,
        model_name=model_name,
        common_params=common_params,
        config_params=effective_config_params,
        use_regression=target_type == TargetTypes.REGRESSION,
        mlframe_models=mlframe_models,
        linear_model_config=linear_model_config,
        train_df_size_bytes=train_df_size_bytes,
        val_df_size_bytes=val_df_size_bytes,
        target_type=target_type,
        n_classes=_n_classes_from_target(target, target_type),
        multilabel_dispatch_config=multilabel_dispatch_config,
        **effective_behavior_params,
    )

    rfecv_models_params = dict(
        cb_rfecv=cb_rfecv,
        lgb_rfecv=lgb_rfecv,
        xgb_rfecv=xgb_rfecv,
    )
    return common_params, models_params, rfecv_models_params, cpu_configs, gpu_configs


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
    return train_and_evaluate_model(
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
    )


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
        if verbose:
            logger.info("Loading model from file %s", fpath)
        loaded_model = load_mlframe_model(fpath)
        if loaded_model is None:
            # Load returned None (e.g. _SafeUnpickler rejected an unsafe class,
            # file corrupted, version skew). The loader logs the root cause;
            # we fall back to retraining rather than attempting to use a
            # half-loaded artifact and tripping AttributeError downstream on
            # loaded_model.model.
            logger.warning(
                f"Cached model load returned None at {fpath} -- "
                f"retraining. (Check earlier WARN for the real cause: "
                f"unsafe class blocked by allowlist, corrupted file, etc.)"
            )
            use_cached_model = False
        else:
            if verbose:
                logger.info(f"Loaded.")
            mismatch = _validate_cached_model_schema(loaded_model, common_params.get("train_df"))
            if mismatch:
                logger.warning(f"Invalidating stale cached model at {fpath}: {mismatch}. Retraining.")
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
                if (
                    _model_cls_name.startswith("CatBoost")
                    and not getattr(model_obj, "_mlframe_polars_fastpath_broken", False)
                ):
                    try:
                        model_obj._mlframe_polars_fastpath_broken = True
                    except Exception:
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
        model_type_name = type(model_obj).__name__
        logger.info(
            f"Starting train_and_evaluate {model_type_name} on {target_type} {pipeline_label} {model_name.strip()}"
            f", RAM usage {get_own_memory_usage():.1f}GBs...".replace("  ", " ")
        )

    model, train_df_transformed, val_df_transformed, test_df_transformed = _call_train_evaluate_with_configs(
        model_obj=model_obj,
        model_params=model_params,
        common_params=effective_common_params,
        pre_pipeline=pre_pipeline,
        skip_pre_pipeline_transform=skip_pre_pipeline_transform,
        skip_preprocessing=skip_preprocessing,
        model_name_prefix=pre_pipeline_name,
        just_evaluate=use_cached_model,
        verbose=verbose,
        trainset_features_stats=trainset_features_stats,
    )

    # Handle failed model - don't save or add to lists
    if model.model is None:
        logger.warning(f"Skipping failed model {model_name}")
        return trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed

    if not use_cached_model:
        end = timer()
        if verbose:
            logger.info(f"Finished training, took {(end-start)/60:.1f} min. RAM usage {get_own_memory_usage():.1f}GBs...")
        if fpath:
            save_mlframe_model(model, fpath)

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
