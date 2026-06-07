"""``select_target`` carved out of ``mlframe.training.train_eval``.

Re-imported at the parent module's bottom so historical
``from mlframe.training.train_eval import select_target`` import sites keep working.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from ..configs import (
    LinearModelConfig,
    ModelHyperparamsConfig,
    MultilabelDispatchConfig,
    TargetTypes,
    TrainingBehaviorConfig,
)

logger = logging.getLogger("mlframe.training.train_eval")


def _n_classes_from_target(target, target_type):
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

    See parent ``mlframe.training.train_eval`` docstring history for full
    parameter / return semantics.
    """
    # Lazy import of trainer.configure_training_params to avoid the
    # heavyweight trainer module being imported on every train_eval import.
    from ..trainer import configure_training_params

    # Per-split target summary. Previously the BT= / MT= / ML= summary was
    # computed on the FULL target (train+val+test), which masked
    # split-specific drift. New format: model_name carries ONLY the
    # train-side rate as ``BTTR=`` / ``MTTR=`` / ``MLTR=``. Per-split
    # report headers splice the matching ``/BTV=`` / ``/BTTS=`` inline.

    def _select(target_arr, idx):
        """Slice target by index array (np / pd / pl-aware)."""
        if idx is None:
            return None
        if isinstance(idx, (bool, np.bool_)) and not bool(idx):
            return None
        if hasattr(idx, "__len__") and len(idx) == 0:
            return None
        if isinstance(target_arr, (pl.Series, np.ndarray)):
            return target_arr[idx]
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
        from .._format import format_metric as _fmt
        from ..composite.transforms import is_composite_target_name
        _is_composite = is_composite_target_name(model_name)
        _tag = "MTRESID" if _is_composite else "MTTR"
        if train_t is not None and train_t.size > 0:
            model_name += f" {_tag}={_fmt(train_t.mean())}"
        else:
            model_name += f" MT={_fmt(target.mean())}"
    elif target_type == TargetTypes.MULTILABEL_CLASSIFICATION:
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
        # Binary / multiclass -- train rate on the model_name; per-split
        # contextual rates are appended downstream in _compute_split_metrics.

        def _binary_pos_rate(arr):
            """Robust binary-positive-rate computation."""
            if arr is None:
                return None
            try:
                if hasattr(arr, "to_numpy"):
                    arr_np = arr.to_numpy()
                else:
                    arr_np = np.asarray(arr)
                if arr_np.dtype == object and arr_np.size > 0 and isinstance(arr_np.flat[0], np.ndarray):
                    arr_np = np.concatenate([np.asarray(a).ravel() for a in arr_np.ravel()])
                arr_np = arr_np.ravel()
                size = arr_np.size
                if size == 0:
                    return None
                count = int(np.asarray(arr_np == 1, dtype=bool).sum())
                return float(count) / size
            except Exception:
                return None

        train_perc = _binary_pos_rate(train_t)

        if train_perc is not None:
            model_name += f" BTTR={train_perc*100:.0f}%"
            perc = train_perc
        else:
            # No train indices -- fall back to whole-target rate.
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

        # Degenerate-target guard.
        if 0.0 < perc < 1.0:
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

    effective_config_params = hyperparams_config.model_dump(exclude_none=True)
    defined_behavior_fields = set(TrainingBehaviorConfig.model_fields.keys())
    _SUITE_LEVEL_FLAGS = {
        "enable_crash_reporting",
        "continue_on_model_failure",
        "align_polars_categorical_dicts",
        "model_file_hash_suffix",
        "target_temporal_audit_column",
        "target_temporal_audit_granularity",
        "target_temporal_audit_save_plot",
        "report_residual_audit",
        "confidence_ensemble_quantile",
        "feature_drift_auto_apply_neural_overrides",
    }
    effective_behavior_params = {
        k: v for k, v in behavior_config.model_dump(exclude_none=True).items()
        if k in defined_behavior_fields and k not in _SUITE_LEVEL_FLAGS
    }
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
        # All regression flavours (REGRESSION, MULTI_TARGET_REGRESSION,
        # QUANTILE_REGRESSION) must build REGRESSOR estimators. Keying on the
        # bare ``== REGRESSION`` left MTR/QR with use_regression=False, so
        # configure_training_params built CLASSIFIERS for them; the MTR
        # MultiRMSE loss is then layered on later (only on a regressor), and a
        # classifier with a classification loss on a 2-D continuous target
        # crashes at fit ("Target Labels for MultiLogloss must be 0 or 1").
        use_regression=target_type.is_any_regression,
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
