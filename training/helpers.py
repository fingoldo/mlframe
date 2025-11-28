"""
Training helper functions and callback classes.

This module contains helper utilities migrated from training_old.py:
- parse_catboost_devices: GPU device parsing for CatBoost
- get_training_configs: Training configuration factory
- get_trainset_features_stats: Compute training set statistics (pandas)
- get_trainset_features_stats_polars: Compute training set statistics (polars)
- UniversalCallback: Base callback class for training monitoring
- LightGBMCallback, XGBoostCallback, CatBoostCallback: Model-specific callbacks
"""

import logging
import psutil
from timeit import default_timer as timer
from types import SimpleNamespace
from typing import Optional, Dict, List, Callable, Sequence, Any, Union

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
import xgboost as xgb
from xgboost.callback import TrainingCallback

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from pyutilz.system import get_gpuinfo_gpu_info, tqdmu
from pyutilz.pythonlib import get_parent_func_args, store_params_in_object

from mlframe.helpers import get_own_ram_usage
from mlframe.metrics import (
    compute_probabilistic_multiclass_error,
    robust_mlperf_metric,
    ICE,
)
from mlframe.lightninglib import MLPTorchModel, TorchDataModule

from .utils import get_numeric_columns, get_categorical_columns

logger = logging.getLogger(__name__)


# Constant - CUDA availability
try:
    from numba.cuda import is_available as is_cuda_available
    CUDA_IS_AVAILABLE = is_cuda_available()
except (ImportError, AttributeError, ModuleNotFoundError):
    CUDA_IS_AVAILABLE = False


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# GPU Device Parsing
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def parse_catboost_devices(devices: str, all_gpus: list = None) -> List[Dict]:
    """
    Parses a GPU devices string and returns a list of GPU info dicts
    corresponding to the specified device indices.

    Parameters
    ----------
    devices : str
        A string specifying device indices. Formats supported:
          - "0"             (single GPU)
          - "0:1:3"         (multiple GPUs)
          - "0-3"           (range of GPUs, inclusive)

    Returns
    -------
    list[dict]
        Filtered list of GPU info dictionaries.
    """

    if not all_gpus:
        all_gpus = get_gpuinfo_gpu_info()

    if not devices:
        return all_gpus

    # Parse the devices string
    device_indices = []
    try:
        if "-" in devices:  # range format
            parts = devices.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid range format '{devices}'. Expected 'start-end' (e.g., '0-3')")
            start, end = parts
            start_int, end_int = int(start), int(end)
            if start_int > end_int:
                raise ValueError(f"Invalid range '{devices}': start ({start_int}) > end ({end_int})")
            device_indices = list(range(start_int, end_int + 1))
        elif ":" in devices:  # multiple specific GPUs
            device_indices = [int(x) for x in devices.split(":")]
        else:  # single GPU
            device_indices = [int(devices)]
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"Invalid device specification '{devices}'. Must contain integers only.") from e
        raise

    # Validate indices
    max_index = len(all_gpus) - 1
    invalid = [i for i in device_indices if i < 0 or i > max_index]
    if invalid:
        raise ValueError(f"Invalid GPU indices {invalid}. Available range: 0-{max_index}")

    # Filter GPU list
    filtered_gpus = [gpu for gpu in all_gpus if gpu["index"] in device_indices]
    return filtered_gpus


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Training Configuration Factory
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def get_training_configs(
    iterations: int = 5000,
    early_stopping_rounds: int = 0,
    validation_fraction: float = 0.1,
    use_explicit_early_stopping: bool = True,
    has_time: bool = True,
    has_gpu: bool = None,
    subgroups: dict = None,
    learning_rate: float = 0.1,
    def_regr_metric: str = "MAE",
    def_classif_metric: str = "AUC",
    catboost_custom_classif_metrics: Optional[Sequence] = None,
    catboost_custom_regr_metrics: Optional[Sequence] = None,
    random_seed: Optional[int] = None,
    verbose: int = 0,
    # ----------------------------------------------------------------------------------------------------------------------------
    # probabilistic errors
    # ----------------------------------------------------------------------------------------------------------------------------
    method: str = "multicrit",
    mae_weight: float = 3,
    std_weight: float = 2,
    roc_auc_weight: float = 1.5,
    pr_auc_weight: float = 0.1,
    brier_loss_weight: float = 0.4,
    min_roc_auc: float = 0.54,
    roc_auc_penalty: float = 0.00,
    use_weighted_calibration: bool = True,
    weight_by_class_npositives: bool = False,
    nbins: int = 10,
    # ----------------------------------------------------------------------------------------------------------------------------
    # robustness parameters for early stopping metric
    # ----------------------------------------------------------------------------------------------------------------------------
    robustness_num_ts_splits: int = 0,  # 0 = disabled, >0 = number of consecutive time splits
    robustness_std_coeff: float = 0.1,  # multiplier for std penalty
    robustness_greater_is_better: bool = False,  # False for ICE (lower is better)
    # ----------------------------------------------------------------------------------------------------------------------------
    # model-specific params
    # ----------------------------------------------------------------------------------------------------------------------------
    cb_kwargs: dict = None,
    hgb_kwargs: dict = None,
    lgb_kwargs: dict = None,
    xgb_kwargs: dict = None,
    mlp_kwargs: dict = None,
    ngb_kwargs: dict = None,
    # ----------------------------------------------------------------------------------------------------------------------------
    # featureselectors
    # ----------------------------------------------------------------------------------------------------------------------------
    rfecv_kwargs: dict = None,
) -> tuple:
    """Returns comparable training configs for different types of models,
    based on general params supplied like learning rate, task type, time budget.
    Useful for more or less fair comparison between different models on the same data/task, and their upcoming ensembling.
    This procedure is good for getting the feeling of what ML models are capable of for a particular task.
    """

    if has_gpu is None:
        has_gpu = CUDA_IS_AVAILABLE

    # Initialize mutable defaults
    if catboost_custom_classif_metrics is None:
        catboost_custom_classif_metrics = ["AUC", "BrierScore", "PRAUC"]
    if catboost_custom_regr_metrics is None:
        catboost_custom_regr_metrics = ["RMSE", "MAPE"]

    # Initialize kwargs dicts with defaults, making copies to avoid mutating caller's dicts
    if cb_kwargs is None:
        cb_kwargs = dict(verbose=0)
    else:
        cb_kwargs = cb_kwargs.copy()  # Don't mutate caller's dict
    if lgb_kwargs is None:
        lgb_kwargs = dict(verbose=-1)
    else:
        lgb_kwargs = lgb_kwargs.copy()  # Don't mutate caller's dict
    if xgb_kwargs is None:
        xgb_kwargs = dict(verbosity=0)
    else:
        xgb_kwargs = xgb_kwargs.copy()  # Don't mutate caller's dict
    if hgb_kwargs is None:
        hgb_kwargs = dict(verbose=0)
    if mlp_kwargs is None:
        mlp_kwargs = dict()
    if ngb_kwargs is None:
        ngb_kwargs = dict(verbose=True)

    if not early_stopping_rounds:
        early_stopping_rounds = max(2, iterations // 3)

    def neg_ovr_roc_auc_score(*args, **kwargs):
        return -roc_auc_score(*args, **kwargs, multi_class="ovr")

    CB_GENERAL_PARAMS = dict(
        iterations=iterations,
        has_time=has_time,
        learning_rate=learning_rate,
        eval_fraction=(0.0 if use_explicit_early_stopping else validation_fraction),
        task_type=cb_kwargs.pop("task_type", "GPU" if has_gpu else "CPU"),  # Pop from kwargs if present
        early_stopping_rounds=early_stopping_rounds,
        random_seed=random_seed,
        **cb_kwargs,
    )

    CB_CLASSIF = CB_GENERAL_PARAMS.copy()
    CB_CLASSIF.update({"eval_metric": def_classif_metric, "custom_metric": catboost_custom_classif_metrics})

    CB_REGR = CB_GENERAL_PARAMS.copy()
    CB_REGR.update({"eval_metric": def_regr_metric, "custom_metric": catboost_custom_regr_metrics})

    HGB_GENERAL_PARAMS = dict(
        max_iter=iterations,
        learning_rate=learning_rate,
        early_stopping=True,
        validation_fraction=(None if use_explicit_early_stopping else validation_fraction),
        n_iter_no_change=early_stopping_rounds,
        categorical_features="from_dtype",
        random_state=random_seed,
        **hgb_kwargs,
    )

    XGB_GENERAL_PARAMS = dict(
        n_estimators=iterations,
        learning_rate=learning_rate,
        enable_categorical=True,
        max_cat_to_onehot=1,
        max_cat_threshold=100,  # affects model size heavily when high cardinality cat features r present!
        tree_method="hist",
        device=xgb_kwargs.pop("device", "cuda" if has_gpu else "cpu"),  # Pop from kwargs if present
        n_jobs=psutil.cpu_count(logical=False),
        early_stopping_rounds=early_stopping_rounds,
        random_seed=random_seed,
        **xgb_kwargs,
    )

    XGB_GENERAL_CLASSIF = XGB_GENERAL_PARAMS.copy()
    XGB_GENERAL_CLASSIF.update({"objective": "binary:logistic", "eval_metric": neg_ovr_roc_auc_score})

    def integral_calibration_error(y_true: np.ndarray, y_score: np.ndarray, verbose: bool = False) -> float:
        """Compute integral calibration error for probabilistic predictions.

        Wraps compute_probabilistic_multiclass_error with the outer function's
        configuration parameters (method, weights, etc.).

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.
        y_score : np.ndarray
            Predicted probabilities.
        verbose : bool, default=False
            If True, print calibration error info.

        Returns
        -------
        float
            The computed calibration error (lower is better).
        """
        err = compute_probabilistic_multiclass_error(
            y_true=y_true,
            y_score=y_score,
            method=method,
            mae_weight=mae_weight,
            std_weight=std_weight,
            brier_loss_weight=brier_loss_weight,
            roc_auc_weight=roc_auc_weight,
            pr_auc_weight=pr_auc_weight,
            min_roc_auc=min_roc_auc,
            roc_auc_penalty=roc_auc_penalty,
            use_weighted_calibration=use_weighted_calibration,
            weight_by_class_npositives=weight_by_class_npositives,
            nbins=nbins,
            verbose=verbose,
        )
        if verbose:
            print(len(y_true), "integral_calibration_error=", err)
        return err

    def make_robust_ts_metric(
        metric_fn,
        num_splits: int,
        std_coeff: float,
        greater_is_better: bool,
        min_samples_per_split: int = 100,
        ensure_enough_classes: bool = False,
        verbose: int = 0,
    ):
        """Wrap a metric to evaluate across consecutive time splits.

        Returns mean(metric_values) ± std(metric_values) * std_coeff
        where ± is + if greater_is_better=False (penalize variance for minimization)
              and - if greater_is_better=True (penalize variance for maximization)
        """

        def robust_metric(y_true: np.ndarray, y_score: np.ndarray, *args, **kwargs):
            n = len(y_true)

            # Fallback 1: Not enough data for any splits
            if n < min_samples_per_split:
                if verbose:
                    logger.info(f"make_robust_ts_metric: n={n} < min_samples_per_split={min_samples_per_split}, using full data")
                return metric_fn(y_true, y_score, *args, **kwargs)

            # Compute actual number of splits we can do
            actual_splits = min(num_splits, n // min_samples_per_split)

            # Fallback 2: Can only do 1 split
            if actual_splits <= 1:
                if verbose:
                    logger.info(f"make_robust_ts_metric: actual_splits={actual_splits} <= 1, using full data")
                return metric_fn(y_true, y_score, *args, **kwargs)

            # Split into consecutive intervals
            split_size = n // actual_splits
            values = []

            for i in range(actual_splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < actual_splits - 1 else n

                y_true_split = y_true[start_idx:end_idx]
                y_score_split = y_score[start_idx:end_idx]

                # Skip split if not enough samples
                if len(y_true_split) < min_samples_per_split:
                    if verbose:
                        logger.info(f"make_robust_ts_metric: split {i} skipped, len={len(y_true_split)} < {min_samples_per_split}")
                    continue

                # Skip split if single class (classification only)
                if ensure_enough_classes and len(np.unique(y_true_split)) < 2:
                    if verbose:
                        logger.info(f"make_robust_ts_metric: split {i} skipped, single class in y_true")
                    continue

                val = metric_fn(y_true_split, y_score_split, *args, **kwargs)
                if not np.isnan(val):
                    values.append(val)

            # Fallback 3: No valid splits computed
            if len(values) == 0:
                if verbose:
                    logger.info("make_robust_ts_metric: no valid splits, using full data")
                return metric_fn(y_true, y_score, *args, **kwargs)

            # Fallback 4: Only one valid split
            if len(values) == 1:
                if verbose:
                    logger.info(f"make_robust_ts_metric: only 1 valid split, returning {values[0]:.6f}")
                return values[0]

            mean_val = np.mean(values)
            std_val = np.std(values)

            if verbose:
                logger.info(f"make_robust_ts_metric: {len(values)} splits, mean={mean_val:.6f}, std={std_val:.6f}")

            # Penalize high variance
            if greater_is_better:
                # For maximization: subtract std penalty (lower result = worse)
                return mean_val - std_val * std_coeff
            else:
                # For minimization: add std penalty (higher result = worse)
                return mean_val + std_val * std_coeff

        return robust_metric

    if subgroups:

        def final_integral_calibration_error(y_true: np.ndarray, y_score: np.ndarray, *args, **kwargs):  # partial won't work with xgboost
            return robust_mlperf_metric(
                y_true,
                y_score,
                *args,
                metric=integral_calibration_error,
                higher_is_better=False,
                subgroups=subgroups,
                **kwargs,
            )

    else:
        final_integral_calibration_error = integral_calibration_error

    # Apply robustness wrapper if enabled
    if robustness_num_ts_splits > 0:
        final_integral_calibration_error = make_robust_ts_metric(
            final_integral_calibration_error,
            num_splits=robustness_num_ts_splits,
            std_coeff=robustness_std_coeff,
            greater_is_better=robustness_greater_is_better,
            ensure_enough_classes=True,  # ICE is for classification
            verbose=verbose,
        )

    def fs_and_hpt_integral_calibration_error(*args, verbose: bool = True, **kwargs):
        err = compute_probabilistic_multiclass_error(
            *args,
            **kwargs,
            mae_weight=mae_weight,
            std_weight=std_weight,
            brier_loss_weight=brier_loss_weight,
            roc_auc_weight=roc_auc_weight,
            pr_auc_weight=pr_auc_weight,
            min_roc_auc=min_roc_auc,
            roc_auc_penalty=roc_auc_penalty,
            use_weighted_calibration=use_weighted_calibration,
            weight_by_class_npositives=weight_by_class_npositives,
            nbins=nbins,
            verbose=verbose,
        )
        return err

    XGB_CALIB_CLASSIF = XGB_GENERAL_CLASSIF.copy()
    XGB_CALIB_CLASSIF.update({"eval_metric": final_integral_calibration_error})

    def lgbm_integral_calibration_error(y_true, y_score):
        metric_name = "integral_calibration_error"
        value = final_integral_calibration_error(y_true, y_score)
        higher_is_better = False
        return metric_name, value, higher_is_better

    CB_CALIB_CLASSIF = CB_CLASSIF.copy()
    CB_CALIB_CLASSIF.update({"eval_metric": ICE(metric=final_integral_calibration_error, higher_is_better=False, max_arr_size=0)})

    LGB_GENERAL_PARAMS = dict(
        n_estimators=iterations,
        early_stopping_rounds=early_stopping_rounds,
        device_type=lgb_kwargs.pop("device_type", "cuda" if has_gpu else "cpu"),  # Pop from kwargs if present
        random_state=random_seed,
        # histogram_pool_size=16384,
        **lgb_kwargs,
    )

    NGB_GENERAL_PARAMS = dict(
        n_estimators=iterations,
        learning_rate=learning_rate,
        **ngb_kwargs,
    )

    mlp_trainer_params: dict = dict(
        devices=1,  # Always use single device by default to avoid multi-GPU complexity
        # ----------------------------------------------------------------------------------------------------------------------
        # Runtime:
        # ----------------------------------------------------------------------------------------------------------------------
        min_epochs=1,
        max_epochs=iterations,
        max_time={"days": 0, "hours": 0, "minutes": 30},
        # max_steps=1,
        # ----------------------------------------------------------------------------------------------------------------------
        # Intervals:
        # ----------------------------------------------------------------------------------------------------------------------
        check_val_every_n_epoch=1,
        # val_check_interval=val_check_interval,
        # log_every_n_steps=log_every_n_steps,
        # ----------------------------------------------------------------------------------------------------------------------
        # Flags:
        # ----------------------------------------------------------------------------------------------------------------------
        enable_model_summary=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=2,
        # ----------------------------------------------------------------------------------------------------------------------
        # Precision & accelerators:
        # ----------------------------------------------------------------------------------------------------------------------
        precision="32-true",
        num_nodes=1,
        # ----------------------------------------------------------------------------------------------------------------------
        # Logging:
        # ----------------------------------------------------------------------------------------------------------------------
        default_root_dir="logs",
    )

    if mlp_kwargs:
        mlp_trainer_params.update(mlp_kwargs.get("trainer_params", {}))

    # Default loss function and dtype (classification)
    loss_fn = F.cross_entropy
    labels_dtype = torch.int64

    mlp_model_params = dict(
        loss_fn=loss_fn,
        learning_rate=1e-3,
        l1_alpha=0.0,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={},
        lr_scheduler=None,
        lr_scheduler_kwargs={},
    )
    if mlp_kwargs:
        mlp_model_params.update(mlp_kwargs.get("model_params", {}))

    mlp_dataloader_params = dict(
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=None,
        persistent_workers=False,
        batch_size=1024,
        shuffle=False,
    )
    if mlp_kwargs:
        mlp_dataloader_params.update(mlp_kwargs.get("dataloader_params", {}))

    mlp_datamodule_params = dict(
        read_fcn=None, data_placement_device=None, features_dtype=torch.float32, labels_dtype=labels_dtype, dataloader_params=mlp_dataloader_params
    )
    if mlp_kwargs:
        mlp_datamodule_params.update(mlp_kwargs.get("datamodule_params", {}))

    MLP_GENERAL_PARAMS = dict(
        model_class=MLPTorchModel,
        model_params=mlp_model_params,
        datamodule_class=TorchDataModule,
        datamodule_params=mlp_datamodule_params,  # includes dataloader_params
        trainer_params=mlp_trainer_params,
        use_swa=mlp_kwargs.get("use_swa", False) if mlp_kwargs else False,
        swa_params=mlp_kwargs.get("swa_params", dict(swa_epoch_start=5, annealing_epochs=5, swa_lrs=1e-4)) if mlp_kwargs else dict(swa_epoch_start=5, annealing_epochs=5, swa_lrs=1e-4),
        tune_params=mlp_kwargs.get("tune_params", False) if mlp_kwargs else False,
        float32_matmul_precision=mlp_kwargs.get("float32_matmul_precision", None) if mlp_kwargs else None,
        early_stopping_rounds=early_stopping_rounds,
    )

    if rfecv_kwargs is None:
        rfecv_kwargs = {}

    cv = rfecv_kwargs.get("cv")
    if not cv:
        if has_time:
            cv = TimeSeriesSplit(n_splits=rfecv_kwargs.get("cv_n_splits", 3))
            logger.info(f"Using TimeSeriesSplit for RFECV...")
        else:
            cv = None
        rfecv_kwargs["cv"] = cv

    if "cv_n_splits" in rfecv_kwargs:
        del rfecv_kwargs["cv_n_splits"]

    COMMON_RFECV_PARAMS = dict(
        early_stopping_rounds=early_stopping_rounds,
        cv=cv,
        cv_shuffle=not has_time,
    )
    COMMON_RFECV_PARAMS.update(rfecv_kwargs)

    return SimpleNamespace(
        integral_calibration_error=integral_calibration_error,
        final_integral_calibration_error=final_integral_calibration_error,
        lgbm_integral_calibration_error=lgbm_integral_calibration_error,
        fs_and_hpt_integral_calibration_error=fs_and_hpt_integral_calibration_error,
        CB_GENERAL_PARAMS=CB_GENERAL_PARAMS,
        CB_REGR=CB_REGR,
        CB_CLASSIF=CB_CLASSIF,
        CB_CALIB_CLASSIF=CB_CALIB_CLASSIF,
        HGB_GENERAL_PARAMS=HGB_GENERAL_PARAMS,
        LGB_GENERAL_PARAMS=LGB_GENERAL_PARAMS,
        XGB_GENERAL_PARAMS=XGB_GENERAL_PARAMS,
        XGB_GENERAL_CLASSIF=XGB_GENERAL_CLASSIF,
        XGB_CALIB_CLASSIF=XGB_CALIB_CLASSIF,
        COMMON_RFECV_PARAMS=COMMON_RFECV_PARAMS,
        MLP_GENERAL_PARAMS=MLP_GENERAL_PARAMS,
        NGB_GENERAL_PARAMS=NGB_GENERAL_PARAMS,
    )


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Training Set Feature Statistics
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def get_trainset_features_stats(train_df: pd.DataFrame, max_ncats_to_track: int = 1000) -> dict:
    """Computes ranges of numerical and categorical variables"""
    res = {}
    num_cols = get_numeric_columns(train_df)
    if num_cols:
        if len(num_cols) == train_df.shape[1]:
            res["min"] = train_df.min(axis=0)
            res["max"] = train_df.max(axis=0)
        else:
            # TypeError: Categorical is not ordered for operation min. you can use .as_ordered() to change the Categorical to an ordered one.
            res["min"] = pd.Series({col: train_df[col].min() for col in num_cols})
            res["max"] = pd.Series({col: train_df[col].max() for col in num_cols})

    cat_cols = get_categorical_columns(train_df, include_string=False)
    if cat_cols:
        cat_vals = {}
        for col in tqdmu(cat_cols, desc="cat vars stats", leave=False):
            unique_vals = train_df[col].unique()
            if not max_ncats_to_track or (len(unique_vals) <= max_ncats_to_track):
                cat_vals[col] = unique_vals
        res["cat_vals"] = cat_vals
    return res


def get_trainset_features_stats_polars(train_df: pl.DataFrame, max_ncats_to_track: int = 1000) -> dict:
    """Computes ranges of numerical and categorical variables using Polars.

    Uses lazy mode and selectors for parallel computation.

    Args:
        train_df: Polars DataFrame
        max_ncats_to_track: Max unique values to track for categorical columns

    Returns:
        dict with "min", "max" (as pd.Series) and "cat_vals" (dict of arrays)
    """
    import polars.selectors as cs

    res = {}
    lf = train_df.lazy()

    # Compute numeric min/max and categorical n_unique in a single parallel select
    stats = lf.select(
        # Numeric: min and max
        cs.numeric().min().name.suffix("__min"),
        cs.numeric().max().name.suffix("__max"),
        # Categorical: n_unique to filter before getting unique values
        cs.by_dtype(pl.String, pl.Categorical).n_unique().name.suffix("__n_unique"),
    ).collect()

    # Extract numeric stats
    if len(stats.columns) > 0:
        mins = {}
        maxs = {}
        for col in stats.columns:
            if col.endswith("__min"):
                orig_col = col[:-5]
                mins[orig_col] = stats[col][0]
            elif col.endswith("__max"):
                orig_col = col[:-5]
                maxs[orig_col] = stats[col][0]

        if mins:
            res["min"] = pd.Series(mins)
        if maxs:
            res["max"] = pd.Series(maxs)

    # Extract categorical columns that are under the threshold
    cat_cols_to_fetch = []
    for col in stats.columns:
        if col.endswith("__n_unique"):
            orig_col = col[:-10]
            n_unique = stats[col][0]
            if not max_ncats_to_track or n_unique <= max_ncats_to_track:
                cat_cols_to_fetch.append(orig_col)

    # Get unique values for qualifying categorical columns
    if cat_cols_to_fetch:
        cat_vals = {}
        # Compute unique values in parallel using lazy mode
        unique_exprs = [pl.col(c).unique().alias(c) for c in cat_cols_to_fetch]
        unique_results = lf.select(unique_exprs).collect()
        for col in cat_cols_to_fetch:
            cat_vals[col] = unique_results[col].to_numpy()
        res["cat_vals"] = cat_vals

    return res


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Callback Classes for Training Monitoring
# -----------------------------------------------------------------------------------------------------------------------------------------------------


class UniversalCallback:
    def __init__(
        self,
        time_budget_mins: Optional[float] = None,
        reporting_interval_mins: Optional[float] = 1.0,
        patience: Optional[int] = None,
        min_delta: float = 0.0,
        monitor_dataset: Optional[str] = None,
        monitor_metric: Optional[str] = None,
        mode: Optional[str] = None,
        stop_flag: Optional[Callable[[], bool]] = None,
        ndigits: int = 6,
        verbose: int = 1,
    ) -> None:

        params = get_parent_func_args()
        store_params_in_object(obj=self, params=params)

        self.start_time = None
        self.best_metric = None
        self.first_iteration = True
        self.iterations_since_improvement = 0
        self.metric_history: Dict[str, Dict[str, List[float]]] = {}
        self.stop_flag = stop_flag if stop_flag is not None else lambda: False

        if self.verbose > 0:
            logger.info(
                "UniversalCallback initialized with params: "
                f"time_budget_mins={time_budget_mins}, patience={patience}, min_delta={min_delta}, "
                f"monitor_dataset={monitor_dataset}, monitor_metric={monitor_metric}, mode={mode}"
            )

    def on_start(self) -> None:
        self.start_time = timer()
        if self.verbose > 0:
            self.last_reporting_ts = self.start_time
            logger.info(f"Training started. Timer initiated. RAM usage {get_own_ram_usage():.1f}GB.")

    def update_history(self, metrics_dict: Dict[str, Dict[str, float]]) -> None:
        for dataset in metrics_dict:
            if dataset not in self.metric_history:
                self.metric_history[dataset] = {}
            for metric, value in metrics_dict[dataset].items():
                self.metric_history[dataset].setdefault(metric, []).append(value)
        if self.verbose > 1:
            logger.debug(f"Updated metric history: {metrics_dict}")

    def derive_mode(self, metric_name: str) -> str:
        known_metric_modes = {
            "auc": "max",
            "accuracy": "max",
            "acc": "max",
            "f1": "max",
            "map": "max",
            "ndcg": "max",
            "ice": "min",
            "mae": "min",
            "mse": "min",
            "mape": "min",
            "rmse": "min",
            "logloss": "min",
            "error": "min",
            "loss": "min",
        }

        name = metric_name.lower()
        for key, default_mode in known_metric_modes.items():
            if key == name:
                return default_mode
        if "score" in name or "auc" in name or "accuracy" in name:
            return "max"
        elif "loss" in name or "error" in name:
            return "min"
        elif name.endswith("e"):
            return "min"
        else:
            logger.warning(f"Unsure about correct optimization mode for metric={name}, using min for now.")
            return "min"  # fallback default

    def set_default_monitor_metric(self, metrics_dict: Dict[str, Dict[str, float]]) -> None:
        if self.monitor_dataset not in metrics_dict:
            raise ValueError(f"Monitor dataset '{self.monitor_dataset}' not found in metrics.")
        available_metrics = list(metrics_dict[self.monitor_dataset].keys())
        logger.info(f"available_metrics={available_metrics}")
        for preferred in ["ICE", "integral_calibration_error", "auc", "AUC"]:
            if preferred in available_metrics:
                self.monitor_metric = preferred
                break
        else:
            self.monitor_metric = available_metrics[0]
        self.mode = self.derive_mode(self.monitor_metric)
        if self.verbose > 0:
            logger.info(f"Auto-selected monitor_metric: {self.monitor_metric}, mode: {self.mode}")

    def _get_state(self, current_value: float) -> str:
        return f"iter={self.iter:_}, {self.monitor_dataset} {self.monitor_metric}: current={current_value:.{self.ndigits}f}, best={self.best_metric:.{self.ndigits}f} @{self.best_iter:_}. RAM usage {get_own_ram_usage():.1f}GB."

    def should_stop(self) -> bool:
        cur_ts = timer()
        if self.time_budget_mins is not None and self.start_time is not None:

            elapsed = cur_ts - self.start_time
            if elapsed > self.time_budget_mins * 60:
                if self.verbose > 0:
                    logger.info(f"Stopping early due to time budget exceeded ({elapsed:.2f} sec).")
                return True

        if self.stop_flag():
            if self.verbose > 0:
                logger.info("Stopping early due to external stop flag.")
            return True

        if self.monitor_dataset in self.metric_history and self.monitor_metric in self.metric_history[self.monitor_dataset]:
            history = self.metric_history[self.monitor_dataset][self.monitor_metric]
            if history:
                current_value = history[-1]
                if self.best_metric is None:
                    self.iter = 0
                    self.best_iter = self.iter
                    self.best_metric = current_value
                    self.iterations_since_improvement = 0
                    if self.verbose > 0:
                        logger.info(f"Initial metric value: {current_value:.{self.ndigits}f}")
                        self.last_reporting_ts = cur_ts
                else:
                    self.iter += 1
                    improved = (self.mode == "min" and current_value < self.best_metric - self.min_delta) or (
                        self.mode == "max" and current_value > self.best_metric + self.min_delta
                    )
                    # Pre-compute reporting condition (used in both branches)
                    should_report = self.verbose > 0 and (
                        not self.reporting_interval_mins or (cur_ts - self.last_reporting_ts) >= self.reporting_interval_mins * 60
                    )
                    if improved:
                        self.best_iter = self.iter
                        self.best_metric = current_value
                        self.iterations_since_improvement = 0
                    else:
                        self.iterations_since_improvement += 1
                    if should_report:
                        logger.info(self._get_state(current_value=current_value))
                        self.last_reporting_ts = cur_ts
                    if self.patience is not None and self.iterations_since_improvement >= self.patience:
                        if self.verbose > 0:
                            logger.info(
                                f"Stopping early due to no improvement for {self.iterations_since_improvement} iterations. {self._get_state(current_value=current_value)}"
                            )
                            self.last_reporting_ts = cur_ts
                        return True
        return False


class LightGBMCallback(UniversalCallback):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.monitor_dataset = self.monitor_dataset or "valid_0"

    def __call__(self, env: lgb.callback.CallbackEnv) -> bool:
        if self.first_iteration:
            self.on_start()
            self.first_iteration = False
        metrics_dict = {}
        for dataset, metric, value, _ in env.evaluation_result_list:
            metrics_dict.setdefault(dataset, {})[metric] = value
        self.update_history(metrics_dict)
        if self.monitor_metric is None:
            self.set_default_monitor_metric(metrics_dict)
        if self.should_stop():
            if hasattr(self, "best_iter"):
                best_iter = self.best_iter
            else:
                best_iter = 0
            raise lgb.callback.EarlyStopException(best_iter, [(dataset, metric, self.best_metric, False)])


class XGBoostCallback(UniversalCallback, TrainingCallback):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.monitor_dataset = self.monitor_dataset or "validation_0"

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: Dict[str, Dict[str, List[float]]]) -> bool:
        if self.first_iteration:
            self.on_start()
            self.first_iteration = False
        metrics_dict = {dataset: {metric: values[-1] for metric, values in metric_dict.items()} for dataset, metric_dict in evals_log.items()}
        self.update_history(metrics_dict)
        if self.monitor_metric is None:
            self.set_default_monitor_metric(metrics_dict)

        if self.should_stop():
            if hasattr(self, "best_iter"):
                best_iter = self.best_iter
            else:
                best_iter = 0
            model.set_attr(best_score=self.best_metric, best_iteration=best_iter)
            return True


class CatBoostCallback(UniversalCallback):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.monitor_dataset = self.monitor_dataset or "validation"

    def after_iteration(self, info: Any) -> bool:
        if self.first_iteration:
            self.on_start()
            self.first_iteration = False
        metrics_dict = {dataset: {metric: values[-1] for metric, values in metric_dict.items()} for dataset, metric_dict in info.metrics.items()}
        self.update_history(metrics_dict)
        if self.monitor_metric is None:
            self.set_default_monitor_metric(metrics_dict)
        return not self.should_stop()


__all__ = [
    'parse_catboost_devices',
    'get_training_configs',
    'get_trainset_features_stats',
    'get_trainset_features_stats_polars',
    'UniversalCallback',
    'LightGBMCallback',
    'XGBoostCallback',
    'CatBoostCallback',
    'CUDA_IS_AVAILABLE',
]
