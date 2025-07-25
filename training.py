# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from typing import *  # noqa: F401 pylint: disable=wildcard-import,unused-wildcard-import
from .config import *

# from pyutilz.pythonlib import ensure_installed;ensure_installed("pandas numpy numba scikit-learn lightgbm catboost xgboost shap")

import re
import copy
import inspect

import os
import zstandard as zstd
from functools import partial
from types import SimpleNamespace
from collections import defaultdict


from timeit import default_timer as timer
from pyutilz.system import ensure_dir_exists, tqdmu
from pyutilz.pythonlib import prefix_dict_elems, get_human_readable_set_size

from mlframe.helpers import get_model_best_iter, ensure_no_infinity, get_own_ram_usage

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Filesystem
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import glob
from os.path import basename
from os.path import join, exists
from pyutilz.strings import slugify
from pyutilz.system import ensure_dir_exists

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Dimreducers
# -----------------------------------------------------------------------------------------------------------------------------------------------------

# import umap

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# OD
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.ensemble import IsolationForest

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Ensembling
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from mlframe.ensembling import ensemble_probabilistic_predictions, score_ensemble, compare_ensembles

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# FE
# -----------------------------------------------------------------------------------------------------------------------------------------------------


from mlframe.feature_engineering.basic import create_date_features
from mlframe.feature_engineering.timeseries import create_aggregated_features
from mlframe.feature_engineering.numerical import (
    compute_simple_stats_numba,
    get_simple_stats_names,
    compute_numaggs,
    get_numaggs_names,
    compute_numaggs_parallel,
)
from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering
from mlframe.feature_engineering.categorical import compute_countaggs, get_countaggs_names

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Base classes
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from abc import ABC
from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin, is_classifier

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from IPython.display import display


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# IPython
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from IPython.display import display

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Pandas
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import polars.selectors as cs
import pandas as pd, numpy as np, polars as pl
from pyutilz.pandaslib import get_df_memory_consumption, showcase_df_columns
from pyutilz.pandaslib import ensure_dataframe_float32_convertability, optimize_dtypes, remove_constant_columns, convert_float64_to_float32

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Hi perf & parallel
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from pyutilz.system import clean_ram

import numba
from numba import njit, prange
from numba.cuda import is_available as is_cuda_available

import psutil

import dill
import joblib
from joblib import delayed, Parallel
from pyutilz.parallel import distribute_work, parallel_run

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Curated models
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBClassifier, XGBRegressor, DMatrix, QuantileDMatrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor

from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# FS
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from mlframe.feature_selection.wrappers import RFECV, VotesAggregation, OptimumSearch
from mlframe.feature_selection.filters import MRMR

try:
    from optbinning import BinningProcess
except Exception as e:
    pass

from mlframe.custom_estimators import log_plus_c, inv_log_plus_c, box_cox_plus_c, inv_box_cox_plus_c

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Cats
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import category_encoders as ce

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Splitters
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Pipelines
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.compose import ColumnTransformer


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import TransformedTargetRegressor


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Pre- & postprocessing
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from mlframe.preprocessing import prepare_df_for_catboost

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# FIs
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import shap
from mlframe.feature_importance import *


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from mlframe.metrics import create_robustness_subgroups
from mlframe.metrics import fast_roc_auc, fast_calibration_report, compute_probabilistic_multiclass_error, ICE
from mlframe.metrics import create_robustness_subgroups, create_robustness_subgroups_indices, compute_robustness_metrics, robust_mlperf_metric

from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import mean_absolute_error, max_error, mean_absolute_percentage_error, mean_squared_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except Exception as e:

    def root_mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):

        output_errors = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight, multioutput="raw_values"))

        if isinstance(multioutput, str):
            if multioutput == "raw_values":
                return output_errors
            elif multioutput == "uniform_average":
                # pass None as weights to np.average: uniform mean
                multioutput = None

        return np.average(output_errors, weights=multioutput)


# ----------------------------------------------------------------------------------------------------------------------------
# Early stopping
# ----------------------------------------------------------------------------------------------------------------------------

import os
from timeit import default_timer as timer
from xgboost.callback import TrainingCallback

import catboost as cb
import xgboost as xgb
import lightgbm as lgb

from pyutilz.pythonlib import store_params_in_object, get_parent_func_args

# ----------------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------------


def get_function_param_names(func):
    signature = inspect.signature(func)
    return list(signature.parameters.keys())


# ----------------------------------------------------------------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------------------------------------------------------------

from enum import StrEnum, auto


class TargetTypes(StrEnum):
    REGRESSION = auto()
    BINARY_CLASSIFICATION = auto()


# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

import sklearn

sklearn.set_config(transform_output="pandas")  # need this for val_df = pre_pipeline.transform(val_df) to work for SimpleImputer

CUDA_IS_AVAILABLE = is_cuda_available()
MODELS_SUBDIR = "models"
PARQUET_COMPRESION: str = "zstd"
all_results = {}

# ----------------------------------------------------------------------------------------------------------------------------
# Custom Error Metrics & training configs
# ----------------------------------------------------------------------------------------------------------------------------


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
    catboost_custom_classif_metrics: Sequence = ["AUC", "BrierScore", "PRAUC"],
    catboost_custom_regr_metrics: Sequence = ["RMSE", "MAPE"],
    random_seed=None,
    verbose: int = 0,
    # ----------------------------------------------------------------------------------------------------------------------------
    # probabilistic errors
    # ----------------------------------------------------------------------------------------------------------------------------
    method: str = "multicrit",
    mae_weight: float = 3,
    std_weight: float = 2,
    roc_auc_weight: float = 1.5,
    brier_loss_weight: float = 0.4,
    min_roc_auc: float = 0.54,
    roc_auc_penalty: float = 0.00,
    use_weighted_calibration: bool = True,
    weight_by_class_npositives: bool = False,
    nbins: int = 10,
    cb_kwargs: dict = dict(verbose=0),
    lgb_kwargs: dict = dict(verbose=-1),
    xgb_kwargs: dict = dict(verbosity=0),
    # ----------------------------------------------------------------------------------------------------------------------------
    # featureselectors
    # ----------------------------------------------------------------------------------------------------------------------------
    rfecv_kwargs: dict = None,
) -> tuple:
    """Returns approximately same training configs for different types of models,
    based on general params supplied like learning rate, task type, time budget.
    Useful for more or less fair comparison between different models on the same data/task, and their upcoming ensembling.
    This procedure is good for manual EDA and getting the feeling of what ML models are capable of for a particular task.
    """

    if has_gpu is None:
        has_gpu = CUDA_IS_AVAILABLE

    if not early_stopping_rounds:
        early_stopping_rounds = max(2, iterations // 3)

    def neg_ovr_roc_auc_score(*args, **kwargs):
        return -roc_auc_score(*args, **kwargs, multi_class="ovr")

    CB_GENERAL_PARAMS = dict(
        iterations=iterations,
        has_time=has_time,
        learning_rate=learning_rate,
        eval_fraction=(0.0 if use_explicit_early_stopping else validation_fraction),
        task_type=("GPU" if has_gpu else "CPU"),
        early_stopping_rounds=early_stopping_rounds,
        random_seed=random_seed,
        **cb_kwargs,
    )

    CB_CLASSIF = CB_GENERAL_PARAMS.copy()
    CB_CLASSIF.update({"eval_metric": def_classif_metric, "custom_metric": catboost_custom_classif_metrics})

    CB_REGR = CB_GENERAL_PARAMS.copy()
    CB_REGR.update({"eval_metric": def_regr_metric, "custom_metric": catboost_custom_regr_metrics})

    XGB_GENERAL_PARAMS = dict(
        n_estimators=iterations,
        enable_categorical=True,
        max_cat_to_onehot=1,
        max_cat_threshold=100,  # affects model size heavily when high cardinality cat features r present!
        tree_method="hist",
        device=("cuda" if has_gpu else "cpu"),
        n_jobs=psutil.cpu_count(logical=False),
        early_stopping_rounds=early_stopping_rounds,
        random_seed=random_seed,
        **xgb_kwargs,
    )

    XGB_GENERAL_CLASSIF = XGB_GENERAL_PARAMS.copy()
    XGB_GENERAL_CLASSIF.update({"objective": "binary:logistic", "eval_metric": neg_ovr_roc_auc_score})

    def integral_calibration_error(y_true, y_score, verbose: bool = False):

        err = compute_probabilistic_multiclass_error(
            y_true=y_true,
            y_score=y_score,
            method=method,
            mae_weight=mae_weight,
            std_weight=std_weight,
            brier_loss_weight=brier_loss_weight,
            roc_auc_weight=roc_auc_weight,
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

    def fs_and_hpt_integral_calibration_error(*args, verbose: bool = True, **kwargs):
        err = compute_probabilistic_multiclass_error(
            *args,
            **kwargs,
            mae_weight=mae_weight,
            std_weight=std_weight,
            brier_loss_weight=brier_loss_weight,
            roc_auc_weight=roc_auc_weight,
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
        device_type=("cuda" if has_gpu else "cpu"),
        random_state=random_seed,
        # histogram_pool_size=16384,
        **lgb_kwargs,
    )
    """device_type 🔗︎, default = cpu, type = enum, options: cpu, gpu, cuda, aliases: device

    device for the tree learning

    cpu supports all LightGBM functionality and is portable across the widest range of operating systems and hardware

    cuda offers faster training than gpu or cpu, but only works on GPUs supporting CUDA

    gpu can be faster than cpu and works on a wider range of GPUs than CUDA

    Note: it is recommended to use the smaller max_bin (e.g. 63) to get the better speed up"""

    # XGB_CALIB_CLASSIF_CPU.update({"device": "cpu","n_jobs":psutil.cpu_count(logical=False)})

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
        LGB_GENERAL_PARAMS=LGB_GENERAL_PARAMS,
        XGB_GENERAL_PARAMS=XGB_GENERAL_PARAMS,
        XGB_GENERAL_CLASSIF=XGB_GENERAL_CLASSIF,
        XGB_CALIB_CLASSIF=XGB_CALIB_CLASSIF,
        COMMON_RFECV_PARAMS=COMMON_RFECV_PARAMS,
    )


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Outliers detection
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def get_trainset_features_stats(train_df: pd.DataFrame, max_ncats_to_track: int = 1000) -> dict:
    """Computes ranges of numerical and categorical variables"""
    res = {}
    num_cols = train_df.head().select_dtypes("number").columns.tolist()
    if num_cols:
        if len(num_cols) == train_df.shape[1]:
            res["min"] = train_df.min(axis=0)
            res["max"] = train_df.max(axis=0)
        else:
            # TypeError: Categorical is not ordered for operation min. you can use .as_ordered() to change the Categorical to an ordered one.
            res["min"] = pd.Series({col: train_df[col].min() for col in num_cols})
            res["max"] = pd.Series({col: train_df[col].max() for col in num_cols})

    cat_cols = train_df.head().select_dtypes("category").columns.tolist()
    if cat_cols:
        cat_vals = {}
        for col in tqdmu(cat_cols, desc="cat vars stats", leave=False):
            unique_vals = train_df[col].unique()
            if not max_ncats_to_track or (len(unique_vals) <= max_ncats_to_track):
                cat_vals[col] = unique_vals
        res["cat_vals"] = cat_vals
    return res


def compute_outlier_detector_score(df: pd.DataFrame, outlier_detector: object, columns: Sequence = None) -> np.ndarray:
    is_inlier = outlier_detector.predict(
        df.loc[:, columns]
    )  # For each observation, model is expected to tell whether or not (+1 or -1) it should be considered as an inlier
    return (is_inlier == -1).astype(int)  # converts 1 to 0 and -1 to 1, as we need to transform inliers to outliers


@njit()
def count_num_outofranges(vals: np.ndarray, extremums: np.ndarray, mode: str) -> np.ndarray:
    if mode == "min":
        return (vals < extremums).sum(axis=1)
    elif mode == "max":
        return (vals > extremums).sum(axis=1)


def compute_naive_outlier_score(df: pd.DataFrame, trainset_features_stats: dict, columns: Sequence = None, dtype=np.float64) -> np.ndarray:
    """Checks deviation from trainset_features_stats (% of features out of range, per observation/row)."""
    scores = np.zeros(len(df), dtype=np.float64)
    if "min" in trainset_features_stats:
        if columns is None:
            columns = df.columns
        num_cols = df.head().select_dtypes("number").columns.tolist()
        columns = [col for col in columns if col in num_cols]
        if columns:
            tmp = df.loc[:, columns].values
            mins = pd.Series(trainset_features_stats["min"]).loc[columns].values.astype(dtype)
            maxs = pd.Series(trainset_features_stats["max"]).loc[columns].values.astype(dtype)

            scores += (tmp < mins).sum(axis=1) / len(columns)
            scores += (tmp > maxs).sum(axis=1) / len(columns)
    return scores


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def train_and_evaluate_model(
    model: object,  # s
    df: pd.DataFrame = None,
    target: pd.Series = None,  # s
    groups: pd.Series = None,
    group_ids: np.ndarray = None,
    outlier_detector: object = None,
    od_val_set: bool = True,
    trainset_features_stats: dict = None,
    sample_weight: pd.Series = None,
    model_name: str = "",
    model_name_prefix: str = "",
    pre_pipeline: TransformerMixin = None,
    fit_params: Optional[dict] = None,
    drop_columns: list = [],
    default_drop_columns: list = [],
    target_label_encoder: Optional[LabelEncoder] = None,
    train_df: pd.DataFrame = None,
    test_df: pd.DataFrame = None,
    val_df: pd.DataFrame = None,
    train_target: pd.Series = None,
    test_target: pd.Series = None,
    val_target: pd.Series = None,
    train_idx: Optional[np.ndarray] = None,
    test_idx: Optional[np.ndarray] = None,
    val_idx: Optional[np.ndarray] = None,
    train_preds: Optional[np.ndarray] = None,
    train_probs: Optional[np.ndarray] = None,
    test_preds: Optional[np.ndarray] = None,
    test_probs: Optional[np.ndarray] = None,
    val_preds: Optional[np.ndarray] = None,
    val_probs: Optional[np.ndarray] = None,
    custom_ice_metric: Callable = None,
    custom_rice_metric: Callable = None,
    subgroups: dict = None,
    figsize: tuple = (15, 5),
    print_report: bool = True,
    plot_file: str = "",
    show_perf_chart: bool = True,
    show_fi: bool = True,
    fi_kwargs: dict = {},
    use_cache: bool = False,
    nbins: int = 10,
    compute_trainset_metrics: bool = False,
    compute_valset_metrics: bool = True,
    compute_testset_metrics: bool = True,
    data_dir: str = "",
    models_subdir: str = MODELS_SUBDIR,
    display_sample_size: int = 0,
    show_feature_names: bool = False,
    verbose: bool = False,
    use_hpt: bool = False,
    skip_infinity_checks: bool = False,
    # confidence_analysis
    include_confidence_analysis: bool = False,
    confidence_analysis_use_shap: bool = True,
    confidence_analysis_max_features: int = 6,
    confidence_analysis_cmap: str = "bwr",
    confidence_analysis_alpha: float = 0.9,
    confidence_analysis_ylabel: str = "Feature value",
    confidence_analysis_title: str = "Confidence of correct Test set predictions",
    confidence_model_kwargs: dict = {},
    #
    train_details: str = "",
    val_details: str = "",
    test_details: str = "",
    #
    callback_params: dict = None,
):
    """Trains & evaluates given model/pipeline on train/test sets.
    Supports feature selection via pre_pipeline.
    Supports early stopping via val_idx.
    Optionally dumps resulting model & test set predictions into the models dir, and loads back by model name on the next call, to save time.
    Example of real OD:
        outlier_detector=Pipeline([("enc",ColumnTransformer(transformers=[('enc', ce.CatBoostEncoder(),['secid'])],remainder='passthrough')),("imp", SimpleImputer()), ("est", IsolationForest(contamination=0.01,n_estimators=500,n_jobs=-1))])

    train_idx etc indices must be fet to .iloc[] after, ie, be integer & unique

    group_ids:np.ndarray used to compute per-group AUCs (useful in recsys tasks).
    """

    clean_ram()
    columns = []
    best_iter = None

    if not skip_infinity_checks:
        if df is not None:
            ensure_no_infinity(df)
        else:
            if train_df is not None:
                ensure_no_infinity(train_df)

    if not custom_ice_metric:
        custom_ice_metric = compute_probabilistic_multiclass_error(nbins=nbins)

    if df is not None:
        real_drop_columns = [col for col in drop_columns + default_drop_columns if col in df.columns]
    elif train_df is not None:
        real_drop_columns = [col for col in drop_columns + default_drop_columns if col in train_df.columns]

    if type(model).__name__ == "Pipeline":
        model_obj = model.named_steps["est"]  # model.steps[-1]
    else:
        model_obj = model

    if model_obj is not None:
        if isinstance(model_obj, TransformedTargetRegressor):
            model_obj = model_obj.regressor
    model_type_name = type(model_obj).__name__ if model_obj is not None else ""

    if plot_file:
        if not plot_file.endswith(os.path.sep):
            plot_file = plot_file + "_"
        if model_name_prefix:
            plot_file = plot_file + slugify(model_name_prefix) + " "  # for ensembles
        if model_type_name:
            plot_file = plot_file + slugify(model_type_name) + " "

        plot_file = plot_file.strip()

    if model_name_prefix:
        model_name = model_name_prefix + model_name

    if model_type_name not in model_name:
        model_name = model_type_name + " " + model_name

    ensure_dir_exists(join(data_dir, models_subdir))
    model_file_name = join(data_dir, models_subdir, f"{model_name}.dump")

    if use_cache and exists(model_file_name):
        logger.info(f"Loading model from file {model_file_name}")
        model, *_, pre_pipeline = joblib.load(model_file_name)

    if fit_params is None:
        fit_params = {}
    else:
        fit_params = copy.copy(fit_params)  # to modify cat_features later and not affect next models

    train_od_idx, val_od_idx = None, None

    if train_target is None:
        train_target = target.iloc[train_idx] if isinstance(target, pd.Series) else target.gather(train_idx)
    if val_target is None and val_idx is not None:
        val_target = target.iloc[val_idx] if isinstance(target, pd.Series) else target.gather(val_idx)
    if test_target is None and test_idx is not None:
        test_target = target.iloc[test_idx] if isinstance(target, pd.Series) else target.gather(test_idx)

    if (df is not None) or (train_df is not None):
        if isinstance(df, pd.DataFrame):
            if train_df is None:
                train_df = df.iloc[train_idx].drop(columns=real_drop_columns)
            if val_df is None and val_idx is not None:
                val_df = df.iloc[val_idx].drop(columns=real_drop_columns)
        elif isinstance(df, pl.DataFrame):
            if train_df is None:
                train_df = df[train_idx].drop(real_drop_columns)
            if val_df is None and val_idx is not None:
                val_df = df[val_idx].drop(real_drop_columns)

        if not trainset_features_stats and isinstance(train_df, pd.DataFrame):
            if verbose:
                logger.info("Computing trainset_features_stats...")
            trainset_features_stats = get_trainset_features_stats(train_df)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # Place to inject Outlier Detector [OD] here!
        # -----------------------------------------------------------------------------------------------------------------------------------------------------

        if outlier_detector is not None:
            if verbose:
                logger.info("Fitting outlier detector...")
            outlier_detector.fit(train_df, train_target)
            # train
            is_inlier = outlier_detector.predict(train_df)
            train_od_idx = is_inlier == 1
            if train_od_idx.sum() < len(train_df):
                logger.info(f"Outlier rejection: received {len(train_df):_} train samples, kept {train_od_idx.sum():_}.")
                if train_idx is not None:
                    train_idx = train_idx[train_od_idx]
                    train_df = df.iloc[train_idx].drop(columns=real_drop_columns)
                    train_target = target.iloc[train_idx]
                else:
                    train_df = train_df.iloc[train_od_idx, :]
                    train_target = train_target.iloc[train_od_idx]

            # val
            if val_df is not None and od_val_set:
                is_inlier = outlier_detector.predict(val_df)
                val_od_idx = is_inlier == 1
                if val_od_idx.sum() < len(val_df):
                    logger.info(f"Outlier rejection: received {len(val_df):_} val samples, kept {val_od_idx.sum():_}.")
                    if val_idx is not None:
                        val_idx = val_idx[val_od_idx]
                        val_df = df.iloc[val_idx].drop(columns=real_drop_columns)
                        val_target = target.iloc[val_idx]
                    else:
                        val_df = val_df.iloc[val_od_idx, :]
                        val_target = val_target.iloc[val_od_idx]
                clean_ram()

    if model is not None and pre_pipeline:
        if use_cache and exists(model_file_name):
            train_df = pre_pipeline.transform(train_df, train_target)
        else:
            train_df = pre_pipeline.fit_transform(train_df, train_target, groups=groups)
        if val_df is not None:
            val_df = pre_pipeline.transform(val_df)
        clean_ram()

    if val_df is not None:
        # insert eval_set where needed

        if callback_params:
            if "callbacks" not in fit_params:
                if model_type_name not in XGBOOST_MODEL_TYPES:
                    fit_params["callbacks"] = []

        if model_type_name in LGBM_MODEL_TYPES:
            fit_params["eval_set"] = (val_df, val_target)
            if callback_params:
                es_callback = LightGBMCallback(**callback_params)
                fit_params.get("callbacks").append(es_callback)
        elif model_type_name in CATBOOST_MODEL_TYPES or model_type_name in XGBOOST_MODEL_TYPES:
            fit_params["eval_set"] = [
                (val_df, val_target),
            ]
            if callback_params:
                if model_type_name in CATBOOST_MODEL_TYPES:
                    es_callback = CatBoostCallback(**callback_params)
                    fit_params.get("callbacks").append(es_callback)
                elif model_type_name in XGBOOST_MODEL_TYPES:
                    es_callback = XGBoostCallback(**callback_params)
                    callbacks = model_obj.get_params().get("callbacks", [])
                    if callbacks is None:
                        callbacks = []
                    if es_callback not in callbacks:
                        callbacks.append(es_callback)
                    model_obj.set_params(callbacks=callbacks)

        elif model_type_name in TABNET_MODEL_TYPES:
            fit_params["eval_set"] = [
                (val_df.values, val_target.values),
            ]
        elif model_type_name in PYTORCH_MODEL_TYPES:
            fit_params["eval_set"] = (val_df, val_target)
        clean_ram()

    if model is not None and fit_params:
        if "cat_features" in fit_params:
            if isinstance(train_df, pd.DataFrame):
                fit_params["cat_features"] = [
                    col for col in fit_params["cat_features"] if col in train_df.head(5).select_dtypes(["category", "object"]).columns
                ]
            elif isinstance(train_df, pl.DataFrame):
                fit_params["cat_features"] = [col for col in fit_params["cat_features"] if col in train_df.head(5).select(pl.col(pl.Categorical)).columns]

    if model is not None:
        if (not use_cache) or (not exists(model_file_name)):
            if sample_weight is not None:
                if "sample_weight" in get_function_param_names(model_obj.fit):
                    if train_idx is not None:
                        fit_params["sample_weight"] = sample_weight.iloc[train_idx].values
                    else:
                        fit_params["sample_weight"] = sample_weight.values
            if verbose:
                logger.info(f"{model_name} training dataset shape: {train_df.shape}")

            if display_sample_size:
                display(train_df.head(display_sample_size).style.set_caption(f"{model_name} features head"))
                display(train_df.tail(display_sample_size).style.set_caption(f"{model_name} features tail"))

            if train_df is not None:

                report_title = f"Training {model_name} model on {train_df.shape[1]} feature(s)"  # textwrap.shorten("Hello world", width=10, placeholder="...")
                if show_feature_names:
                    report_title += ": " + ", ".join(train_df.columns.to_list())
                report_title += f", {len(train_df):_} records"

            if model_type_name in TABNET_MODEL_TYPES:
                train_df = train_df.values

            if fit_params and type(model).__name__ == "Pipeline":
                fit_params = prefix_dict_elems(fit_params, "est__")

            if use_hpt:

                import optuna

                def objective(trial):

                    param = {
                        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
                        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
                        "depth": trial.suggest_int("depth", 1, 12),
                        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
                    }

                    if param["bootstrap_type"] == "Bayesian":
                        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
                    elif param["bootstrap_type"] == "Bernoulli":
                        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

                    tune_model = model.copy()
                    tune_model.set_params(**param)

                    clean_ram()
                    tune_model.fit(train_df, train_target, **fit_params)
                    clean_ram()

                    temp_metrics = {}
                    columns = val_df.columns
                    tune_val_preds, tune_val_probs = report_model_perf(
                        targets=val_target,
                        columns=columns,
                        df=val_df.values if model_type_name in TABNET_MODEL_TYPES else val_df,
                        model_name="VAL " + model_name,
                        model=tune_model,
                        target_label_encoder=target_label_encoder,
                        preds=val_preds,
                        probs=val_probs,
                        figsize=figsize,
                        report_title="",
                        nbins=nbins,
                        print_report=False,
                        show_perf_chart=False,
                        show_fi=False,
                        fi_kwargs=fi_kwargs,
                        subgroups=subgroups,
                        subset_index=val_idx,
                        custom_ice_metric=custom_ice_metric,
                        custom_rice_metric=custom_rice_metric,
                        metrics=temp_metrics,
                    )
                    return temp_metrics[1]["class_robust_integral_error"]

                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=100, timeout=60 * 60)

                print("Number of finished trials: {}".format(len(study.trials)))

                print("Best trial:")
                trial = study.best_trial

                print("  Value: {}".format(trial.value))

                print("  Params: ", trial.params)
                model.set_params(**trial.params)

            clean_ram()
            if verbose:
                logger.info("Training the model...")

            try:
                model.fit(train_df, train_target, **fit_params)
            except Exception as e:
                try_again = False
                if "out of memory" in str(e):
                    if model_type_name in XGBOOST_MODEL_TYPES:
                        if model_obj.get_params().get("device") in ("gpu", "cuda"):
                            model_obj.set_params(device="cpu")
                            try_again = True
                    elif model_type_name in CATBOOST_MODEL_TYPES:
                        if model_obj.get_params().get("task_type") == "GPU":
                            model_obj.set_params(task_type="CPU")
                            try_again = True
                    elif model_type_name in LGBM_MODEL_TYPES:
                        if model_obj.get_params().get("device_type") in ("gpu", "cuda"):
                            model_obj.set_params(device_type="cpu")
                            try_again = True
                if try_again:
                    logger.warning(f"{model_type_name} experienced OOM on gpu, switching to cpu...")
                    clean_ram()
                    model.fit(train_df, train_target, **fit_params)
                else:
                    raise e

            clean_ram()

            model_name = model_name + "\n" + " ".join([f" trained on {get_human_readable_set_size(len(train_df))} rows", train_details])

            if model is not None:
                # get number of the best iteration
                try:
                    best_iter = get_model_best_iter(model_obj)
                    if best_iter:
                        print(f"es_best_iter: {best_iter:_}")
                        model_name = model_name + f" @iter={best_iter:_}"
                except Exception as e:
                    logger.warning(e)

    metrics = {"train": {}, "val": {}, "test": {}, "best_iter": best_iter}

    if compute_trainset_metrics or compute_valset_metrics or compute_testset_metrics:
        if verbose:
            logger.info("Computing model's performance...")
        if compute_trainset_metrics and (train_idx is not None or train_df is not None):
            if df is None and train_df is None:
                train_df = None
                columns = []
            else:
                columns = train_df.columns

            train_preds, train_probs = report_model_perf(
                targets=train_target,
                columns=columns,
                df=train_df.values if model_type_name in TABNET_MODEL_TYPES else train_df,
                model_name=model_name,
                model=model,
                target_label_encoder=target_label_encoder,
                preds=train_preds,
                probs=train_probs,
                figsize=figsize,
                report_title=" ".join(["TRAIN", train_details]),
                nbins=nbins,
                print_report=print_report,
                plot_file=plot_file + "_train" if plot_file else "",
                show_perf_chart=show_perf_chart,
                show_fi=show_fi and (test_df is None) and (val_df is None),
                fi_kwargs=fi_kwargs,
                subgroups=subgroups,
                subset_index=train_idx,
                custom_ice_metric=custom_ice_metric,
                custom_rice_metric=custom_rice_metric,
                metrics=metrics["train"],
                group_ids=group_ids[train_idx] if group_ids is not None else None,
            )

        if compute_valset_metrics and ((val_idx is not None and len(val_idx) > 0) or val_df is not None):
            if df is None and val_df is None:
                val_df = None
                columns = []
            else:
                columns = val_df.columns

            val_preds, val_probs = report_model_perf(
                targets=val_target,
                columns=columns,
                df=val_df.values if model_type_name in TABNET_MODEL_TYPES else val_df,
                model_name=model_name,
                model=model,
                target_label_encoder=target_label_encoder,
                preds=val_preds,
                probs=val_probs,
                figsize=figsize,
                report_title=" ".join(["VAL", val_details]),
                nbins=nbins,
                print_report=print_report,
                plot_file=plot_file + "_val" if plot_file else "",
                show_perf_chart=show_perf_chart,
                show_fi=show_fi and (test_idx is None and test_df is None),
                fi_kwargs=fi_kwargs,
                subgroups=subgroups,
                subset_index=val_idx,
                custom_ice_metric=custom_ice_metric,
                custom_rice_metric=custom_rice_metric,
                metrics=metrics["val"],
                group_ids=group_ids[val_idx] if group_ids is not None else None,
            )

        if compute_testset_metrics and ((test_idx is not None and len(test_idx) > 0) or test_df is not None):
            if (df is not None) or (test_df is not None):

                del train_df
                clean_ram()

                if test_df is None:
                    if isinstance(df, pd.DataFrame):
                        test_df = df.iloc[test_idx].drop(columns=real_drop_columns)
                    elif isinstance(df, pl.DataFrame):
                        test_df = df[test_idx].drop(real_drop_columns)

                if test_target is None:
                    test_target = target.iloc[test_idx] if isinstance(target, pd.Series) else target.gather(test_idx)

                if model is not None and pre_pipeline:
                    test_df = pre_pipeline.transform(test_df)
                if model_type_name in TABNET_MODEL_TYPES:
                    test_df = test_df.values
                columns = test_df.columns
            else:
                columns = []
                test_df = None

            test_preds, test_probs = report_model_perf(
                targets=test_target,
                columns=columns,
                df=test_df,
                model_name=model_name,
                model=model,
                target_label_encoder=target_label_encoder,
                preds=test_preds,
                probs=test_probs,
                figsize=figsize,
                report_title=" ".join(["TEST", test_details]),
                nbins=nbins,
                print_report=print_report,
                plot_file=plot_file + "_test" if plot_file else "",
                show_perf_chart=show_perf_chart,
                show_fi=show_fi,
                fi_kwargs=fi_kwargs,
                subgroups=subgroups,
                subset_index=test_idx,
                custom_ice_metric=custom_ice_metric,
                custom_rice_metric=custom_rice_metric,
                metrics=metrics["test"],
                group_ids=group_ids[test_idx] if group_ids is not None else None,
            )

            if include_confidence_analysis:
                """Separate analysis: having original dataset, and test predictions made by a trained model,
                find what original factors are the most discriminative regarding prediction accuracy. for that,
                training a meta-model on test set could do it. use original features, as targets use prediction-ground truth,
                train a regression boosting & check its feature importances."""

                # for (any, even multiclass) classification, targets are probs of ground truth classes
                if test_df is not None:
                    if verbose:
                        logger.info("Runnig confidence analysis on teh test set...")
                    confidence_model = CatBoostRegressor(
                        verbose=0, eval_fraction=0.1, task_type=("GPU" if CUDA_IS_AVAILABLE else "CPU"), **confidence_model_kwargs
                    )

                    if model_type_name == type(confidence_model).__name__:
                        fit_params_copy = copy.copy(fit_params)
                        if "eval_set" in fit_params_copy:
                            del fit_params_copy["eval_set"]
                    else:
                        fit_params_copy = {}

                    if "cat_features" not in fit_params_copy:
                        fit_params_copy["cat_features"] = test_df.head().select_dtypes("category").columns.tolist()

                    fit_params_copy["plot"] = False

                    clean_ram()
                    confidence_model.fit(test_df, test_probs[np.arange(test_probs.shape[0]), test_target], **fit_params_copy)
                    clean_ram()

                    if confidence_analysis_use_shap:
                        explainer = shap.TreeExplainer(confidence_model)
                        shap_values = explainer(test_df)
                        shap.plots.beeswarm(
                            shap_values,
                            max_display=confidence_analysis_max_features,
                            color=plt.get_cmap(confidence_analysis_cmap),
                            alpha=confidence_analysis_alpha,
                            color_bar_label=confidence_analysis_ylabel,
                            show=False,
                        )
                        plt.xlabel(confidence_analysis_title)
                        plt.show()
                    else:
                        plot_model_feature_importances(
                            model=confidence_model,
                            columns=test_df.columns,
                            model_name=confidence_analysis_title,
                            num_factors=confidence_analysis_max_features,
                            figsize=(figsize[0] * 0.7, figsize[1] / 2),
                        )

    clean_ram()

    return SimpleNamespace(
        model=model,
        test_preds=test_preds,
        test_probs=test_probs,
        test_target=test_target,
        val_preds=val_preds,
        val_probs=val_probs,
        val_target=val_target,
        train_preds=train_preds,
        train_probs=train_probs,
        train_target=train_target,
        metrics=metrics,
        columns=columns,
        pre_pipeline=pre_pipeline,
        outlier_detector=outlier_detector,
        train_od_idx=train_od_idx,
        val_od_idx=val_od_idx,
        trainset_features_stats=trainset_features_stats,
    )


def report_model_perf(
    targets: Union[np.ndarray, pd.Series],
    columns: Sequence,
    model_name: str,
    model: ClassifierMixin,
    subgroups: dict = None,
    subset_index: np.ndarray = None,
    report_ndigits: int = 4,
    figsize: tuple = (15, 5),
    report_title: str = "",
    use_weights: bool = True,
    calib_report_ndigits: int = 2,
    verbose: bool = False,
    classes: Sequence = [],
    preds: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    target_label_encoder: Optional[LabelEncoder] = None,
    nbins: int = 10,
    print_report: bool = True,
    show_perf_chart: bool = True,
    show_fi: bool = True,
    fi_kwargs: dict = {},
    plot_file: str = "",
    custom_ice_metric: Callable = None,
    custom_rice_metric: Callable = None,
    metrics: dict = None,
    group_ids: np.ndarray = None,
):

    if is_classifier(model) or (model is None and probs is not None):
        preds, probs = report_probabilistic_model_perf(
            targets=targets,
            columns=columns,
            model_name=model_name,
            model=model,
            subgroups=subgroups,
            subset_index=subset_index,
            report_ndigits=report_ndigits,
            figsize=figsize,
            report_title=report_title,
            use_weights=use_weights,
            calib_report_ndigits=calib_report_ndigits,
            verbose=verbose,
            classes=classes,
            preds=preds,
            probs=probs,
            df=df,
            target_label_encoder=target_label_encoder,
            nbins=nbins,
            print_report=print_report,
            show_perf_chart=show_perf_chart,
            plot_file=plot_file,
            custom_ice_metric=custom_ice_metric,
            custom_rice_metric=custom_rice_metric,
            metrics=metrics,
            group_ids=group_ids,
        )
    else:

        preds, probs = report_regression_model_perf(
            targets=targets,
            columns=columns,
            model_name=model_name,
            model=model,
            subgroups=subgroups,
            subset_index=subset_index,
            report_ndigits=report_ndigits,
            figsize=figsize,
            report_title=report_title,
            verbose=verbose,
            preds=preds,
            df=df,
            print_report=print_report,
            show_perf_chart=show_perf_chart,
            plot_file=plot_file,
            metrics=metrics,
        )
    if show_fi:
        nfeatures = f"{len(columns):_}F/" if (columns is not None and len(columns) > 0) else ""
        feature_importances = plot_model_feature_importances(
            model=model,
            columns=columns,
            model_name=(report_title + " " + model_name + f" [{nfeatures}{get_human_readable_set_size(len(preds))} rows]").strip(),
            plot_file=plot_file + "_fiplot.png" if plot_file else "",
            **fi_kwargs,
        )
        if metrics is not None:
            metrics.update({"feature_importances": feature_importances})

    return preds, probs


def report_regression_model_perf(
    targets: Union[np.ndarray, pd.Series],
    columns: Sequence,
    model_name: str,
    model: RegressorMixin,
    subgroups: dict = None,
    subset_index: np.ndarray = None,
    report_ndigits: int = 4,
    figsize: tuple = (15, 5),
    report_title: str = "",
    verbose: bool = False,
    preds: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    print_report: bool = True,
    show_perf_chart: bool = True,
    plot_file: str = "",
    plot_marker: str = "o",
    plot_sample_size: int = 500,
    metrics: dict = None,
):
    """Detailed performance report (usually on a test set)."""

    if preds is None:
        preds = model.predict(df)

    if isinstance(targets, pd.Series):
        targets = targets.values

    MAE = mean_absolute_error(y_true=targets, y_pred=preds)
    MaxError = max_error(y_true=targets, y_pred=preds)
    R2 = r2_score(y_true=targets, y_pred=preds)
    # MAPE = mean_absolute_percentage_error(y_true=targets, y_pred=preds)
    RMSE = root_mean_squared_error(y_true=targets, y_pred=preds)

    current_metrics = dict(
        MAE=MAE,
        MaxError=MaxError,
        R2=R2,
        # MAPE=MAPE,
        RMSE=RMSE,
    )
    if metrics is not None:
        metrics.update(current_metrics)

    if show_perf_chart or plot_file:
        title = report_title + " " + model_name
        nfeatures = f"{len(columns):_}F/" if (columns is not None and len(columns) > 0) else ""
        title += f" [{nfeatures}{get_human_readable_set_size(len(targets))} rows]" + "\n"

        title += f" MAE={MAE:.{report_ndigits}f}"
        title += f" RMSE={RMSE:.{report_ndigits}f}"
        title += f" MaxError={MaxError:.{report_ndigits}f}"
        # title += f" MAPE={MAPE*100:.{report_ndigits//2}f}%"
        title += f" R2={R2:.{report_ndigits}f}"

        np.random.seed(42)
        idx = np.random.choice(np.arange(len(preds)), size=min(plot_sample_size, len(preds)), replace=False)
        idx = idx[np.argsort(preds[idx])]

        fig = plt.figure(figsize=figsize)
        plt.scatter(preds[idx], targets[idx], marker=plot_marker, alpha=0.3)
        plt.plot(preds[idx], preds[idx], linestyle="--", color="green", label="Perfect fit")

        plt.xlabel("Predictions")
        plt.ylabel("True values")
        plt.title(title)

        if plot_file:
            fig.savefig(plot_file)

        if show_perf_chart:
            plt.ion()
            plt.show()
        else:
            plt.close(fig)

    if print_report:
        print(report_title + " " + model_name)
        print(f"MAE: {MAE:.{report_ndigits}f}")
        print(f"RMSE: {RMSE:.{report_ndigits}f}")
        print(f"MaxError: {MaxError:.{report_ndigits}f}")
        # print(f"MAPE: {MAPE*100:.{report_ndigits//2}f}%")
        print(f"R2: {R2:.{report_ndigits}f}")

    if subgroups:
        robustness_report = compute_robustness_metrics(
            subgroups=subgroups,
            subset_index=subset_index,
            y_true=targets,
            y_pred=preds,
            metrics={"MAE": mean_absolute_error, "RMSE": root_mean_squared_error},
            metrics_higher_is_better={"MAE": False, "RMSE": False},
        )
        if robustness_report is not None:
            if print_report:
                display(robustness_report)
            if metrics is not None:
                metrics.update(dict(robustness_report=robustness_report))

    return preds, None


def report_probabilistic_model_perf(
    targets: Union[np.ndarray, pd.Series],
    columns: Sequence,
    model_name: str,
    model: ClassifierMixin,
    subgroups: dict = None,
    subset_index: np.ndarray = None,
    report_ndigits: int = 4,
    figsize: tuple = (15, 5),
    report_title: str = "",
    use_weights: bool = True,
    calib_report_ndigits: int = 2,
    verbose: bool = False,
    classes: Sequence = [],
    preds: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    target_label_encoder: Optional[LabelEncoder] = None,
    nbins: int = 10,
    print_report: bool = True,
    show_perf_chart: bool = True,
    plot_file: str = "",
    custom_ice_metric: Callable = None,
    custom_rice_metric: Callable = None,
    metrics: dict = None,
    group_ids: np.ndarray = None,
):
    """Detailed performance report (usually on a test set)."""

    if probs is None:
        probs = model.predict_proba(df)

    if preds is None:
        preds = np.argmax(probs, axis=1)  # need this for ensembling to work. preds = model.predict(df) will do the same but more labour.

    if isinstance(targets, pd.Series):
        targets = targets.values

    brs = []
    calibs = []
    pr_aucs = []
    roc_aucs = []
    integral_errors = []
    log_losses = []
    robust_integral_errors = []

    integral_error = custom_ice_metric(y_true=targets, y_score=probs) if custom_ice_metric else 0.0
    if custom_rice_metric and custom_rice_metric != custom_ice_metric:
        robust_integral_error = custom_rice_metric(y_true=targets, y_score=probs)

    if not classes:
        if model is not None:
            classes = model.classes_
        elif target_label_encoder:
            classes = np.arange(len(target_label_encoder.classes_)).tolist()
        else:
            classes = np.unique(targets)

    true_classes = []
    for class_id, class_name in enumerate(classes):
        if str(class_name).isnumeric() and target_label_encoder:
            str_class_name = str(target_label_encoder.classes_[class_name])
        else:
            str_class_name = str(class_name)
        true_classes.append(str_class_name)

        if len(classes) == 2 and class_id == 0:
            continue

        y_true, y_score = (targets == class_name), probs[:, class_id]
        if isinstance(y_true, pl.Series):
            y_true = y_true.to_numpy()

        title = report_title + " " + model_name
        if len(classes) != 2:
            title += "-" + str_class_name

        class_integral_error = custom_ice_metric(y_true=y_true, y_score=y_score) if custom_ice_metric else 0.0
        nfeatures = f"{len(columns):_}F/" if (columns is not None and len(columns) > 0) else ""
        title += f" [{nfeatures}{get_human_readable_set_size(len(y_true))} rows]"
        if custom_rice_metric and custom_rice_metric != custom_ice_metric:
            class_robust_integral_error = custom_rice_metric(y_true=y_true, y_score=y_score)
            title += f", RICE={class_robust_integral_error:.{calib_report_ndigits}f}"

        brier_loss, calibration_mae, calibration_std, calibration_coverage, roc_auc, pr_auc, ice, ll, *_, metrics_string, fig = fast_calibration_report(
            y_true=y_true,
            y_pred=y_score,
            use_weights=use_weights,
            nbins=nbins,
            group_ids=group_ids,
            #
            title=title,
            figsize=figsize,
            plot_file=plot_file + "_perfplot.png" if plot_file else "",
            show_plots=show_perf_chart,
            ndigits=calib_report_ndigits,
            verbose=verbose,
        )

        if print_report:

            calibs.append(
                f"\t{str_class_name}: MAE{'W' if use_weights else ''}={calibration_mae * 100:.{calib_report_ndigits}f}%, STD={calibration_std * 100:.{calib_report_ndigits}f}%, COV={calibration_coverage * 100:.0f}%"
            )
            pr_aucs.append(f"{str_class_name}={pr_auc:.{report_ndigits}f}")
            roc_aucs.append(f"{str_class_name}={roc_auc:.{report_ndigits}f}")
            brs.append(f"{str_class_name}={brier_loss * 100:.{report_ndigits}f}%")
            integral_errors.append(f"{str_class_name}={ice:.{report_ndigits}f}")
            log_losses.append(f"{str_class_name}={ll:.{report_ndigits}f}")
            if custom_rice_metric and custom_rice_metric != custom_ice_metric:
                robust_integral_errors.append(f"{str_class_name}={class_robust_integral_error:.{report_ndigits}f}")

        if metrics is not None:
            class_metrics = dict(
                roc_auc=roc_auc,
                pr_auc=pr_auc,
                calibration_mae=calibration_mae,
                calibration_std=calibration_std,
                brier_loss=brier_loss,
                log_loss=ll,
                ice=ice,
                class_integral_error=class_integral_error,
            )
            if custom_rice_metric and custom_rice_metric != custom_ice_metric:
                class_metrics["class_robust_integral_error"] = class_robust_integral_error
            metrics.update({class_id: class_metrics})

    if print_report:

        print(report_title + " " + model_name)
        print(classification_report(targets, preds, zero_division=0, digits=report_ndigits))
        print(f"ROC AUCs: {', '.join(roc_aucs)}")
        print(f"PR AUCs: {', '.join(pr_aucs)}")
        print(f"CALIBRATIONs: \n{', '.join(calibs)}")
        print(f"BRIER LOSSes: \n\t{', '.join(brs)}")
        print(f"LOG_LOSSes: \n\t{', '.join(log_losses)}")
        print(f"ICEs: \n\t{', '.join(integral_errors)}")
        if custom_ice_metric != custom_rice_metric:
            print(f"RICEs: \n\t{', '.join(robust_integral_errors)}")

        print(f"TOTAL INTEGRAL ERROR: {integral_error:.4f}")
        if custom_rice_metric and custom_rice_metric != custom_ice_metric:
            print(f"TOTAL ROBUST INTEGRAL ERROR: {robust_integral_error:.4f}")

    if subgroups:

        subgroups_metrics = {"ICE": custom_ice_metric}
        metrics_higher_is_better = {"ICE": False}

        if probs.shape[1] == 2:
            subgroups_metrics["ROC AUC"] = fast_roc_auc
            metrics_higher_is_better["ROC AUC"] = True

        robustness_report = compute_robustness_metrics(
            subgroups=subgroups,
            subset_index=subset_index,
            y_true=targets,
            y_pred=probs,
            metrics=subgroups_metrics,
            metrics_higher_is_better=metrics_higher_is_better,
        )
        if robustness_report is not None:
            if print_report:
                display(robustness_report.style.set_caption("ML perf robustness by group"))
            if metrics is not None:
                metrics.update(dict(robustness_report=robustness_report))

    return preds, probs


def get_model_feature_importances(model: object, columns: Sequence, return_df: bool = False):
    if isinstance(model, Pipeline):
        model = model.steps[-1][1]
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        if model.coef_.ndim == 1:
            feature_importances = model.coef_
        else:
            feature_importances = model.coef_[-1, :]
    else:
        feature_importances = None

    if feature_importances is not None:

        if return_df:
            feature_importances = pd.DataFrame({"feature": columns, "importance": feature_importances})

    return feature_importances


def plot_model_feature_importances(
    model: object,
    columns: Sequence,
    model_name: str = None,
    num_factors: int = 40,
    figsize: tuple = (15, 10),
    positive_fi_only: bool = False,
    plot_file: str = "",
) -> np.ndarray:

    feature_importances = get_model_feature_importances(model=model, columns=columns)

    if feature_importances is not None:
        try:
            plot_feature_importance(
                feature_importances=feature_importances,
                columns=columns,
                kind=model_name,
                figsize=figsize,
                plot_file=plot_file,
                positive_fi_only=positive_fi_only,
                n=num_factors,
            )
        except Exception:
            logger.warning("Could not plot feature importances. Maybe data shape is changed within a pipeline?")

        return feature_importances


def get_sample_weights_by_recency(date_series: pd.Series, min_weight: float = 1.0, weight_drop_per_year: float = 0.1) -> np.ndarray:

    span = (date_series.max() - date_series.min()).days
    max_drop = np.log(span) * weight_drop_per_year

    sample_weight = min_weight + max_drop - np.log((date_series.max() - date_series).dt.days) * weight_drop_per_year

    return sample_weight


def configure_training_params(
    df: pd.DataFrame = None,
    train_df: pd.DataFrame = None,
    test_df: pd.DataFrame = None,
    val_df: pd.DataFrame = None,
    target: pd.Series = None,
    train_target: pd.Series = None,
    test_target: pd.Series = None,
    val_target: pd.Series = None,
    train_idx: np.ndarray = None,
    val_idx: np.ndarray = None,
    test_idx: np.ndarray = None,
    robustness_features: Sequence = [],
    target_label_encoder: object = None,
    sample_weight: np.ndarray = None,
    prefer_gpu_configs: bool = True,
    use_robust_eval_metric: bool = False,
    nbins: int = 10,
    use_regression: bool = False,
    cont_nbins: int = 6,
    verbose: bool = True,
    rfecv_model_verbose: bool = True,
    prefer_cpu_for_lightgbm: bool = True,
    prefer_cpu_for_xgboost: bool = False,
    xgboost_verbose: Union[int, bool] = False,
    cb_fit_params: dict = {},  # cb_fit_params=dict(embedding_features=['embeddings'])
    prefer_calibrated_classifiers: bool = True,
    default_regression_scoring: dict = None,
    default_classification_scoring: dict = None,
    train_details: str = "",
    val_details: str = "",
    test_details: str = "",
    group_ids: np.ndarray = None,
    model_name: str = "",
    common_params: dict = None,
    config_params: dict = None,
    metamodel_func: callable = None,
):
    if metamodel_func is None:
        metamodel_func = lambda x: x

    if default_regression_scoring is None:
        default_regression_scoring = dict(score_func=mean_absolute_error, needs_proba=False, needs_threshold=False, greater_is_better=False)

    if default_classification_scoring is None:
        default_classification_scoring = dict(score_func=fast_roc_auc, needs_proba=True, needs_threshold=False, greater_is_better=True)

    if common_params is None:
        common_params = {}
    if config_params is None:
        config_params = {}

    for next_df in (df, train_df):
        if next_df is not None:
            if isinstance(next_df, pd.DataFrame):
                cat_features = next_df.head().select_dtypes(("category", "object")).columns.tolist()
            elif isinstance(next_df, pl.DataFrame):
                cat_features = next_df.head().select(pl.col(pl.Categorical)).columns
            break

    if cat_features:
        for next_df in (df, train_df, val_df, test_df):
            if next_df is not None and isinstance(next_df, pd.DataFrame):
                prepare_df_for_catboost(
                    df=next_df,
                    cat_features=cat_features,
                )

    # ensure_dataframe_float32_convertability(df)

    if robustness_features:
        for next_df in (df, train_df):
            if next_df is not None:
                subgroups = create_robustness_subgroups(next_df, features=robustness_features, cont_nbins=cont_nbins)
                break
    else:
        subgroups = None

    if use_robust_eval_metric and subgroups is not None:
        indexed_subgroups = create_robustness_subgroups_indices(
            subgroups=subgroups, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, group_weights={}, cont_nbins=cont_nbins
        )
    else:
        indexed_subgroups = None

    if not use_regression:
        if "catboost_custom_classif_metrics" not in config_params:
            nlabels = len(np.unique(target))
            if nlabels > 2:
                catboost_custom_classif_metrics = ["AUC", "PRAUC"]
            else:
                catboost_custom_classif_metrics = ["AUC", "PRAUC", "BrierScore"]
            config_params["catboost_custom_classif_metrics"] = catboost_custom_classif_metrics

    cpu_configs = get_training_configs(has_gpu=False, subgroups=indexed_subgroups, **config_params)
    gpu_configs = get_training_configs(has_gpu=None, subgroups=indexed_subgroups, **config_params)

    data_fits_gpu_ram = True
    from pyutilz.pandaslib import get_df_memory_consumption
    from pyutilz.system import compute_total_gpus_ram, get_gpuinfo_gpu_info

    configs = gpu_configs if (prefer_gpu_configs and data_fits_gpu_ram) else cpu_configs

    common_params = dict(
        nbins=nbins,
        subgroups=subgroups,
        sample_weight=sample_weight,
        df=df,
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        target=target,
        train_target=train_target,
        test_target=test_target,
        val_target=val_target,
        train_idx=train_idx,
        test_idx=test_idx,
        val_idx=val_idx,
        target_label_encoder=target_label_encoder,
        custom_ice_metric=configs.integral_calibration_error,
        custom_rice_metric=configs.final_integral_calibration_error,
        train_details=train_details,
        val_details=val_details,
        test_details=test_details,
        group_ids=group_ids,
        model_name=model_name,
        **common_params,
    )

    common_cb_params = dict(
        model=(
            metamodel_func(CatBoostRegressor(**configs.CB_REGR))
            if use_regression
            else CatBoostClassifier(**(configs.CB_CALIB_CLASSIF if prefer_calibrated_classifiers else configs.CB_CLASSIF))
        ),
        fit_params=dict(plot=verbose, cat_features=cat_features, **cb_fit_params),
    )  # TransformedTargetRegressor(CatBoostRegressor(**configs.CB_REGR),transformer=PowerTransformer())

    if prefer_cpu_for_xgboost:
        common_xgb_params = dict(
            model=(
                metamodel_func(XGBRegressor(**cpu_configs.XGB_GENERAL_PARAMS))
                if use_regression
                else XGBClassifier(**(cpu_configs.XGB_CALIB_CLASSIF if prefer_calibrated_classifiers else cpu_configs.XGB_GENERAL_CLASSIF))
            ),
            fit_params=dict(verbose=xgboost_verbose),
        )
    else:
        common_xgb_params = dict(
            model=(
                metamodel_func(XGBRegressor(**configs.XGB_GENERAL_PARAMS))
                if use_regression
                else XGBClassifier(**(configs.XGB_CALIB_CLASSIF if prefer_calibrated_classifiers else configs.XGB_GENERAL_CLASSIF))
            ),
            fit_params=dict(verbose=xgboost_verbose),
        )

    if prefer_cpu_for_lightgbm:
        common_lgb_params = dict(
            model=metamodel_func(LGBMRegressor(**cpu_configs.LGB_GENERAL_PARAMS)) if use_regression else LGBMClassifier(**cpu_configs.LGB_GENERAL_PARAMS),
            fit_params=(dict(eval_metric=cpu_configs.lgbm_integral_calibration_error) if prefer_calibrated_classifiers else {}),
        )
    else:
        common_lgb_params = dict(
            model=metamodel_func(LGBMRegressor(**configs.LGB_GENERAL_PARAMS)) if use_regression else LGBMClassifier(**configs.LGB_GENERAL_PARAMS),
            fit_params=(dict(eval_metric=configs.lgbm_integral_calibration_error) if prefer_calibrated_classifiers else {}),
        )

    rfecv_params = configs.COMMON_RFECV_PARAMS.copy()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Setting up RFECV
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    if use_regression:
        rfecv_scoring = make_scorer(**default_regression_scoring)
    else:
        if prefer_calibrated_classifiers:

            def fs_and_hpt_integral_calibration_error(*args, **kwargs):
                return configs.fs_and_hpt_integral_calibration_error(*args, **kwargs, verbose=rfecv_model_verbose)

            rfecv_scoring = make_scorer(
                score_func=fs_and_hpt_integral_calibration_error,
                needs_proba=True,
                needs_threshold=False,
                greater_is_better=False,
            )
        else:
            rfecv_scoring = make_scorer(**default_classification_scoring)

    cb_rfecv = RFECV(
        estimator=(
            metamodel_func(CatBoostRegressor(**configs.CB_REGR))
            if use_regression
            else CatBoostClassifier(**(configs.CB_CALIB_CLASSIF if prefer_calibrated_classifiers else configs.CB_CLASSIF))
        ),
        fit_params=dict(plot=rfecv_model_verbose > 1),
        cat_features=cat_features,
        scoring=rfecv_scoring,
        **rfecv_params,
    )

    if prefer_cpu_for_lightgbm:
        lgb_fit_params = dict(eval_metric=cpu_configs.lgbm_integral_calibration_error) if prefer_calibrated_classifiers else {}
    else:
        lgb_fit_params = dict(eval_metric=cpu_configs.lgbm_integral_calibration_error) if prefer_calibrated_classifiers else {}

    lgb_rfecv = RFECV(
        estimator=metamodel_func(LGBMRegressor(**configs.LGB_GENERAL_PARAMS)) if use_regression else LGBMClassifier(**configs.LGB_GENERAL_PARAMS),
        fit_params=lgb_fit_params,
        cat_features=cat_features,
        scoring=rfecv_scoring,
        **rfecv_params,
    )

    xgb_rfecv = RFECV(
        estimator=(
            metamodel_func(XGBRegressor(**configs.XGB_GENERAL_PARAMS))
            if use_regression
            else XGBClassifier(**(configs.XGB_CALIB_CLASSIF if prefer_calibrated_classifiers else configs.XGB_GENERAL_CLASSIF))
        ),
        fit_params=dict(verbose=False),
        cat_features=cat_features,
        scoring=rfecv_scoring,
        **rfecv_params,
    )

    return common_params, common_cb_params, common_lgb_params, common_xgb_params, cb_rfecv, lgb_rfecv, xgb_rfecv, cpu_configs, gpu_configs


def post_calibrate_model(
    original_model: object,
    target_series: pd.Series,
    target_label_encoder: object,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    configs: dict,
    calib_set_size: int = 2000,
    nbins: int = 10,
    show_val: bool = False,
    meta_model: object = None,
    **fit_params,
):

    if meta_model is None:
        meta_model = CatBoostClassifier(
            iterations=3000,
            verbose=False,
            has_time=False,
            learning_rate=0.2,
            eval_fraction=0.1,
            task_type="GPU",
            early_stopping_rounds=400,
            eval_metric=ICE(metric=configs.integral_calibration_error, higher_is_better=False),
            custom_metric="AUC",
        )
    model, test_preds, test_probs, val_preds, val_probs, columns, pre_pipeline, metrics = original_model

    meta_model.fit(test_probs[:calib_set_size, 1].reshape(-1, 1), target_series.iloc[test_idx].values[:calib_set_size], **fit_params)

    if show_val:
        meta_val_probs = meta_model.predict_proba(val_probs[:, 1].reshape(-1, 1))
        _ = report_model_perf(
            targets=target_series.iloc[val_idx],
            columns=columns,
            df=None,
            model_name="VAL",
            model=None,
            target_label_encoder=target_label_encoder,
            preds=val_preds,
            probs=val_probs,
            report_title="",
            nbins=nbins,
            print_report=False,
            show_fi=False,
            custom_ice_metric=configs.integral_calibration_error,
        )
        _ = report_model_perf(
            targets=target_series.iloc[val_idx],
            columns=columns,
            df=None,
            model_name="VAL fixed",
            model=None,
            target_label_encoder=target_label_encoder,
            preds=val_preds,
            probs=meta_val_probs,
            report_title="",
            nbins=nbins,
            print_report=False,
            show_fi=False,
            custom_ice_metric=configs.integral_calibration_error,
        )

    meta_test_probs = meta_model.predict_proba(test_probs[:, 1].reshape(-1, 1))

    _ = report_model_perf(
        targets=target_series.iloc[test_idx],
        columns=columns,
        df=None,
        model_name="TEST original",
        model=None,
        target_label_encoder=target_label_encoder,
        preds=test_preds,
        probs=test_probs,
        report_title="",
        nbins=nbins,
        print_report=False,
        show_fi=False,
        custom_ice_metric=configs.integral_calibration_error,
    )

    _ = report_model_perf(
        targets=target_series.iloc[test_idx].values[calib_set_size:],
        columns=columns,
        df=None,
        model_name="TEST fixed ",
        model=None,
        target_label_encoder=target_label_encoder,
        preds=(meta_test_probs[calib_set_size:, 1] > 0.5).astype(int),
        probs=meta_test_probs[calib_set_size:, :],
        report_title="",
        nbins=nbins,
        print_report=True,
        show_fi=False,
        custom_ice_metric=configs.integral_calibration_error,
    )

    _ = report_model_perf(
        targets=target_series.iloc[test_idx].values[:calib_set_size],
        columns=columns,
        df=None,
        model_name="TEST fixed lucky",
        model=None,
        target_label_encoder=target_label_encoder,
        preds=(meta_test_probs[:calib_set_size:, 1] > 0.5).astype(int),
        probs=meta_test_probs[:calib_set_size, :],
        report_title="",
        nbins=nbins,
        print_report=True,
        show_fi=False,
        custom_ice_metric=configs.integral_calibration_error,
    )

    return model, test_preds, meta_test_probs, val_preds, meta_val_probs, columns, pre_pipeline, metrics


def create_ts_train_val_test_split(
    df: pd.DataFrame,
    dates: pd.Series,
    consecutive_val_ndays: int = 30,  # ts consecutive last val days
    scattered_val_ndays: int = 0,  # ts scattered val days
    test_ndays: int = 60,
):
    """Produce train-val-test split."""

    all_dates = sorted(dates.unique())
    print(f"{len(all_dates):_} distinct days")

    val_dates = all_dates[-(consecutive_val_ndays + test_ndays) : -test_ndays]
    if scattered_val_ndays:
        val_dates += np.random.choice(all_dates[: -(consecutive_val_ndays + test_ndays)], scattered_val_ndays).tolist()
    test_dates = all_dates[-test_ndays:]
    train_dates = set(all_dates) - set(val_dates) - set(test_dates)

    train_idx, val_idx, test_idx = (
        df.index[dates.isin(train_dates)],
        df.index[dates.isin(val_dates)],
        df.index[dates.isin(test_dates)],
    )

    logger.info(
        f"Training on {len(train_dates):_} dates, vaildating on {len(val_dates):_} dates [{pd.to_datetime(val_dates[0]).date()}->{pd.to_datetime(val_dates[-1]).date()}], testing on {len(test_dates):_} dates [{pd.to_datetime(test_dates[0]).date()}->{pd.to_datetime(test_dates[-1]).date()}]"
    )

    return train_idx, val_idx, test_idx


def save_mlframe_model(model: object, file: str, zstd_kwargs: dict = None) -> bool:
    if zstd_kwargs is None:
        zstd_kwargs = dict(level=4, write_checksum=True, write_content_size=True, threads=-1)
    try:
        with open(file, "wb") as f:
            compressor = zstd.ZstdCompressor(**zstd_kwargs)
            with compressor.stream_writer(f) as zf:
                dill.dump(model, zf)
        return True
    except Exception as e:
        logger.error(f"Could not save model to file {file}: {e}")


def load_mlframe_model(file: str) -> object:
    try:
        with open(file, "rb") as f:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(f) as zf:
                model = dill.load(zf)
        return model
    except Exception as e:
        logger.error(f"Could not load model from file {file}: {e}")


def clean_mlframe_model(model: SimpleNamespace) -> SimpleNamespace:
    """Delete extra fields from the model's namespace, to be RAM efficient."""
    for (
        col
    ) in "test_preds test_probs test_target val_preds val_probs val_target train_preds train_probs train_target train_od_idx val_od_idx trainset_features_stats".split():
        if hasattr(model, col):
            delattr(model, col)
    return model


def load_production_models(
    models_dir: str, target_name: str, featureset_name: str, task_type=TargetTypes.BINARY_CLASSIFICATION, directions: list = [], clean_models: bool = True
) -> dict:
    """Reads models from disk, instantiates SHAP explainers where possible."""

    models = {}
    explainers = {}
    postcalibrators = {}

    from mlframe.ensembling import SIMPLE_ENSEMBLING_METHODS

    logger.info(f"Loading trained production {featureset_name} {task_type} models for target {target_name}...")

    featureset_dir = join(models_dir, target_name, featureset_name, task_type)
    trainset_features_stats = None
    for direction in tqdmu(directions, desc="direction", leave=False):

        models[direction] = {}
        explainers[direction] = {}
        postcalibrators[direction] = {}

        final_models_dir = join(featureset_dir, direction)

        for fpath in glob.glob(join(final_models_dir, f"*_model.dump")):
            base_model_name = basename(fpath)

            model = load_mlframe_model(fpath)
            if trainset_features_stats is None:
                trainset_features_stats = model.trainset_features_stats
            if clean_models:
                clean_mlframe_model(model)
            model_name = base_model_name.replace(f"_model.dump", "")
            models[direction][model_name] = model

            calib_fpath = fpath.replace("_model.dump", "_model_postcalibrator.dump")
            if exists(calib_fpath):
                postcalibrator = joblib.load(calib_fpath)
                postcalibrators[direction][model_name] = postcalibrator

            explainer = None
            try:
                explainer = shap.TreeExplainer(model.model)
                explainers[direction][model_name] = explainer
            except Exception as e:
                pass

        # ens calibrators
        for ensembling_method in SIMPLE_ENSEMBLING_METHODS:
            ens_name = f"ens_{ensembling_method}"
            calib_fpath = join(final_models_dir, f"{ens_name}_postcalibrator.dump")
            if exists(calib_fpath):
                postcalibrator = joblib.load(calib_fpath)
                postcalibrators[direction][ens_name] = postcalibrator

        logger.info(f"Loaded {len(models[direction]):_} production {direction} model(s): {', '.join(models[direction].keys())}")

    return models, explainers, postcalibrators, trainset_features_stats


# Training Suite


def make_train_test_split(df: pd.DataFrame, timestamps: pd.Series, test_size: float = 0.1, val_size: float = 0.1, shuffle: bool = False) -> tuple:

    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=test_size, shuffle=shuffle)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size, shuffle=shuffle)

    train_details: str = f"{timestamps.iloc[train_idx].min():%Y-%m-%d}/{timestamps.iloc[train_idx].max():%Y-%m-%d}"
    val_details: str = f"{timestamps.iloc[val_idx].min():%Y-%m-%d}/{timestamps.iloc[val_idx].max():%Y-%m-%d}"
    test_details: str = f"{timestamps.iloc[test_idx].min():%Y-%m-%d}/{timestamps.iloc[test_idx].max():%Y-%m-%d}"

    print(f"{len(train_idx):_} train rows {train_details}, {len(val_idx):_} val rows {val_details}, {len(test_idx):_} test rows {test_details}.")

    return train_idx, val_idx, test_idx, train_details, val_details, test_details


def select_target(
    model_name: str,
    target: np.ndarray,
    df: pd.DataFrame,
    train_df: pd.DataFrame = None,
    test_df: pd.DataFrame = None,
    val_df: pd.DataFrame = None,
    train_idx: np.ndarray = None,
    val_idx: np.ndarray = None,
    test_idx: np.ndarray = None,
    train_details: str = "",
    val_details: str = "",
    test_details: str = "",
    group_ids: np.ndarray = None,
    #
    control_params: dict = None,
    control_params_override: dict = None,
    config_params: dict = None,
    config_params_override: dict = None,
    common_params: dict = None,
    #
    sample_weight: np.ndarray = None,
):
    """From multiple possible targets in a dataframe, selects required one and adjusts params of respective level 0 models."""
    if target.dtype in (np.float64, np.float32, np.float16):
        model_name += f" MT={target.mean():.4f}"
    else:
        model_name += f" BT={target.value_counts(normalize=True).loc[1]*100:.0f}%"
    print("model_name=", model_name)

    if control_params is not None:
        effective_control_params = control_params
    else:
        effective_control_params = dict(
            use_regression=False,
            prefer_gpu_configs=True,
            robustness_features=None,
            use_robust_eval_metric=True,
            nbins=10,
            xgboost_verbose=0,
            rfecv_model_verbose=0,
            prefer_cpu_for_lightgbm=True,
            prefer_calibrated_classifiers=True,
        )
    if control_params_override:
        effective_control_params.update(control_params_override)

    effective_config_params = dict(
        has_time=False,
        learning_rate=0.2,
        iterations=700,
        early_stopping_rounds=100,
        catboost_custom_classif_metrics=None,
        rfecv_kwargs=dict(
            max_runtime_mins=60 * 3,
            cv_n_splits=4,
            max_noimproving_iters=15,
        ),
    )
    if config_params:
        effective_config_params = config_params
    if config_params_override:
        effective_config_params.update(config_params_override)

    common_params, common_cb_params, common_lgb_params, common_xgb_params, cb_rfecv, lgb_rfecv, xgb_rfecv, cpu_configs, gpu_configs = configure_training_params(
        df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target=target,
        target_label_encoder=None,
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
        **effective_control_params,
    )

    return common_params, common_cb_params, common_lgb_params, common_xgb_params, cb_rfecv, lgb_rfecv, xgb_rfecv, cpu_configs, gpu_configs


def process_model(
    model_file: str,
    model_name: str,
    target_type: str,
    pre_pipeline: object,
    pre_pipeline_name: str,
    cur_target: str,
    trainset_features_stats: dict,
    models: dict,
    model_params: dict,
    common_params: dict,
    ens_models: list,
    verbose: int,
) -> dict:

    fname = f"{model_name}.dump"
    if pre_pipeline_name:
        fname = pre_pipeline_name + " " + fname
    fpath = join(model_file, fname)
    if exists(fpath):
        model = load_mlframe_model(fpath)
    else:
        if verbose:
            logger.info(f"Starting train_and_evaluate {target_type} {model_name}, RAM usage {get_own_ram_usage():.1f}GBs...")
        model = train_and_evaluate_model(
            pre_pipeline=pre_pipeline,
            **model_params,
            **common_params,
            model_name_prefix=pre_pipeline_name,
        )
        save_mlframe_model(model, fpath)
    models[cur_target][target_type].append(model)

    if ens_models is not None:
        ens_models.append(model)

    if trainset_features_stats is None:
        trainset_features_stats = model.trainset_features_stats
        common_params["trainset_features_stats"] = trainset_features_stats

    clean_ram()
    if verbose:
        logger.info(f"Finished training. RAM usage {get_own_ram_usage():.1f}GBs...")

    return trainset_features_stats


def showcase_targets(target_types: dict):
    """Show distribution of targets"""
    for target_type, targets in target_types.items():
        for target_name, target in targets.items():
            display(f"{target_type} {target_name}")
            if target_type == TargetTypes.REGRESSION:
                plt.hist(target, bins=30, color="skyblue", edgecolor="black")

                # Add titles and labels
                plt.title(f"{target_name} Histogram")
                plt.xlabel("Value")
                plt.ylabel("Frequency")

                # Show the plot
                plt.show()
                if isinstance(target, (pl.Series, pd.Series)):
                    display(target.describe())
                elif isinstance(target, (np.ndarray)):
                    display(pl.Series(target).describe())
            elif target_type == TargetTypes.BINARY_CLASSIFICATION:
                display(target.value_counts(normalize=True))


def intize_targets(targets: dict) -> None:
    for target_name, target in targets.copy().items():
        if isinstance(target, pl.Series):
            target = target.cast(pl.Int8)
        else:
            target = target.astype(np.int8)
        targets[target_name] = target


def train_mlframe_models_suite(
    polars_df: pl.DataFrame,
    pandas_df: pd.DataFrame,
    target_name: str,
    model_name: str,
    preprocess_dataframe: Callable,
    #
    columns: list = None,
    data_dir: str = "",
    models_dir: str = MODELS_SUBDIR,
    #
    use_mlframe_models: bool = True,
    use_mlframe_xgboost: bool = True,
    use_mlframe_catboost: bool = True,
    use_mlframe_lightgbm: bool = True,
    use_mlframe_ensembles: bool = True,
    #
    use_autogluon_models: bool = False,
    autogluon_init_params: dict = None,
    autogluon_fit_params: dict = None,
    #
    use_lama_models: bool = False,
    lama_init_params: dict = None,
    lama_fit_params: dict = None,
    #
    use_rfecv: int = 0,
    report_params: dict = None,
    skip_infinity_checks: bool = True,
    verbose: int = 1,
    #
    automl_verbose: int = 1,
    automl_show_fi: bool = True,
    automl_target_label: str = "target",
    #
    config_params: dict = None,
    config_params_override: dict = None,
    control_params: dict = None,
    control_params_override: dict = None,
    init_common_params: dict = None,
    #
    drop_columns: list = None,
    tail: int = None,
) -> dict:

    # cb_kwargs=dict(devices='0-4')

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Inits
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    trainset_features_stats = None

    if init_common_params is None:
        init_common_params = {}

    if autogluon_fit_params is None:
        autogluon_fit_params = {}
    if autogluon_init_params is None:
        autogluon_init_params = {}

    if lama_init_params is None:
        lama_init_params = dict(task=Task("binary"))
    if lama_fit_params is None:
        lama_fit_params = {}

    if report_params is None:
        report_params = dict(
            nbins=10,
            # figsize=figsize,
            # print_report=print_report,
            # plot_file=plot_file + "_test" if plot_file else "",
            # show_perf_chart=show_perf_chart,
            # show_fi=show_fi,
            # fi_kwargs=fi_kwargs,
            # subgroups=subgroups,
            # subset_index=test_idx,
            # custom_ice_metric=custom_ice_metric,
            # custom_rice_metric=custom_rice_metric,
            # metrics=metrics["test"],
        )

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Conversion to Pandas & splitting
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if polars_df is not None and pandas_df is None:
        if verbose:
            logger.info(f"Converting polars df to pandas...")

        if columns:
            tmp = polars_df.select(columns)
        else:
            tmp = polars_df

        if verbose:
            logger.info(f"Converting polars df to pandas, RAM usage before: {get_own_ram_usage():.1f}GBs...")
        pandas_df = tmp.with_columns(pl.col(pl.Float64).cast(pl.Float32)).to_pandas()
        if verbose:
            logger.info(f"Converted polars df to pandas, RAM usage after: {get_own_ram_usage():.1f}GBs...")
    else:
        if isinstance(pandas_df, str):
            if verbose:
                logger.info(f"Loading pandas df from file, RAM usage before: {get_own_ram_usage():.1f}GBs...")
            pandas_df = pd.read_parquet(pandas_df, columns=columns)
            if verbose:
                logger.info(f"Loaded pandas df from file, RAM usage after: {get_own_ram_usage():.1f}GBs...")
    clean_ram()

    if drop_columns:
        for col in drop_columns:
            if col in pandas_df:
                del pandas_df[col]
                logger.info(f"Dropped column {col}")
        clean_ram()

    if tail:
        pandas_df = pandas_df.tail(tail)
        clean_ram()

    if verbose:
        logger.info(f"preprocess_dataframe...")
    pandas_df, target_types, group_ids, timestamps = preprocess_dataframe(pandas_df)

    clean_ram()

    if verbose:
        logger.info(f"make_train_test_split...")
    train_idx, val_idx, test_idx, train_details, val_details, test_details = make_train_test_split(df=pandas_df, timestamps=timestamps)

    if timestamps is not None:
        ts_file = join(data_dir, models_dir, slugify(target_name), slugify(model_name), "test_timestamps.parquet")
        if not exists(ts_file):
            ensure_dir_exists(join(data_dir, models_dir, slugify(target_name), slugify(model_name)))
            timestamps.iloc[test_idx].to_frame(name="ts").to_parquet(ts_file, compression=PARQUET_COMPRESION)

    if verbose:
        logger.info(f"creating train_df,val_df,test_df...")

    train_df = pandas_df.iloc[train_idx]
    val_df = pandas_df.iloc[val_idx]
    test_df = pandas_df.iloc[test_idx]

    columns = pandas_df.columns

    clean_ram()

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Checks
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if use_autogluon_models or use_lama_models:
        if automl_target_label in pandas_df:
            logger.warning(f"Your datafame ALREADY contains {automl_target_label} column that will be used by automl as temp target!")

        tran_val_idx = np.array(train_idx.tolist() + val_idx.tolist())
        if verbose:
            logger.info(f"RSS at start: {get_own_ram_usage():.1f}GBs")
    else:
        if verbose:
            logger.info(f"Ram usage before deleting main pandas df: {get_own_ram_usage():.1f}GBs")
        del pandas_df
        clean_ram()
        if verbose:
            logger.info(f"Ram usage after deleting main pandas df: {get_own_ram_usage():.1f}GBs")

    models = defaultdict(lambda: defaultdict(list))

    for target_type, targets in tqdmu(target_types.items(), desc="target type"):
        for cur_target, target in tqdmu(targets.items(), desc="target"):
            if use_mlframe_models:

                if use_autogluon_models or use_lama_models:
                    if automl_target_label in pandas_df:
                        del pandas_df[automl_target_label]
                        if verbose:
                            logger.info(f"RSS after automl_target_label deletion: {get_own_ram_usage():.1f}GBs")

                parts = slugify(target_name), slugify(model_name), slugify(target_type.lower()), slugify(cur_target)
                plot_file = join(data_dir, "charts", *parts) + os.path.sep
                ensure_dir_exists(plot_file)

                model_file = join(data_dir, models_dir, *parts) + os.path.sep
                ensure_dir_exists(model_file)

                if verbose:
                    logger.info(f"select_target...")

                cur_control_params_override = control_params_override.copy()
                cur_control_params_override["use_regression"] = target_type == TargetTypes.REGRESSION

                common_params, common_cb_params, common_lgb_params, common_xgb_params, cb_rfecv, lgb_rfecv, xgb_rfecv, cpu_configs, gpu_configs = select_target(
                    model_name=f"{target_name} {model_name} {cur_target}",
                    target=target,
                    df=None,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    train_details=train_details,
                    val_details=val_details,
                    test_details=test_details,
                    group_ids=group_ids,
                    config_params=config_params,
                    config_params_override=config_params_override,
                    control_params=control_params,
                    control_params_override=cur_control_params_override,
                    common_params=dict(
                        trainset_features_stats=trainset_features_stats, skip_infinity_checks=skip_infinity_checks, plot_file=plot_file, **init_common_params
                    ),
                )

                pre_pipelines = [None]
                pre_pipeline_names = [""]
                if use_rfecv >= 1:
                    pre_pipelines.append(cb_rfecv)
                    pre_pipeline_names.append("cb_rfecv ")
                if use_rfecv >= 2:
                    pre_pipelines.append(lgb_rfecv)
                    pre_pipeline_names.append("lgb_rfecv ")
                if use_rfecv >= 3:
                    pre_pipelines.append(xgb_rfecv)
                    pre_pipeline_names.append("xgb_rfecv ")

                for pre_pipeline, pre_pipeline_name in zip(pre_pipelines, pre_pipeline_names):
                    ens_models = [] if use_mlframe_ensembles else None
                    if use_mlframe_catboost:
                        trainset_features_stats = process_model(
                            model_file=model_file,
                            model_name="cb_model",
                            target_type=target_type,
                            pre_pipeline=pre_pipeline,
                            pre_pipeline_name=pre_pipeline_name,
                            cur_target=cur_target,
                            models=models,
                            model_params=common_cb_params,
                            common_params=common_params,
                            ens_models=ens_models,
                            trainset_features_stats=trainset_features_stats,
                            verbose=verbose,
                        )

                    if use_mlframe_lightgbm:
                        trainset_features_stats = process_model(
                            model_file=model_file,
                            model_name="lgb_model",
                            target_type=target_type,
                            pre_pipeline=pre_pipeline,
                            pre_pipeline_name=pre_pipeline_name,
                            cur_target=cur_target,
                            models=models,
                            model_params=common_lgb_params,
                            common_params=common_params,
                            ens_models=ens_models,
                            trainset_features_stats=trainset_features_stats,
                            verbose=verbose,
                        )

                    if use_mlframe_xgboost:
                        trainset_features_stats = process_model(
                            model_file=model_file,
                            model_name="xgb_model",
                            target_type=target_type,
                            pre_pipeline=pre_pipeline,
                            pre_pipeline_name=pre_pipeline_name,
                            cur_target=cur_target,
                            models=models,
                            model_params=common_xgb_params,
                            common_params=common_params,
                            ens_models=ens_models,
                            trainset_features_stats=trainset_features_stats,
                            verbose=verbose,
                        )

                    if ens_models and len(ens_models) > 1:

                        if verbose:
                            logger.info(f"evaluating simple ensembles...")

                        ensembles = score_ensemble(
                            models_and_predictions=ens_models,
                            ensemble_name=pre_pipeline_name + f"{len(ens_models)}models ",
                            **common_params,
                        )

            if use_autogluon_models or use_lama_models:

                pandas_df[automl_target_label] = target
                print(f"RSS after automl_target_label inserting: {get_own_ram_usage():.1f}GBs")

                automl_train_df = pandas_df.iloc[tran_val_idx].copy()

                test_target = target.iloc[test_idx]

            if use_autogluon_models:
                if verbose:
                    logger.info(f"train_and_evaluate_autogluon...")
                models[cur_target].append(
                    train_and_evaluate_autogluon(
                        train_df=automl_train_df,
                        test_df=test_df,
                        test_target=test_target,
                        columns=columns,
                        model_name=model_name,
                        automl_init_params=autogluon_init_params,
                        automl_fit_params=autogluon_fit_params,
                        automl_target_label=automl_target_label,
                        automl_show_fi=automl_show_fi,
                        automl_verbose=automl_verbose,
                        report_params=report_params,
                        group_ids=group_ids[test_idx] if group_ids is not None else None,
                    )
                )

            if use_lama_models:
                if verbose:
                    logger.info(f"train_and_evaluate_lama...")
                models[cur_target].append(
                    train_and_evaluate_lama(
                        train_df=automl_train_df,
                        test_df=test_df,
                        test_target=test_target,
                        columns=columns,
                        model_name=model_name,
                        automl_init_params=lama_init_params,
                        automl_fit_params=lama_fit_params,
                        automl_target_label=automl_target_label,
                        automl_show_fi=automl_show_fi,
                        automl_verbose=automl_verbose,
                        report_params=report_params,
                        group_ids=group_ids[test_idx] if group_ids is not None else None,
                    )
                )

    return models


# AUTOML


def train_and_evaluate_autogluon(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_target: pd.Series,
    columns: Iterable,
    model_name: str,
    automl_init_params: dict,
    automl_fit_params: dict,
    automl_target_label: str,
    automl_show_fi: str,
    automl_verbose: int,
    report_params: dict,
    group_ids: pd.Series = None,
) -> dict:

    try:
        from autogluon.tabular import TabularDataset, TabularPredictor
    except Exception as e:
        logger.warning(f"Skipping Autogluon: {e}.")
    else:

        model_name = f"AutoGluon_{model_name}"

        train_data = TabularDataset(train_df)

        automl = TabularPredictor(label=automl_target_label, verbosity=automl_verbose, **automl_init_params)
        automl.fit(train_data, **automl_fit_params)
        clean_ram()

        test_probs = automl.predict_proba(test_df).to_numpy()
        metrics = {"test": {}}
        _, _ = report_model_perf(
            targets=test_target,
            columns=columns,
            df=None,
            model_name=model_name,
            model=None,
            target_label_encoder=None,
            preds=None,
            probs=test_probs,
            report_title="TEST",
            metrics=metrics["test"],
            group_ids=group_ids if group_ids is not None else None,
            **report_params,
        )
        if automl_show_fi:
            automl_fi = automl.feature_importance(test_df)
        else:
            automl_fi = None

        return SimpleNamespace(model=automl, metrics=metrics, fi=automl_fi, test_target=test_target, test_probs=test_probs)


def train_and_evaluate_lama(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_target: pd.Series,
    columns: Iterable,
    model_name: str,
    automl_init_params: dict,
    automl_fit_params: dict,
    automl_target_label: str,
    automl_show_fi: str,
    automl_verbose: int,
    report_params: dict,
    group_ids: pd.Series = None,
) -> dict:

    try:
        from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
        from lightautoml.report.report_deco import ReportDeco, ReportDecoUtilized
        from lightautoml.tasks import Task
        import matplotlib as mpl
    except Exception as e:
        logger.warning(f"Skipping LaMa: {e}.")
    else:

        model_name = f"LaMa_{model_name}"

        # RD = ReportDeco(output_path="LaMa_model_report")

        # automl = RD(TabularAutoML(**lama_init_params))
        automl = TabularAutoML(**automl_init_params)

        out_of_fold_predictions = automl.fit_predict(train_df, roles={"target": automl_target_label}, verbose=automl_verbose, **automl_fit_params)

        """
        verbose - controls the verbosity: the higher, the more messages:
        <1  : messages are not displayed;
        >=1 : the computation process for layers is displayed;
        >=2 : the information about folds processing is also displayed;
        >=3 : the hyperparameters optimization process is also displayed;
        >=4 : the training process for every algorithm is displayed;
        """

        test_predictions = automl.predict(test_df)
        test_probs = np.vstack([1 - test_predictions.data[:, 0], test_predictions.data[:, 0]]).T

        mpl.rcParams.update(mpl.rcParamsDefault)

        metrics = {"test": {}}
        _, _ = report_model_perf(
            targets=test_target,
            columns=columns,
            df=None,
            model_name=model_name,
            model=None,
            target_label_encoder=None,
            preds=None,
            probs=test_probs,
            report_title="TEST",
            metrics=metrics["test"],
            group_ids=group_ids if group_ids is not None else None,
            **report_params,
        )
        if automl_show_fi:
            automl_fi = automl.get_feature_scores("fast")
            # automl_fi = automl.feature_importance(calc_method= "accurate",data=temp_test_df)
        else:
            automl_fi = None

        return SimpleNamespace(model=automl, metrics=metrics, fi=automl_fi, test_target=test_target, test_probs=test_probs)


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Early stopping
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def stop_file(fpath: str):
    return lambda: os.path.exists(fpath)


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
            logger.info("Training started. Timer initiated.")

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
        for preferred in ["ICE", "auc", "AUC"]:
            if preferred in available_metrics:
                self.monitor_metric = preferred
                break
        else:
            self.monitor_metric = available_metrics[0]
        self.mode = self.derive_mode(self.monitor_metric)
        if self.verbose > 0:
            logger.info(f"Auto-selected monitor_metric: {self.monitor_metric}, mode: {self.mode}")

    def _get_state(self, current_value: float) -> str:
        return f"iter={self.iter:_}, {self.monitor_dataset} {self.monitor_metric}: current={current_value:.{self.ndigits}f}, best={self.best_metric:.{self.ndigits}f} @{self.best_iter:_}"

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
                    if improved:
                        self.best_iter = self.iter
                        self.best_metric = current_value
                        self.iterations_since_improvement = 0
                        if self.verbose > 0 and (not self.reporting_interval_mins or (cur_ts - self.last_reporting_ts) >= self.reporting_interval_mins * 60):
                            logger.info(self._get_state(current_value=current_value))
                            self.last_reporting_ts = cur_ts
                    else:
                        self.iterations_since_improvement += 1
                        if self.verbose > 0 and (not self.reporting_interval_mins or (cur_ts - self.last_reporting_ts) >= self.reporting_interval_mins * 60):
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
            raise lgb.callback.EarlyStopException(self.best_iter, [(dataset, metric, self.best_metric, False)])
            return True


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
            model.set_attr(best_score=self.best_metric, best_iteration=self.best_iter)
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
