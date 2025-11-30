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

from typing import List, Dict, Iterable, Sequence, Union, Callable, Any, Optional, Tuple  # noqa: F401 pylint: disable=wildcard-import,unused-wildcard-import
from .config import *

# from pyutilz.pythonlib import ensure_installed;ensure_installed("pandas numpy numba scikit-learn lightgbm catboost xgboost shap")

import re
import copy
import inspect
from collections import Counter

import io
import os
import zstandard as zstd
from functools import partial
from types import SimpleNamespace
from collections import defaultdict


from timeit import default_timer as timer
from pyutilz.system import ensure_dir_exists, tqdmu
from pyutilz.system import compute_total_gpus_ram, get_gpuinfo_gpu_info
from pyutilz.pythonlib import prefix_dict_elems, get_human_readable_set_size, is_jupyter_notebook

from mlframe.helpers import get_model_best_iter, ensure_no_infinity, get_own_ram_usage


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# ANNS
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from lightning.pytorch.utilities.warnings import PossibleUserWarning

import torch
import torch.nn as nn
import torch.nn.functional as F


from mlframe.lightninglib import MLPNeuronsByLayerArchitecture, MLPTorchModel, custom_collate_fn
from mlframe.lightninglib import (
    PytorchLightningRegressor,
    PytorchLightningClassifier,
    TorchDataModule,
)


import argparse, warnings

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Filesystem
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import glob
from os.path import basename
from os.path import join, exists
from pyutilz.strings import slugify

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

from mlframe.ensembling import ensemble_probabilistic_predictions, score_ensemble, compare_ensembles, SIMPLE_ENSEMBLING_METHODS

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

# from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering
# # requires local import, causes problems in parallel runs (joblib, processpool, etc)
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
# Pandas
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import polars.selectors as cs
import pandas as pd, numpy as np, polars as pl, pyarrow as pa

from pyutilz.polarslib import polars_df_info, cast_f64_to_f32
from pyutilz.pandaslib import get_df_memory_consumption, showcase_df_columns
from pyutilz.pandaslib import ensure_dataframe_float32_convertability, optimize_dtypes, convert_float64_to_float32

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

from ngboost.scores import LogScore, CRPScore
from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import k_categorical, Bernoulli

from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBClassifier, XGBRegressor, DMatrix, QuantileDMatrix

import flaml.default as flaml_zeroshot

# from flaml.default import LGBMClassifier,LGBMRegressor,XGBClassifier,XGBRegressor
# from flaml.default import ExtraTreesClassifier,ExtraTreesRegressor,RandomForestClassifier,RandomForestRegressor

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
except Exception:
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

from sklearn.preprocessing import FunctionTransformer

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Pre- & postprocessing
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler

from mlframe.preprocessing import prepare_df_for_catboost

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# FIs
# -----------------------------------------------------------------------------------------------------------------------------------------------------


import shap
from mlframe.feature_importance import plot_feature_importance


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.metrics import make_scorer
from mlframe.metrics import fast_roc_auc, fast_calibration_report, compute_probabilistic_multiclass_error, ICE
from mlframe.metrics import create_fairness_subgroups, create_fairness_subgroups_indices, compute_fairness_metrics, robust_mlperf_metric

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Config Classes for train_and_evaluate_model
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import mean_absolute_error, max_error, mean_absolute_percentage_error, mean_squared_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except Exception:

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

from xgboost.callback import TrainingCallback

import catboost as cb
import xgboost as xgb
import lightgbm as lgb

from pyutilz.pythonlib import store_params_in_object, get_parent_func_args

# ----------------------------------------------------------------------------------------------------------------------------
# Helpers - moved to mlframe.training.helpers and mlframe.training.trainer
# Use: from mlframe.training import get_function_param_names, parse_catboost_devices
# ----------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------
# Enums - moved to mlframe.training.configs
# ----------------------------------------------------------------------------------------------------------------------------

from enum import StrEnum, auto

# Use: from mlframe.training import TargetTypes
from mlframe.training.configs import TargetTypes

# Import migrated functions/classes that are still used in this file
from mlframe.training.helpers import (
    get_training_configs,
    parse_catboost_devices,
    get_trainset_features_stats,
    get_trainset_features_stats_polars,
    LightGBMCallback,
    CatBoostCallback,
    XGBoostCallback,
)
from mlframe.training.trainer import get_function_param_names
from mlframe.training.extractors import FeaturesAndTargetsExtractor
from mlframe.training.utils import save_series_or_df

from mlframe.training.utils import log_ram_usage
from mlframe.training.io import load_mlframe_model, save_mlframe_model
from mlframe.training.extractors import intize_targets, showcase_features_and_targets

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

import sklearn

sklearn.set_config(transform_output="pandas")  # need this for val_df = pre_pipeline.transform(val_df) to work for SimpleImputer

CUDA_IS_AVAILABLE = is_cuda_available()
MODELS_SUBDIR = "models"
PARQUET_COMPRESION: str = "zstd"

# ----------------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------------

GPU_VRAM_SAFE_SATURATION_LIMIT: float = 0.9
GPU_VRAM_SAFE_FREE_LIMIT_GB: float = 0.1


# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def _run_hyperparameter_tuning(
    model,
    train_df,
    train_target,
    val_df,
    val_target,
    fit_params: dict,
    model_type_name: str,
    model_name: str,
    target_label_encoder=None,
    val_preds=None,
    val_probs=None,
    figsize=(15, 5),
    nbins: int = 10,
    fi_kwargs: dict = None,
    subgroups: dict = None,
    val_idx=None,
    custom_ice_metric=None,
    custom_rice_metric=None,
    n_trials: int = 100,
    timeout: int = 3600,
    verbose: bool = False,
):
    """Run Optuna hyperparameter tuning for CatBoost models.

    Currently only supports CatBoost models. Optimizes for class_robust_integral_error
    on the validation set.

    Returns:
        Model with best parameters applied
    """
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
            fi_kwargs=fi_kwargs if fi_kwargs else {},
            subgroups=subgroups,
            subset_index=val_idx,
            custom_ice_metric=custom_ice_metric,
            custom_rice_metric=custom_rice_metric,
            metrics=temp_metrics,
        )
        return temp_metrics[1]["class_robust_integral_error"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    if verbose:
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Best trial value: {study.best_trial.value}")
        print(f"Best params: {study.best_trial.params}")

    model.set_params(**study.best_trial.params)
    return model


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
        num_cols = get_numeric_columns(df)
        columns = [col for col in columns if col in num_cols]
        if columns:
            tmp = df.loc[:, columns].values
            mins = pd.Series(trainset_features_stats["min"]).loc[columns].values.astype(dtype)
            maxs = pd.Series(trainset_features_stats["max"]).loc[columns].values.astype(dtype)

            scores += (tmp < mins).sum(axis=1) / len(columns)
            scores += (tmp > maxs).sum(axis=1) / len(columns)
    return scores


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


def load_production_models(
    models_dir: str,
    target_name: str,
    featureset_name: str,
    task_type=TargetTypes.BINARY_CLASSIFICATION,
    directions: list = [],
    clean_models: bool = True,
    model_suffix: str = "_model",
) -> dict:
    """Reads models from disk, instantiates SHAP explainers where possible."""

    models = {}
    explainers = {}
    postcalibrators = {}

    from mlframe.ensembling import SIMPLE_ENSEMBLING_METHODS

    logger.info(f"Loading trained production {featureset_name} {task_type} models for target {target_name}...")

    featureset_dir = join(models_dir, slugify(target_name), slugify(featureset_name), task_type)
    trainset_features_stats = None
    for direction in tqdmu(directions, desc="direction", leave=False):

        models[direction] = {}
        explainers[direction] = {}
        postcalibrators[direction] = {}

        final_models_dir = join(featureset_dir, slugify(direction))

        for fpath in glob.glob(join(final_models_dir, f"*{model_suffix}.dump")):
            base_model_name = basename(fpath)

            model = load_mlframe_model(fpath)
            if trainset_features_stats is None:
                trainset_features_stats = model.trainset_features_stats
            if clean_models:
                clean_mlframe_model(model)
            model_name = base_model_name.replace(f"{model_suffix}.dump", "")
            models[direction][model_name] = model

            calib_fpath = fpath.replace(f"{model_suffix}.dump", "_model_postcalibrator.dump")
            if exists(calib_fpath):
                postcalibrator = joblib.load(calib_fpath)
                postcalibrators[direction][model_name] = postcalibrator

            explainer = None
            try:
                explainer = shap.TreeExplainer(model.model)
                explainers[direction][model_name] = explainer
            except Exception:
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


def train_mlframe_models_suite_old(
    df: Union[pl.DataFrame, pd.DataFrame, str],
    target_name: str,
    model_name: str,
    features_and_targets_extractor: FeaturesAndTargetsExtractor,
    #
    n_rows: int = None,
    tail: int = None,
    columns: list = None,
    drop_columns: list = None,
    #
    data_dir: str = "",
    models_dir: str = MODELS_SUBDIR,
    #
    mlframe_models: list = None,
    use_ordinary_models: bool = True,
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
    rfecv_models: list = None,
    report_params: dict = None,
    #
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
    fix_infinities: bool = True,
    fillna_value: float = None,
    ensure_float32_dtypes: bool = True,
    use_polarsds_pipeline: bool = True,
    #
    test_size: float = 0.1,
    val_size: float = 0.1,
    shuffle_val: bool = False,
    shuffle_test: bool = False,
    val_sequential_fraction: float = 0.5,
    test_sequential_fraction: float = None,
    trainset_aging_limit: float = None,
    wholeday_splitting: bool = True,
    #
    use_mrmr_fs: bool = False,
    mrmr_kwargs: dict = None,
    random_seed: int = 42,
    #
    imputer: object = None,
    scaler: object = None,
    category_encoder: object = None,
) -> dict:
    """In a unified fashion, train a bunch of models over the same data."""

    if verbose:
        logger.info(f"Starting MLFRAME models suite training. RAM usage: {get_own_ram_usage():.1f}GB.")

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Warnings
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    warnings.filterwarnings(
        "ignore",
        message=r"The '.*_dataloader' does not have many workers",
        module="lightning.pytorch.trainer.connectors.data_connector",
    )

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Inits
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    trainset_features_stats = None
    metadata = {}

    # if imputer is None:
    # imputer = SimpleImputer(strategy="most_frequent", add_indicator=False)

    if category_encoder is None:
        category_encoder = ce.CatBoostEncoder()
    if rfecv_models is None:
        rfecv_models = []
    if mrmr_kwargs is None:
        mrmr_kwargs = dict(n_workers=max(1, psutil.cpu_count(logical=False)), verbose=2, fe_max_steps=0)
    if mlframe_models is None:
        mlframe_models = "cb lgb xgb mlp".split()
    if init_common_params is None:
        init_common_params = {}
    if autogluon_fit_params is None:
        autogluon_fit_params = {}
    if autogluon_init_params is None:
        autogluon_init_params = {}

    if lama_init_params is None:
        from lightautoml.tasks import Task

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
    # Load, fill, and prepare df
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if isinstance(df, str):

        if verbose:
            logger.info(f"Loading df from file with Polars...")

        params = dict(parallel="columns")
        if n_rows:
            params["n_rows"] = n_rows
        if columns:
            params["columns"] = columns
        df = pl.read_parquet(df, **params)
        clean_ram()
        if verbose:
            log_ram_usage()

    # Now decrease "attack surface" as much as possible

    if tail:
        df = df.tail(tail)
        clean_ram()

    if verbose:
        logger.info(f"Preprocessing dataframe...")
    df, target_by_type, group_ids_raw, group_ids, timestamps, artifacts, additional_columns_to_drop = features_and_targets_extractor.transform(df)
    clean_ram()
    if verbose:
        log_ram_usage()

    if additional_columns_to_drop:
        if drop_columns:
            drop_columns.extend(additional_columns_to_drop)
        else:
            drop_columns = additional_columns_to_drop
    if drop_columns:

        logger.info(f"Dropping {len(drop_columns):_} columns...")

        if isinstance(df, pd.DataFrame):
            df_columns = set(df.columns)
            for col in drop_columns:
                if col in df_columns:
                    del df[col]
                else:
                    df = df.drop(col)
        elif isinstance(df, pl.DataFrame):
            df = df.drop(drop_columns, strict=False)

        clean_ram()
        if verbose:
            log_ram_usage()

    # Now can perform costly operations

    df = remove_constant_columns(df, verbose=verbose)

    if ensure_float32_dtypes:
        """Lightgbm uses np.result_type(*df_dtypes) to detect array dtype when converting from Pandas input,
        which results in float64 for int32 and above. For the rational mem usage, it makes sense to convert cols to float32 directly before training lightgbm.
        """
        if verbose:
            logger.info(f"Ensuring float32 dtypes...")
        df = ensure_dataframe_float32_convertability(df)
        clean_ram()
        if verbose:
            log_ram_usage()

    if fillna_value is not None:
        df = process_nulls(df, fill_value=fillna_value, verbose=verbose)
        df = process_nans(df, fill_value=fillna_value, verbose=verbose)
        imputer = None

        if fix_infinities:
            df = process_infinities(df, fill_value=fillna_value, verbose=verbose)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Train-val-test split
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if verbose:
        logger.info(f"make_train_test_split...")
    train_idx, val_idx, test_idx, train_details, val_details, test_details = make_train_test_split(
        df=df,
        timestamps=timestamps,
        test_size=test_size,
        val_size=val_size,
        shuffle_val=shuffle_val,
        shuffle_test=shuffle_test,
        val_sequential_fraction=val_sequential_fraction,
        test_sequential_fraction=test_sequential_fraction,
        trainset_aging_limit=trainset_aging_limit,
        wholeday_splitting=wholeday_splitting,
        random_seed=random_seed,
    )
    if verbose:
        log_ram_usage()

    # Save timestamps ,group ids & artifacts per set

    if data_dir is not None and models_dir:
        ensure_dir_exists(join(data_dir, models_dir, slugify(target_name), slugify(model_name)))
        for idx, idx_name in zip([train_idx, val_idx, test_idx], "train val test".split()):
            if idx is None:
                continue
            if timestamps is not None and len(timestamps) > 0:
                ts_file = join(data_dir, models_dir, slugify(target_name), slugify(model_name), f"{idx_name}_timestamps.parquet")
                if not exists(ts_file):
                    save_series_or_df(timestamps[idx], ts_file, PARQUET_COMPRESION, name="ts")
            if group_ids_raw is not None and len(group_ids_raw) > 0:
                gid_file = join(data_dir, models_dir, slugify(target_name), slugify(model_name), f"{idx_name}_group_ids_raw.parquet")
                if not exists(gid_file):
                    save_series_or_df(group_ids_raw[idx], gid_file, PARQUET_COMPRESION)
            if artifacts is not None and len(artifacts) > 0:
                art_file = join(data_dir, models_dir, slugify(target_name), slugify(model_name), f"{idx_name}_artifacts.parquet")
                if not exists(art_file):
                    save_series_or_df(artifacts[idx], art_file, PARQUET_COMPRESION)

    del timestamps, group_ids_raw, artifacts

    if verbose:
        logger.info(f"creating train_df,val_df,test_df...")

    if isinstance(df, pd.DataFrame):

        for next_df in (df,):
            if next_df is not None:
                cat_features = get_categorical_columns(next_df, include_string=True)
                if cat_features:
                    prepare_df_for_catboost(
                        df=next_df,
                        cat_features=cat_features,
                    )

        next_df = None

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx] if test_idx is not None else None
        val_df = df.iloc[val_idx] if val_idx is not None else None

    elif isinstance(df, pl.DataFrame):

        train_df = df[train_idx].clone()
        test_df = df[test_idx].clone() if test_idx is not None else None
        val_df = df[val_idx].clone() if val_idx is not None else None

        if verbose:
            log_ram_usage()

    if use_autogluon_models or use_lama_models:
        tran_val_idx = np.concatenate([train_idx, val_idx])
        if verbose:
            logger.info(f"RSS at start: {get_own_ram_usage():.1f}GBs")
    else:
        if verbose:
            logger.info(f"Ram usage before deleting main df: {get_own_ram_usage():.1f}GBs")
        del df
        clean_ram()
        if verbose:
            logger.info(f"Ram usage after deleting main df: {get_own_ram_usage():.1f}GBs")

    if isinstance(train_df, pl.DataFrame):

        if use_polarsds_pipeline:
            try:
                from polars_ds.pipeline import Pipeline as PdsPipeline, Blueprint as PdsBlueprint
            except Exception as e:
                logger.warning(f"Could not use mighty_scaler from polars-ds: {e}")
            else:

                if verbose:
                    logger.info(f"Fitting mighty_scaler from polars-ds...")

                bp = (
                    PdsBlueprint(
                        train_df,
                        name="mighty_scaler",
                    )
                    .scale(cs.numeric(), method="standard")
                    .ordinal_encode(cols=None, null_value=-1, unknown_value=-2)
                    .int_to_float(f32=True)
                    # .one_hot_encode(cols=None, drop_first=False, drop_cols=True)
                )
                mighty_scaler_pipe: PdsPipeline = bp.materialize()
                del bp
                clean_ram()

                if verbose:
                    log_ram_usage()
                    logger.info(f"Applying mighty_scaler from polars-ds...")

                train_df = mighty_scaler_pipe.transform(train_df)
                if val_idx is not None:
                    val_df = mighty_scaler_pipe.transform(val_df)
                if test_idx is not None:
                    test_df = mighty_scaler_pipe.transform(test_df)

                if ensure_float32_dtypes:
                    train_df = ensure_dataframe_float32_convertability(train_df)
                    if val_idx is not None:
                        val_df = ensure_dataframe_float32_convertability(val_df)
                    if test_idx is not None:
                        test_df = ensure_dataframe_float32_convertability(test_df)
                logger.info(f"train_df dtypes after mighty_scaler={Counter(train_df.dtypes)}")

                metadata["mighty_scaler_pipe"] = mighty_scaler_pipe
                scaler = None

                cat_features = []

    clean_ram()
    if verbose:
        log_ram_usage()

    if cat_features:
        logger.info(f"Ensuring cat_features={','.join(cat_features)}")
        for next_df in (train_df, val_df, test_df):
            if next_df is not None and isinstance(next_df, pd.DataFrame):
                prepare_df_for_catboost(
                    df=next_df,
                    cat_features=cat_features,
                )
        if verbose:
            log_ram_usage()

    if len(val_df) == 0:
        val_df = None

    columns = train_df.columns

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Actual training
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    models = defaultdict(lambda: defaultdict(list))
    for target_type, targets in tqdmu(target_by_type.items(), desc="target type"):
        # !TODO ! optimize for creation of inner feature matrices of cb,lgb,xgb here. They should be created once per featureset, not once per target.
        for cur_target_name, cur_target_values in tqdmu(targets.items(), desc="target"):
            if mlframe_models:
                if use_autogluon_models or use_lama_models:
                    if automl_target_label in df.columns:
                        if isinstance(df, pd.DataFrame):
                            del df[automl_target_label]
                        else:
                            df = df.drop(automl_target_label)
                        if verbose:
                            logger.info(f"RSS after automl_target_label deletion: {get_own_ram_usage():.1f}GBs")

                parts = slugify(target_name), slugify(model_name), slugify(target_type.lower()), slugify(cur_target_name)
                if data_dir is not None:
                    plot_file = join(data_dir, "charts", *parts) + os.path.sep
                    ensure_dir_exists(plot_file)
                else:
                    plot_file = None
                if models_dir is not None:
                    model_file = join(data_dir, models_dir, *parts) + os.path.sep
                    ensure_dir_exists(model_file)
                else:
                    model_file = None

                if verbose:
                    logger.info(f"select_target...")
                cur_control_params_override = control_params_override.copy()
                cur_control_params_override["use_regression"] = target_type == TargetTypes.REGRESSION
                common_params, models_params, rfecv_models_params, cpu_configs, gpu_configs = select_target(
                    model_name=f"{target_name} {model_name} {cur_target_name}",
                    target=cur_target_values,
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
                    cat_features=cat_features,
                    config_params=config_params,
                    config_params_override=config_params_override,
                    control_params=control_params,
                    control_params_override=cur_control_params_override,
                    common_params=dict(
                        trainset_features_stats=trainset_features_stats, skip_infinity_checks=skip_infinity_checks, plot_file=plot_file, **init_common_params
                    ),
                )

                pre_pipelines = []
                pre_pipeline_names = []

                if use_ordinary_models:
                    pre_pipelines.append(None)
                    pre_pipeline_names.append("")

                for rfecv_model_name in rfecv_models:
                    if rfecv_model_name not in rfecv_models_params:
                        logger.warning(f"RFECV model {rfecv_model_name} not known, skipping...")
                    else:
                        pre_pipelines.append(rfecv_models_params[rfecv_model_name])
                        pre_pipeline_names.append(f"{rfecv_model_name} ")

                if use_mrmr_fs:
                    pre_pipelines.append(MRMR(**mrmr_kwargs))
                    pre_pipeline_names.append("MRMR ")

                for pre_pipeline, pre_pipeline_name in zip(pre_pipelines, pre_pipeline_names):
                    if pre_pipeline_name == "cb_rfecv" and target_type == TargetTypes.REGRESSION and control_params_override.get("metamodel_func") is not None:
                        # File /venv/main/lib/python3.12/site-packages/sklearn/base.py:142, in _clone_parametrized(estimator, safe)
                        # RuntimeError: Cannot clone object <catboost.core.CatBoostRegressor object at 0x713048b0e840>, as the constructor either does not set or modifies parameter custom_metric
                        continue
                    ens_models = [] if use_mlframe_ensembles else None
                    orig_pre_pipeline = pre_pipeline
                    for mlframe_model_name in mlframe_models:
                        if mlframe_model_name == "cb" and target_type == TargetTypes.REGRESSION and control_params_override.get("metamodel_func") is not None:
                            continue
                        if mlframe_model_name not in models_params:
                            logger.warning(f"mlframe model {mlframe_model_name} not known, skipping...")
                        else:
                            if mlframe_model_name == "hgb" and cat_features:
                                pre_pipeline = Pipeline(
                                    steps=[
                                        *([("pre", orig_pre_pipeline)] if orig_pre_pipeline else []),
                                        ("ce", ce.CatBoostEncoder(verbose=2)),
                                    ]
                                )
                            elif mlframe_model_name in ("mlp", "ngb"):
                                pre_pipeline = Pipeline(
                                    steps=[
                                        *([("pre", orig_pre_pipeline)] if orig_pre_pipeline else []),
                                        *([("ce", category_encoder)] if cat_features else []),
                                        *([("imp", imputer)] if imputer else []),
                                        *([("scaler", scaler)] if scaler else []),
                                    ]
                                )
                            trainset_features_stats, pre_pipeline = process_model(
                                model_file=model_file,
                                model_name=mlframe_model_name,
                                target_type=target_type,
                                pre_pipeline=pre_pipeline,
                                pre_pipeline_name=pre_pipeline_name,
                                cur_target_name=cur_target_name,
                                models=models,
                                model_params=models_params[mlframe_model_name],
                                common_params=common_params,
                                ens_models=ens_models,
                                trainset_features_stats=trainset_features_stats,
                                verbose=verbose,
                            )
                            if mlframe_model_name not in ("hgb", "mlp", "ngb"):
                                orig_pre_pipeline = pre_pipeline

                    if ens_models and len(ens_models) > 1:
                        if verbose:
                            logger.info(f"evaluating simple ensembles...")
                        ensembles = score_ensemble(
                            models_and_predictions=ens_models,
                            ensemble_name=pre_pipeline_name + f"{len(ens_models)}models ",
                            **common_params,
                        )
            if use_autogluon_models or use_lama_models:
                if isinstance(df, pd.DataFrame):
                    df[automl_target_label] = cur_target_values
                else:
                    df = df.with_columns(pl.Series(automl_target_label, cur_target_values))
                print(f"RSS after automl_target_label inserting: {get_own_ram_usage():.1f}GBs")
                if isinstance(df, pd.DataFrame):
                    automl_train_df = df.iloc[tran_val_idx].copy()
                else:
                    automl_train_df = get_pandas_view_of_polars_df(df[tran_val_idx])
                test_target = cur_target_values[test_idx]
                test_df_automl = test_df if isinstance(test_df, pd.DataFrame) else (get_pandas_view_of_polars_df(test_df) if test_df is not None else None)
            if use_autogluon_models:
                if verbose:
                    logger.info(f"train_and_evaluate_autogluon...")
                models[cur_target_name].append(
                    train_and_evaluate_autogluon(
                        train_df=automl_train_df,
                        test_df=test_df_automl,
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
                models[cur_target_name][target_type].append(
                    train_and_evaluate_lama(
                        train_df=automl_train_df,
                        test_df=test_df_automl,
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
    return models, metadata


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Early stopping & callbacks
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def stop_file(fpath: str):
    return lambda: os.path.exists(fpath)


# Callback classes moved to mlframe.training.helpers
# Use: from mlframe.training import UniversalCallback, LightGBMCallback, XGBoostCallback, CatBoostCallback


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Post-modelling stuff
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def read_oos_predictions(
    models: list,
    models_dir: str,
    max_mae: float = 0,
    max_std: float = 0,
    ensure_prob_limits: bool = True,
    uncertainty_quantile: float = 0.1,
    normalize_stds_by_mean_preds: bool = False,
    ts_field: str = "ts",
    group_field: str = "secid",
    verbose: int = 1,
) -> pl.DataFrame:

    res = {}

    test_timestamps = pl.read_parquet(join(models_dir, "test_timestamps.parquet"))
    test_group_ids_raw = pl.read_parquet(join(models_dir, "test_group_ids_raw.parquet"))

    res[ts_field] = test_timestamps[ts_field]
    res[group_field] = test_group_ids_raw[group_field]
    for target_name, submodels in models.items():
        all_models_predictions = []
        for model_name, model in submodels.items():
            res[target_name] = model.test_target
            key = f"{target_name}-{model_name}"
            if model.test_probs is not None:
                res[key] = model.test_probs[:, 1]
                all_models_predictions.append(model.test_probs)
            else:
                res[key] = model.test_preds
        for ensembling_method in ("harm", "arithm", "median", "quad", "qube", "geo"):
            ensembled_predictions, predictions_stds, confident_indices = ensemble_probabilistic_predictions(
                *all_models_predictions,
                ensemble_method=ensembling_method,
                max_mae=max_mae,
                max_std=max_std,
                ensure_prob_limits=ensure_prob_limits,
                uncertainty_quantile=uncertainty_quantile,
                normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
                verbose=verbose,
            )
            res[f"{target_name}-ens_{ensembling_method}"] = ensembled_predictions[:, 1]

    predictions_df = pl.DataFrame(res)
    return predictions_df


def compute_models_perf(
    df: pl.DataFrame,
    directions: list,
    report_title: str,
    charts_dir: str = None,
    group_field: str = "secid",
    suffixes=["_prob"],
    direct_order: bool = True,
    show_perf_chart: bool = True,
    models: list = None,
):

    report_params = {
        "report_ndigits": 2,
        "calib_report_ndigits": 2,
        "print_report": False,
        "report_title": report_title,
        "use_weights": True,
        "show_perf_chart": show_perf_chart,
    }

    if not models:
        models = "cb lgb xgb".split() + [f"ens_{method}" for method in SIMPLE_ENSEMBLING_METHODS]

    metrics = {}

    group_ids = df[group_field].to_numpy()
    unique_vals, group_ids = np.unique(group_ids, return_inverse=True)

    for direction in directions:
        for model in models:
            for suffix in suffixes:

                if direct_order:
                    model_name = f"{model}_{direction}{suffix}"
                else:
                    model_name = f"{direction}-{model}{suffix}"

                prob_col = f"{model_name}"
                if prob_col not in df:
                    prob_col = f"{model_name}_probs"
                    if prob_col not in df:
                        continue

                if False:
                    up = "UP" in model_name
                    if up:
                        targets = (df[f"target_UP"].to_numpy() >= MIN_SIGNIFICANT_LONG_RETURN).astype(np.int8)
                    else:
                        targets = (df[f"target_DOWN"].to_numpy() >= MIN_SIGNIFICANT_SHORT_RETURN).astype(np.int8)

                targets = df[direction].to_numpy().astype(np.int8)

                probs = df[prob_col].to_numpy()

                metrics[model_name] = {
                    "pred_min": np.min(probs),
                    #'pred_q_0.01':np.quantile(probs,0.01),
                    #'pred_mean':np.mean(probs),
                    "pred_median": np.median(probs),
                    "pred_q_0.999": np.quantile(probs, 0.999),
                    #'pred_max':np.max(probs)
                    "target_mean": np.mean(targets),
                }

                probs = np.vstack([1 - probs, probs]).T

                _, _ = report_model_perf(
                    targets=targets,
                    columns=None,
                    df=None,
                    model_name=f"{model_name}",
                    model=None,
                    target_label_encoder=None,
                    preds=None,
                    probs=probs,
                    plot_file=join(charts_dir, model_name) if charts_dir else None,
                    metrics=metrics[model_name],
                    group_ids=group_ids,
                    **report_params,
                )

    metrics = pd.DataFrame(metrics).T

    transformed = False
    for label in (1, 0):
        if label in metrics.columns:
            transformed = True
            try:
                metrics = (
                    metrics.drop(columns=[label])
                    .join(metrics[label].apply(pd.Series))
                    .drop(columns=["feature_importances", "class_integral_error"])
                    .sort_values("ice")
                )
            except Exception:
                return None
            metrics["flipped"] = label != 1
            break

    if not transformed:
        metrics = None

    return metrics


def compute_ml_perf(
    predictions_df: pl.DataFrame,
    directions: list,
    group_field: str = None,
    show_perf_chart: bool = True,
    ts_field: str = "ts",
    truncate_to: str = "1mo",
    alias: str = "month",
) -> pd.DataFrame:

    perf_stats = []
    by_time = ts_field and truncate_to and alias
    assert (group_field is not None) or by_time

    group_field_name = group_field

    if by_time:
        grouping = pl.col(ts_field).dt.truncate(truncate_to)
    else:
        if isinstance(group_field, str):
            grouping = pl.col(group_field)

        else:
            grouping = group_field
            group_field_name = alias if alias else group_field.meta.root_names()[0]

    for mo, df in tqdmu(list(predictions_df.group_by(grouping, maintain_order=True))):
        if show_perf_chart:
            fields = dict(
                npredictions=pl.len(),
                min_date=pl.col(ts_field).min().dt.date(),
                max_date=pl.col(ts_field).max().dt.date(),
            )
            if group_field is not None:
                if isinstance(group_field, str):
                    if by_time:
                        fields[group_field] = pl.col(group_field).n_unique()
                    else:
                        fields[group_field] = pl.col(group_field).unique()

            stats: dict = df.select(**fields).row(0, named=True)

            if isinstance(group_field, str):
                if by_time:
                    report_title = f"Test {stats['min_date']:%Y-%m-%d}->{stats['max_date']:%Y-%m-%d}, {stats[group_field]/1000:_.2f}K {group_field} "
                else:
                    report_title = f"Test {stats['min_date']:%Y-%m-%d}->{stats['max_date']:%Y-%m-%d}, {group_field}={stats[group_field]} "
            elif isinstance(group_field, pl.Expr):
                report_title = f"Test {stats['min_date']:%Y-%m-%d}->{stats['max_date']:%Y-%m-%d}, {group_field_name}={mo[0]} "

            print(report_title)
        else:
            report_title = ""

        res = compute_models_perf(df=df, directions=directions, report_title=report_title, suffixes=[""], direct_order=False, show_perf_chart=show_perf_chart)

        if res is not None:
            res = res.reset_index(drop=False, names="model")
            res["nrecs"] = len(df)
            if by_time:
                res[alias] = mo[0]
            else:
                if isinstance(group_field, str):
                    res[group_field] = mo[0]
                elif isinstance(group_field, pl.Expr):
                    res[group_field_name] = mo[0]

            perf_stats.append(res)
        else:
            logger.warning(f"Problem computing models perf for {mo[0]}")

    perf_stats = pd.concat(perf_stats).sort_values(["model", alias if by_time else group_field_name])
    return perf_stats


def visualize_ml_metric_by_time(
    perf_stats: pd.DataFrame,
    metric: str = "ice",
    rotation: float = 45,
    good_metric_threshold: float = 0.0,
    good_color="green",
    bad_color="red",
    higher_is_better: bool = False,
    truncated_interval_name: str = "month",
    figsize=(12, 6),
):
    import matplotlib.pyplot as plt

    for model in perf_stats.model.unique():
        tmp = perf_stats[perf_stats.model == model].set_index(truncated_interval_name)
        ax = tmp.plot(y=metric, kind="bar", title=f"{metric.upper()} of {model}: mean={tmp[metric].mean():.3f}, std={tmp[metric].std():.3f}", figsize=figsize)
        ax.xaxis.set_major_formatter(plt.FixedFormatter(tmp.index.strftime("%Y-%m-%d")))

        # Color bars depending on value
        if good_metric_threshold is not None:
            for bar, val in zip(ax.patches, tmp[metric]):
                if val > good_metric_threshold:
                    bar.set_facecolor(bad_color if not higher_is_better else good_color)
                else:
                    bar.set_facecolor(bad_color if higher_is_better else good_color)

        plt.xticks(rotation=rotation)
        plt.tight_layout()
        plt.show()


def predictions_beautify_linear(preds: np.ndarray, known_outcomes: np.ndarray, alpha=0.01):
    """
    Adjust probabilities toward the true labels.

    preds: 1D array of floats in [0,1]
    known_outcomes:     1D array of ints {0,1}
    alpha: how far to move toward truth (0=no change, 1=fully corrected)

    returns: adjusted probabilities
    """
    preds = np.asarray(preds, dtype=float)
    y = np.asarray(known_outcomes, dtype=float)
    return (1 - alpha) * preds + alpha * y
