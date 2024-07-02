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

#from pyutilz.pythonlib import ensure_installed;ensure_installed("pandas numpy numba scikit-learn lightgbm catboost xgboost shap")

import numba
from numba.cuda import is_available as is_cuda_available

import copy
import joblib
import psutil
from gc import collect
from functools import partial
from os.path import join, exists
from types import SimpleNamespace
from collections import defaultdict

import matplotlib.pyplot as plt
from IPython.display import display

import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBClassifier, XGBRegressor, DMatrix, QuantileDMatrix

import shap
import pandas as pd, numpy as np
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import ClassifierMixin, RegressorMixin,TransformerMixin,is_classifier
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import mean_absolute_error,max_error,mean_absolute_percentage_error,root_mean_squared_error

from pyutilz.pythonlib import prefix_dict_elems
from pyutilz.system import ensure_dir_exists, tqdmu
from pyutilz.pandaslib import ensure_dataframe_float32_convertability, optimize_dtypes, remove_constant_columns

from mlframe.helpers import get_model_best_iter
from mlframe.preprocessing import prepare_df_for_catboost
from mlframe.feature_importance import plot_feature_importance
from mlframe.feature_selection.wrappers import RFECV, VotesAggregation, OptimumSearch
from mlframe.metrics import fast_roc_auc, fast_calibration_report, compute_probabilistic_multiclass_error, CB_EVAL_METRIC
from mlframe.metrics import create_robustness_subgroups,create_robustness_subgroups_indices,compute_robustness_metrics,robust_mlperf_metric

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

import sklearn

sklearn.set_config(transform_output="pandas")  # need this for val_df = pre_pipeline.transform(val_df) to work for SimpleImputer

CUDA_IS_AVAILABLE = is_cuda_available()
DATA_DIR = ""
if DATA_DIR:
    ensure_dir_exists(DATA_DIR)
MODELS_SUBDIR = "models"
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
    subgroups:dict=None,
    learning_rate: float = 0.1,

    def_regr_metric: str = "MAE",
    def_classif_metric: str = "AUC",
    catboost_custom_classif_metrics:Sequence = ["AUC", "BrierScore", "PRAUC"],
    catboost_custom_regr_metrics:Sequence = ["RMSE", "MAPE"],    

    random_seed=None,
    verbose: int = 0,    
    # ----------------------------------------------------------------------------------------------------------------------------
    # probabilistic errors
    # ----------------------------------------------------------------------------------------------------------------------------
    method:str="multicrit",
    mae_weight: float = 0.5,
    std_weight: float = 0.5,
    roc_auc_weight: float = 1.5,
    brier_loss_weight: float = 0.2,
    min_roc_auc: float = 0.54,
    roc_auc_penalty: float = 0.00,
    use_weighted_calibration: bool = True,
    weight_by_class_npositives: bool = False,
    nbins: int = 100,
    # ----------------------------------------------------------------------------------------------------------------------------
    # featureselectors
    # ----------------------------------------------------------------------------------------------------------------------------    
    max_runtime_mins:float=60 * 2,
    max_noimproving_iters:int=40,
    cv=None,
    cv_n_splits: int = 5,
) -> tuple:
    """Returns approximately same training configs for different types of models,
    based on general params supplied like learning rate, task type, time budget.
    Useful for more or less fair comparison between different models on the same data/task, and their upcoming ensembling.
    This procedure is good for manual EDA and getting the feeling of what ML models are capable of for a particular task.
    """

    if has_gpu is None:
        has_gpu=CUDA_IS_AVAILABLE

    if not early_stopping_rounds:
        early_stopping_rounds = max(2, iterations // 3)

    def neg_ovr_roc_auc_score(*args, **kwargs):
        return -roc_auc_score(*args, **kwargs, multi_class="ovr")

    CB_GENERAL_PARAMS = dict(
        iterations=iterations,
        verbose=verbose,
        has_time=has_time,
        learning_rate=learning_rate,
        eval_fraction=(0.0 if use_explicit_early_stopping else validation_fraction),
        task_type=("GPU" if has_gpu else "CPU"),        
        early_stopping_rounds=early_stopping_rounds,
        random_seed=random_seed,
    )

    CB_CLASSIF = CB_GENERAL_PARAMS.copy()
    CB_CLASSIF.update({"eval_metric": def_classif_metric, "custom_metric": catboost_custom_classif_metrics})

    CB_REGR = CB_GENERAL_PARAMS.copy()
    CB_REGR.update({"eval_metric": def_regr_metric, "custom_metric": catboost_custom_regr_metrics})


    XGB_GENERAL_PARAMS = dict(
        n_estimators=iterations,
        enable_categorical=True,
        max_cat_to_onehot=1,
        max_cat_threshold=1000,
        tree_method="hist",
        device=("cuda" if has_gpu else "cpu"),
        n_jobs=psutil.cpu_count(logical=False),
        early_stopping_rounds=early_stopping_rounds,
        random_seed=random_seed,
        verbosity=int(verbose),
    )    

    XGB_GENERAL_CLASSIF = XGB_GENERAL_PARAMS.copy()
    XGB_GENERAL_CLASSIF.update({"objective": "binary:logistic","eval_metric":neg_ovr_roc_auc_score})    

    def integral_calibration_error(y_true, y_score,verbose: bool = False):

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
        # final_integral_calibration_error=partial(robust_mlperf_metric,metric=integral_calibration_error,higher_is_better=False,subgroups=subgroups)
        def final_integral_calibration_error(y_true: np.ndarray,y_score: np.ndarray,*args,**kwargs): # partial won't work with xgboost
            return robust_mlperf_metric(y_true,y_score,*args,metric=integral_calibration_error,higher_is_better=False,subgroups=subgroups,**kwargs,)
    else:
        final_integral_calibration_error=integral_calibration_error

    def fs_and_hpt_integral_calibration_error(*args, verbose: bool = False, **kwargs):
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
    CB_CALIB_CLASSIF.update({"eval_metric": CB_EVAL_METRIC(metric=final_integral_calibration_error,higher_is_better=False)})
    
    LGB_GENERAL_PARAMS = dict(
        n_estimators=iterations,
        early_stopping_rounds=early_stopping_rounds,
        device_type=("gpu" if has_gpu else "cpu"),
        verbose=int(verbose),
        random_state=random_seed,
    )

    #XGB_CALIB_CLASSIF_CPU.update({"device": "cpu","n_jobs":psutil.cpu_count(logical=False)})    

    if not cv:
        if has_time:
            cv = TimeSeriesSplit(n_splits=cv_n_splits)
        else:
            cv = StratifiedKFold(n_splits=cv_n_splits, shuffle=not has_time)

    COMMON_RFECV_PARAMS = dict(
        cv=cv,
        cv_shuffle=not has_time,
        skip_retraining_on_same_shape=True,
        top_predictors_search_method=OptimumSearch.ModelBasedHeuristic,
        votes_aggregation_method=VotesAggregation.Borda,
        early_stopping_rounds=early_stopping_rounds,
        use_last_fi_run_only=False,
        verbose=True,
        show_plot=True,
        keep_estimators=False,
        feature_cost=0.0 / 100,
        smooth_perf=0,
        max_refits=None,
        max_runtime_mins=max_runtime_mins,
        max_noimproving_iters=max_noimproving_iters,
        # frac=0.2,
    )

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
# Core
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def train_and_evaluate_model(
    model: ClassifierMixin,
    df: pd.DataFrame,
    target: pd.Series,
    sample_weight: pd.Series = None,
    model_name: str = "",
    pre_pipeline: TransformerMixin = None,
    fit_params: Optional[dict] = None,
    drop_columns: list = [],
    default_drop_columns: list = [],
    target_label_encoder: Optional[LabelEncoder] = None,
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
    print_report:bool=True,
    show_perf_chart:bool=True,
    show_fi:bool=True,
    use_cache: bool = False,
    nbins: int = 100,
    compute_trainset_metrics:bool=True,
    compute_valset_metrics: bool = True,
    compute_testset_metrics: bool = True,
    data_dir: str = DATA_DIR,
    models_subdir: str = MODELS_SUBDIR,
    include_confidence_analysis:bool=False,
    display_sample_size: int =0,
    show_feature_names: bool=False,
    # confidence_analysis
    confidence_analysis_use_shap:bool=True,
    confidence_analysis_max_features:int=6,
    confidence_analysis_cmap:str="bwr",
    confidence_analysis_alpha:float=0.9,
    confidence_analysis_ylabel:str="Feature value",
    confidence_analysis_title:str="Confidence of correct Test set predictions",
):
    """Trains & evaluates given model/pipeline on train/test sets.
    Supports feature selection via pre_pipeline.
    Supports early stopping via val_idx.
    Optionally fumps resulting model & test set predictions into the models dir, and loads back by model name on the next call, to save time.
    """
    
    collect()

    if not custom_ice_metric:
        custom_ice_metric = compute_probabilistic_multiclass_error

    ensure_dir_exists(join(data_dir, models_subdir))
    model_file_name = join(data_dir, models_subdir, f"{model_name}.dump")

    if use_cache and exists(model_file_name):
        logger.info(f"Loading model from file {model_file_name}")
        model, *_, pre_pipeline = joblib.load(model_file_name)

    real_drop_columns = [col for col in drop_columns + default_drop_columns if col in df.columns]

    if fit_params and isinstance(model, Pipeline):
        model_obj=model.named_steps["est"]  # model.steps[-1]
    else:
        model_obj=model

    if model_obj is not None:
        if isinstance(model_obj,TransformedTargetRegressor):
            model_obj=model_obj.regressor             
    model_type_name = type(model_obj).__name__ if model_obj is not None else ""

    if model_type_name not in model_name:
        model_name=model_type_name+" "+model_name

    if fit_params is None:
        fit_params = {}

    if df is not None:
        if val_idx is not None:
            train_df = df.loc[train_idx].drop(columns=real_drop_columns)
            val_df = df.loc[val_idx].drop(columns=real_drop_columns)
        else:
            train_df = df.loc[train_idx].drop(columns=real_drop_columns)

    if model is not None and pre_pipeline:
        if use_cache and exists(model_file_name):
            train_df = pre_pipeline.transform(train_df, target.loc[train_idx])
        else:
            train_df = pre_pipeline.fit_transform(train_df, target.loc[train_idx])
        if val_idx is not None:
            val_df = pre_pipeline.transform(val_df)

    if val_idx is not None:
        # insert eval_set where needed

        if model_type_name in XGBOOST_MODEL_TYPES:
            fit_params["eval_set"] = ((val_df, target.loc[val_idx]),)
        elif model_type_name in LGBM_MODEL_TYPES:
            fit_params["eval_set"] = (val_df, target.loc[val_idx])
            # fit_params["callbacks"] = [lgb.early_stopping(stopping_rounds=early_stopping_rounds)]
        elif model_type_name in CATBOOST_MODEL_TYPES:
            fit_params["eval_set"] = [
                (val_df, target.loc[val_idx]),
            ]
        elif model_type_name in TABNET_MODEL_TYPES:
            fit_params["eval_set"] = [
                (val_df.values, target.loc[val_idx].values),
            ]
        elif model_type_name in PYTORCH_MODEL_TYPES:
            fit_params["eval_set"] = (val_df, target.loc[val_idx])

    if model is not None and fit_params:
        if "cat_features" in fit_params:
            fit_params["cat_features"] = [col for col in fit_params["cat_features"] if col in train_df.columns]

    if fit_params and isinstance(model, Pipeline):
        fit_params = prefix_dict_elems(fit_params, "est__")

    if model is not None:
        if (not use_cache) or (not exists(model_file_name)):
            if sample_weight is not None:
                sample_weight = sample_weight.loc[train_idx].values
            logger.info(f"{model_name} training dataset shape: {train_df.shape}")
            if display_sample_size:
                display(train_df.head(display_sample_size).style.set_caption(f"{model_name} features head"))
                display(train_df.tail(display_sample_size).style.set_caption(f"{model_name} features tail"))
            
            if df is not None:
                
                report_title = f"Training {model_name} model on {train_df.shape[1]} feature(s)" # textwrap.shorten("Hello world", width=10, placeholder="...")
                if show_feature_names:
                    report_title+=": "+', '.join(train_df.columns.to_list())
                report_title+=f", {len(train_df):_} records"

            if model_type_name in TABNET_MODEL_TYPES:
                train_df=train_df.values
            
            model.fit(train_df, target.loc[train_idx], sample_weight=sample_weight, **fit_params)
            if model is not None:
                # get number of the best iteration
                try:
                    best_iter = get_model_best_iter(model)
                    if best_iter:
                        print(f"es_best_iter: {best_iter:_}")
                except Exception as e:
                    logger.warning(e)

    metrics={'train':{},'val':{},'test':{}}
    if compute_trainset_metrics or compute_valset_metrics or compute_testset_metrics:
        if compute_trainset_metrics and train_idx is not None:
            if df is None:
                train_df = None
                columns = []
            else:
                columns = train_df.columns

            train_preds, train_probs = report_model_perf(
                targets=target.loc[train_idx],
                columns=columns,
                df=train_df.values if model_type_name in TABNET_MODEL_TYPES else train_df,
                model_name="TRAIN " + model_name,
                model=model,
                target_label_encoder=target_label_encoder,
                preds=train_preds,
                probs=train_probs,
                figsize=figsize,
                report_title="",
                nbins=nbins,
                print_report=print_report,
                show_perf_chart=show_perf_chart,
                show_fi=False,
                subgroups=subgroups,
                subset_index=train_idx,
                custom_ice_metric=custom_ice_metric,
                custom_rice_metric=custom_rice_metric,
                metrics=metrics['train']
            )
        
        if compute_valset_metrics and val_idx is not None:
            if df is None:
                val_df = None
                columns = []
            else:
                columns = val_df.columns

            val_preds, val_probs = report_model_perf(
                targets=target.loc[val_idx],
                columns=columns,
                df=val_df.values if model_type_name in TABNET_MODEL_TYPES else val_df,
                model_name="VAL " + model_name,
                model=model,
                target_label_encoder=target_label_encoder,
                preds=val_preds,
                probs=val_probs,
                figsize=figsize,
                report_title="",
                nbins=nbins,
                print_report=print_report,
                show_perf_chart=show_perf_chart,
                show_fi=False,
                subgroups=subgroups,
                subset_index=val_idx,
                custom_ice_metric=custom_ice_metric,
                custom_rice_metric=custom_rice_metric,
                metrics=metrics['val']
            )

        if compute_testset_metrics and test_idx is not None:
            if df is not None:            

                del train_df
                collect()

                test_df = df.loc[test_idx].drop(columns=real_drop_columns)
                if model is not None and pre_pipeline:
                    test_df = pre_pipeline.transform(test_df)
                if model_type_name in TABNET_MODEL_TYPES:
                    test_df = test_df.values
                columns = test_df.columns
            else:
                columns = []
                report_title = ""
                test_df = None

            test_preds, test_probs = report_model_perf(
                targets=target.loc[test_idx],
                columns=columns,
                df=test_df,
                model_name="TEST " + model_name,
                model=model,
                target_label_encoder=target_label_encoder,
                preds=test_preds,
                probs=test_probs,
                figsize=figsize,
                report_title=report_title,
                nbins=nbins,
                print_report=print_report,
                show_perf_chart=show_perf_chart,
                show_fi=show_fi,            
                subgroups=subgroups,
                subset_index=test_idx,
                custom_ice_metric=custom_ice_metric,
                custom_rice_metric=custom_rice_metric,
                metrics=metrics['test']
            )

            if include_confidence_analysis:
                """Separate analysis: having original dataset, and test predictions made by a trained model, 
                find what original factors are the most discriminative regarding prediction accuracy. for that, 
                training a meta-model on test set could do it. use original features, as targets use prediction-ground truth, 
                train a regression boosting & check its feature importances."""

                # for (any, even multiclass) classification, targets are probs of ground truth classes
                if test_df is not None:
                    confidence_model=CatBoostRegressor(verbose=0,eval_fraction=0.1,task_type=("GPU" if CUDA_IS_AVAILABLE else "CPU"))
                    
                    if model_type_name == type(confidence_model).__name__:
                        fit_params_copy=copy.copy(fit_params)
                        if 'eval_set' in fit_params_copy: del fit_params_copy['eval_set']
                    else:
                        fit_params_copy={}
                    
                    if 'cat_features' not in fit_params_copy:
                        fit_params_copy[ 'cat_features']=test_df.head().select_dtypes('category').columns.tolist()

                    fit_params_copy['plot']=False

                    confidence_model.fit(test_df,test_probs[np.arange(test_probs.shape[0]), target.loc[test_idx]], **fit_params_copy)

                    if confidence_analysis_use_shap:
                        explainer = shap.TreeExplainer(confidence_model)
                        shap_values = explainer(test_df)
                        shap.plots.beeswarm(shap_values, max_display=confidence_analysis_max_features, color=plt.get_cmap(confidence_analysis_cmap), alpha=confidence_analysis_alpha, 
                                            color_bar_label=confidence_analysis_ylabel,show=False)
                        plt.xlabel(confidence_analysis_title)
                        plt.show()
                    else:
                        plot_model_feature_importances(model=confidence_model,columns=test_df.columns,model_name=confidence_analysis_title,
                                                    num_factors=confidence_analysis_max_features,figsize=(figsize[0]*0.7,figsize[1]/2))
    
    collect()
    
    return SimpleNamespace(model=model, test_preds=test_preds, test_probs=test_probs, val_preds=val_preds,
                 val_probs=val_probs, train_preds=train_preds, train_probs=train_probs,
                   columns=columns, pre_pipeline=pre_pipeline, metrics=metrics)

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
    nbins: int = 100,
    print_report: bool = True,
    show_perf_chart: bool = True,
    show_fi: bool = True,
    custom_ice_metric: Callable = None,
    custom_rice_metric: Callable = None,
    metrics:dict=None,
):
    if probs is not None or is_classifier(model):
        return report_probabilistic_model_perf(
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
            show_fi=show_fi,
            custom_ice_metric=custom_ice_metric,
            custom_rice_metric=custom_rice_metric,
            metrics=metrics,
        )
    else:

        return report_regression_model_perf(
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
            show_fi=show_fi,
            metrics=metrics,
        )
    
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
    show_fi: bool = True,
    metrics:dict=None,
):
    """Detailed performance report (usually on a test set)."""

    if preds is None:
        preds = model.predict(df)

    if isinstance(targets, pd.Series):
        targets = targets.values        

    if show_fi:
            plot_model_feature_importances(model=model,columns=columns,model_name=model_name,figsize=(15,10))
    
    if print_report:

        print(report_title)
        print(f"mean_absolute_error: {mean_absolute_error(y_true=targets,y_pred=preds):.{report_ndigits}f}")
        print(f"max_error: {max_error(y_true=targets,y_pred=preds):.{report_ndigits}f}")
        print(f"mean_absolute_percentage_error: {mean_absolute_percentage_error(y_true=targets,y_pred=preds):.{report_ndigits}f}")
        print(f"root_mean_squared_error: {root_mean_squared_error(y_true=targets,y_pred=preds):.{report_ndigits}f}")
    
    if subgroups:
        robustness_report = compute_robustness_metrics(
            subgroups=subgroups, subset_index=subset_index, y_true=targets, y_pred=preds, 
            metrics={'MAE':mean_absolute_error,'MAPE':mean_absolute_percentage_error},
            metrics_higher_is_better={'MAE':False,'MAPE':False},                
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
    nbins: int = 100,
    print_report: bool = True,
    show_perf_chart: bool = True,
    show_fi: bool = True,
    custom_ice_metric: Callable = None,
    custom_rice_metric: Callable = None,
    metrics:dict=None,
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

    integral_error = custom_ice_metric(y_true=targets, y_score=probs)
    if custom_rice_metric and custom_rice_metric!=custom_ice_metric:
        robust_integral_error = custom_rice_metric(y_true=targets, y_score=probs)

    if not classes:
        if model is not None:
            classes = model.classes_
        elif target_label_encoder:
            classes = np.arange(len(target_label_encoder.classes_)).tolist()
    
    true_classes = []
    for class_id, class_name in enumerate(classes):
        if str(class_name).isnumeric() and target_label_encoder:
            str_class_name = str(target_label_encoder.classes_[class_name])
        else:
            str_class_name = str(class_name)
        true_classes.append(str_class_name)

        y_true, y_score = (targets == class_name), probs[:, class_id]

        if len(classes) == 2 and class_id == 0:
            continue

        title = model_name
        if len(classes) != 2:
            title += "-" + str_class_name
        title += "\n" + f" ICE={integral_error:.4f}"
        if custom_rice_metric and custom_rice_metric!=custom_ice_metric:
            title +=f", RICE={robust_integral_error:.4f}" 

        brier_loss, calibration_mae, calibration_std, calibration_coverage, roc_auc, pr_auc, fig = fast_calibration_report(
            y_true=y_true,
            y_pred=y_score,
            title=title,
            figsize=figsize,
            show_plots=show_perf_chart,
            show_roc_auc_in_title=True,
            show_logloss_in_title=True,
            show_pr_auc_in_title=True,
            use_weights=use_weights,
            ndigits=calib_report_ndigits,
            verbose=verbose,
            nbins=nbins,            
        )

        if print_report:

            calibs.append(
                f"\t{str_class_name}: MAE{'W' if use_weights else ''}={calibration_mae * 100:.{calib_report_ndigits}f}%, STD={calibration_std * 100:.{calib_report_ndigits}f}%, COV={calibration_coverage * 100:.0f}%"
            )
            pr_aucs.append(f"{str_class_name}={pr_auc:.{report_ndigits}f}")
            roc_aucs.append(f"{str_class_name}={roc_auc:.{report_ndigits}f}")
            brs.append(f"{str_class_name}={brier_loss * 100:.{report_ndigits}f}%")

        if metrics is not None:
            class_metrics=dict(roc_auc=roc_auc,pr_auc=pr_auc,calibration_mae=calibration_mae,calibration_std=calibration_std,brier_loss=brier_loss,integral_error=integral_error)
            if custom_rice_metric and custom_rice_metric!=custom_ice_metric:
                class_metrics['robust_integral_error'] = robust_integral_error
            metrics.update({class_id:class_metrics})        

    if show_fi:
        plot_model_feature_importances(model=model,columns=columns,model_name=model_name,figsize=(15,10))

    if print_report:

        print(report_title)
        print(classification_report(targets, preds, zero_division=0, digits=report_ndigits))
        print(f"ROC AUCs: {', '.join(roc_aucs)}")
        print(f"PR AUCs: {', '.join(pr_aucs)}")
        print(f"CALIBRATIONS: \n{', '.join(calibs)}")
        print(f"BRIER LOSS: \n\t{', '.join(brs)}")
        print(f"INTEGRAL ERROR: {integral_error:.4f}")
        if custom_rice_metric and custom_rice_metric!=custom_ice_metric:
            print(f"ROBUST INTEGRAL ERROR: {robust_integral_error:.4f}")

    if subgroups:
        robustness_report = compute_robustness_metrics(
            subgroups=subgroups, subset_index=subset_index, y_true=y_true, y_pred=probs, 
            metrics={'ICE':custom_ice_metric,'ROC AUC':fast_roc_auc},
            metrics_higher_is_better={'ICE':False,'ROC AUC':True},                
        )
        if robustness_report is not None:
            if print_report:
                display(robustness_report.style.set_caption("ML perf robustness by group"))
            if metrics is not None:
                metrics.update(dict(robustness_report=robustness_report))

    return preds, probs

def plot_model_feature_importances(model:object,columns:Sequence,model_name:str=None,num_factors:int=40,figsize: tuple = (15, 10),positive_fi_only:bool=False):
    
    if isinstance(model, Pipeline):
        model = model.steps[-1][1]
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        feature_importances = model.coef_[-1, :]
    else:
        feature_importances = None

    if feature_importances is not None:
        try:
            plot_feature_importance(
                feature_importances=feature_importances,
                columns=columns,
                kind=model_name,
                figsize=figsize,
                positive_fi_only=False,
                n=num_factors,
            )
        except Exception:
            logger.warning("Could not plot feature importances. Maybe data shape is changed within a pipeline?")

def get_sample_weights_by_recency(date_series: pd.Series, min_weight: float = 1.0, weight_drop_per_year: float = 0.1)->np.ndarray:

    span = (date_series.max() - date_series.min()).days
    max_drop = np.log(span) * weight_drop_per_year

    sample_weight = min_weight + max_drop - np.log((date_series.max() - date_series).dt.days) * weight_drop_per_year

    return sample_weight

def configure_training_params(df:pd.DataFrame,target:pd.Series,train_idx:np.ndarray,val_idx:np.ndarray,test_idx:np.ndarray,robustness_features:Sequence=[],                              
                              target_label_encoder:object=None,sample_weight:np.ndarray=None,has_time:bool=True,prefer_gpu_configs:bool=True,
                              use_robust_eval_metric:bool=False,nbins:int=100,use_regression:bool=False,cont_nbins:int=6,
                              max_runtime_mins:float=60*1,max_noimproving_iters:int=10,**config_kwargs):
    
    cat_features=df.head().select_dtypes(('category','object')).columns.tolist()
    
    if cat_features:
        prepare_df_for_catboost(df=df,cat_features=cat_features)
    
    ensure_dataframe_float32_convertability(df)

    if robustness_features:
        subgroups=create_robustness_subgroups(df,features=robustness_features,cont_nbins=cont_nbins)
    else:
        subgroups=None
    
    if use_robust_eval_metric:
        indexed_subgroups=create_robustness_subgroups_indices(subgroups=subgroups, train_idx=train_idx, val_idx=val_idx, group_weights = {}, cont_nbins=cont_nbins)
    else:
        indexed_subgroups=None
    
    cpu_configs=get_training_configs(has_time=has_time,has_gpu=False,nbins=nbins,subgroups=indexed_subgroups,**config_kwargs)
    gpu_configs=get_training_configs(has_time=has_time,has_gpu=None,nbins=nbins,subgroups=indexed_subgroups,**config_kwargs)
    
    configs=gpu_configs if prefer_gpu_configs else cpu_configs

    common_params=dict(nbins=nbins,subgroups=subgroups,sample_weight=sample_weight,df=df,target=target,train_idx=train_idx,test_idx=test_idx,val_idx=val_idx,target_label_encoder=target_label_encoder,custom_ice_metric=configs.integral_calibration_error,custom_rice_metric=configs.final_integral_calibration_error)

    common_cb_params=dict(model=TransformedTargetRegressor(CatBoostRegressor(**configs.CB_REGR),transformer=PowerTransformer()) if use_regression else CatBoostClassifier(**configs.CB_CALIB_CLASSIF),fit_params=dict(plot=True,cat_features=cat_features))

    common_xgb_params=dict(model=XGBRegressor(**configs.XGB_GENERAL_PARAMS) if use_regression else XGBClassifier(**configs.XGB_CALIB_CLASSIF),fit_params=dict(verbose=False))

    if cat_features:
        common_lgb_params=dict(model=LGBMRegressor(**cpu_configs.LGB_GENERAL_PARAMS) if use_regression else LGBMClassifier(**cpu_configs.LGB_GENERAL_PARAMS),fit_params=dict(eval_metric=configs.lgbm_integral_calibration_error))
    else:
        common_lgb_params=dict(model=LGBMRegressor(**gpu_configs.LGB_GENERAL_PARAMS) if use_regression else LGBMClassifier(**gpu_configs.LGB_GENERAL_PARAMS),fit_params=dict(eval_metric=configs.lgbm_integral_calibration_error))
    
    params=configs.COMMON_RFECV_PARAMS.copy()
    params['max_runtime_mins']=max_runtime_mins
    params['max_noimproving_iters']=max_noimproving_iters

    cb_rfecv = RFECV(
        estimator=CatBoostRegressor(**configs.CB_REGR) if use_regression else CatBoostClassifier(**configs.CB_CALIB_CLASSIF),
        fit_params=dict(plot=False),
        cat_features=cat_features,
        scoring=make_scorer(score_func=mean_absolute_error, needs_proba=False, needs_threshold=False, greater_is_better=False) if use_regression else make_scorer(score_func=configs.fs_and_hpt_integral_calibration_error, needs_proba=True, needs_threshold=False, greater_is_better=False),
        **params
    )

    lgb_rfecv = RFECV(
        estimator=LGBMRegressor(**configs.LGB_GENERAL_PARAMS) if use_regression else LGBMClassifier(**configs.LGB_GENERAL_PARAMS),
        fit_params=dict(eval_metric=configs.lgbm_integral_calibration_error),
        cat_features=cat_features,
        scoring=make_scorer(score_func=mean_absolute_error, needs_proba=False, needs_threshold=False, greater_is_better=False) if use_regression else make_scorer(score_func=configs.fs_and_hpt_integral_calibration_error, needs_proba=True, needs_threshold=False, greater_is_better=False),
        **params
    )

    xgb_rfecv = RFECV(
        estimator=XGBRegressor(**configs.XGB_GENERAL_PARAMS) if use_regression else XGBClassifier(**configs.XGB_CALIB_CLASSIF),
        fit_params=dict(verbose=False),
        cat_features=cat_features,
        scoring=make_scorer(score_func=mean_absolute_error, needs_proba=False, needs_threshold=False, greater_is_better=False) if use_regression else make_scorer(score_func=configs.fs_and_hpt_integral_calibration_error, needs_proba=True, needs_threshold=False, greater_is_better=False),
        **params
    )    
        
    return common_params,common_cb_params,common_lgb_params,common_xgb_params,cb_rfecv,lgb_rfecv,xgb_rfecv,cpu_configs,gpu_configs

def post_calibrate_model(original_model:object,target_series:pd.Series,target_label_encoder:object,val_idx:np.ndarray,
                         test_idx:np.ndarray,configs:dict,calib_set_size:int=2000,nbins:int=10,show_val:bool=False,
                         meta_model:object=None,
                **fit_params):

    if meta_model is None:
        meta_model=CatBoostClassifier(iterations=3000,
                        verbose=False,
                        has_time=False,
                        learning_rate=0.2,
                        eval_fraction=0.1,
                        task_type="GPU",        
                        early_stopping_rounds=400,
                        eval_metric= CB_EVAL_METRIC(metric=configs.integral_calibration_error,higher_is_better=False),
                        custom_metric="AUC"
                )
    model, test_preds, test_probs, val_preds, val_probs, columns, pre_pipeline, metrics=original_model
    
    
    meta_model.fit(test_probs[:calib_set_size,1].reshape(-1, 1),target_series.loc[test_idx].values[:calib_set_size],**fit_params)    
    
    if show_val:
        meta_val_probs=meta_model.predict_proba(val_probs[:,1].reshape(-1,1))
        _=report_model_perf(
                        targets=target_series.loc[val_idx],
                        columns=columns,
                        df=None,
                        model_name="VAL",
                        model=None,
                        target_label_encoder=target_label_encoder,
                        preds=val_preds,
                        probs=val_probs,
                        report_title="",
                        nbins=10,
                        print_report=False,
                        show_fi=False,
                        custom_ice_metric=configs.integral_calibration_error,
                    )        
        _=report_model_perf(
                        targets=target_series.loc[val_idx],
                        columns=columns,
                        df=None,
                        model_name="VAL fixed",
                        model=None,
                        target_label_encoder=target_label_encoder,
                        preds=val_preds,
                        probs=meta_val_probs,
                        report_title="",
                        nbins=10,
                        print_report=False,
                        show_fi=False,
                        custom_ice_metric=configs.integral_calibration_error,
                    )        
    
    meta_test_probs=meta_model.predict_proba(test_probs[:,1].reshape(-1,1))
    
    _=report_model_perf(
                    targets=target_series.loc[test_idx],
                    columns=columns,
                    df=None,
                    model_name="TEST original",
                    model=None,
                    target_label_encoder=target_label_encoder,
                    preds=test_preds,
                    probs=test_probs,
                    report_title="",
                    nbins=10,
                    print_report=False,
                    show_fi=False,
                    custom_ice_metric=configs.integral_calibration_error,
                )

    _=report_model_perf(
                    targets=target_series.loc[test_idx].values[calib_set_size:],
                    columns=columns,
                    df=None,
                    model_name="TEST fixed ",
                    model=None,
                    target_label_encoder=target_label_encoder,
                    preds=(meta_test_probs[calib_set_size:,1]>0.5).astype(int),
                    probs=meta_test_probs[calib_set_size:,:],
                    report_title="",
                    nbins=10,
                    print_report=True,
                    show_fi=False,
                    custom_ice_metric=configs.integral_calibration_error,
                )    
    
    _=report_model_perf(
                    targets=target_series.loc[test_idx].values[:calib_set_size],
                    columns=columns,
                    df=None,
                    model_name="TEST fixed lucky",
                    model=None,
                    target_label_encoder=target_label_encoder,
                    preds=(meta_test_probs[:calib_set_size:,1]>0.5).astype(int),
                    probs=meta_test_probs[:calib_set_size,:],
                    report_title="",
                    nbins=10,
                    print_report=True,
                    show_fi=False,
                    custom_ice_metric=configs.integral_calibration_error,
                )    
    
    
    return model, test_preds, meta_test_probs, val_preds, meta_val_probs, columns, pre_pipeline, metrics
