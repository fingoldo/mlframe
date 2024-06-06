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

import numba
from numba.cuda import is_available as is_cuda_available

import copy
import joblib
from os.path import join, exists
from collections import defaultdict

from IPython.display import display

from catboost import CatBoostRegressor

import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

from pyutilz.pythonlib import prefix_dict_elems
from pyutilz.system import ensure_dir_exists, tqdmu

from mlframe.helpers import get_model_best_iter
from mlframe.feature_importance import plot_feature_importance
from mlframe.feature_selection.wrappers import RFECV, VotesAggregation, OptimumSearch
from mlframe.metrics import fast_roc_auc, fast_calibration_report, compute_probabilistic_multiclass_error, CB_INTEGRAL_CALIB_ERROR
from mlframe.stats import get_tukey_fences_multiplier_for_quantile
from mlframe.metrics import create_robustness_subgroups
from pyutilz.pythonlib import sort_dict_by_value


from types import SimpleNamespace

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
    iterations: int = 2000,
    early_stopping_rounds: int = 0,
    use_explicit_early_stopping: bool = True,
    has_time: bool = True,
    learning_rate: float = 0.1,
    mae_weight: float = 1.0,
    std_weight: float = 1.0,
    roc_auc_weight: float = 0.7,
    brier_loss_weight: float = 0.5,
    min_roc_auc: float = 0.51,
    roc_auc_penalty: float = 0.2,
    use_weighted_calibration: bool = True,
    weight_by_class_npositives: bool = False,
    nbins: int = 100,
    validation_fraction: float = 0.1,
    def_regr_metric: str = "MSE",
    def_classif_metric: str = "AUC",
    cv=None,
    cv_n_splits: int = 3,
    random_seed=None,
    verbose: int = 0,
) -> tuple:
    """Returns approximately same training configs for different types of models,
    based on general params supplied like learning rate, task type, time budget.
    Useful for more or less fair comparison between different models on the same data/task, and their upcoming ensembling.
    This procedure is good for manual EDA and getting the feeling of what ML models are capable of for a particular task.
    """

    if not early_stopping_rounds:
        early_stopping_rounds = max(2, iterations // 4)

    def integral_calibration_error(y_true, y_score, verbose: bool = False):
        err = compute_probabilistic_multiclass_error(
            y_true=y_true,
            y_score=y_score,
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

    def lgbm_integral_calibration_error(y_true, y_score):
        metric_name = "integral_calibration_error"
        value = integral_calibration_error(y_true, y_score)
        is_higher_better = False

        return metric_name, value, is_higher_better

    def neg_ovr_roc_auc_score(*args, **kwargs):
        return -roc_auc_score(*args, **kwargs, multi_class="ovr")

    catboost_custom_classif_metrics = ["AUC", "BrierScore", "PRAUC"]
    catboost_custom_regr_metrics = ["MSE", "MAE"]

    CB_GENERAL = dict(
        iterations=iterations,
        verbose=verbose,
        has_time=has_time,
        learning_rate=learning_rate,
        eval_fraction=(0.0 if use_explicit_early_stopping else validation_fraction),
        task_type=("GPU" if CUDA_IS_AVAILABLE else "CPU"),
        eval_metric=def_classif_metric,
        custom_metric=catboost_custom_classif_metrics,
        early_stopping_rounds=early_stopping_rounds,
        random_seed=random_seed,
    )

    CB_GENERAL_CLASSIF = CB_GENERAL.copy()
    CB_GENERAL_CLASSIF.update({"eval_metric": def_classif_metric, "custom_metric": catboost_custom_classif_metrics})

    CB_CPU_CLASSIF = CB_GENERAL_CLASSIF.copy()
    CB_CPU_CLASSIF.update({"task_type": "CPU"})

    CB_CALIB_CLASSIF = CB_GENERAL_CLASSIF.copy()
    CB_CALIB_CLASSIF.update(
        {
            "eval_metric": CB_INTEGRAL_CALIB_ERROR(
                mae_weight=mae_weight,
                std_weight=std_weight,
                brier_loss_weight=brier_loss_weight,
                roc_auc_weight=roc_auc_weight,
                min_roc_auc=min_roc_auc,
                roc_auc_penalty=roc_auc_penalty,
                use_weighted_calibration=use_weighted_calibration,
                weight_by_class_npositives=weight_by_class_npositives,
                nbins=nbins,
            )
        }
    )

    LGBM_GENERAL_PARAMS = dict(
        n_estimators=iterations,
        early_stopping_rounds=early_stopping_rounds,
        device_type=("gpu" if CUDA_IS_AVAILABLE else "cpu"),
        verbose=int(verbose),
        random_state=random_seed,
    )

    LGBM_CPU_PARAMS = LGBM_GENERAL_PARAMS.copy()
    LGBM_CPU_PARAMS.update({"device_type": "cpu"})

    XGB_GENERAL_CLASSIF = dict(
        n_estimators=iterations,
        objective="binary:logistic",
        enable_categorical=True,
        max_cat_threshold=1000,
        tree_method="hist",
        device=("cuda" if CUDA_IS_AVAILABLE else "cpu"),
        eval_metric=neg_ovr_roc_auc_score,
        early_stopping_rounds=early_stopping_rounds,
        random_seed=random_seed,
        verbosity=int(verbose),
    )

    XGB_CALIB_CLASSIF = XGB_GENERAL_CLASSIF.copy()
    XGB_CALIB_CLASSIF.update({"eval_metric": integral_calibration_error})

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
        smooth_perf=3,
        max_refits=None,
        max_runtime_mins=60 * 2,
        max_noimproving_iters=40,
        # frac=0.2,
    )

    return SimpleNamespace(
        integral_calibration_error=integral_calibration_error,
        lgbm_integral_calibration_error=lgbm_integral_calibration_error,
        fs_and_hpt_integral_calibration_error=fs_and_hpt_integral_calibration_error,
        CB_GENERAL_CLASSIF=CB_GENERAL_CLASSIF,
        CB_CPU_CLASSIF=CB_CPU_CLASSIF,
        CB_CALIB_CLASSIF=CB_CALIB_CLASSIF,
        LGBM_GENERAL_PARAMS=LGBM_GENERAL_PARAMS,
        LGBM_CPU_PARAMS=LGBM_CPU_PARAMS,
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
    test_preds: Optional[np.ndarray] = None,
    test_probs: Optional[np.ndarray] = None,
    val_preds: Optional[np.ndarray] = None,
    val_probs: Optional[np.ndarray] = None,
    custom_ice_metric: Callable = None,
    subgroups: dict = None,
    figsize: tuple = (15, 5),
    verbose: bool = True,
    use_cache: bool = False,
    nbins: int = 100,
    show_val_chart: bool = True,
    data_dir: str = DATA_DIR,
    models_subdir: str = MODELS_SUBDIR,
    include_confidence_analysis:bool=True,
):
    """Trains & evaluates given model/pipeline on train/test sets.
    Supports feature selection via pre_pipeline.
    Supports early stopping via val_idx.
    Optionally fumps resulting model & test set predictions into the models dir, and loads back by model name on the next call, to save time.
    """

    if not custom_ice_metric:
        custom_ice_metric = compute_probabilistic_multiclass_error

    ensure_dir_exists(join(data_dir, models_subdir))
    model_file_name = join(data_dir, models_subdir, f"{model_name}.dump")

    if use_cache and exists(model_file_name):
        logger.info(f"Loading model from file {model_file_name}")
        model, *_, pre_pipeline = joblib.load(model_file_name)

    real_drop_columns = [col for col in drop_columns + default_drop_columns if col in df.columns]

    if fit_params and isinstance(model, Pipeline):
        model_type_name = type(model.named_steps["est"]).__name__  # type(model.steps[-1]).__name__
    else:
        model_type_name = type(model).__name__

    if fit_params is None:
        fit_params = {}

    if df is not None:
        if val_idx is not None:
            train_df = df.loc[train_idx].drop(columns=real_drop_columns)
            val_df = df.loc[val_idx].drop(columns=real_drop_columns)
        else:
            train_df = df.loc[train_idx].drop(columns=real_drop_columns)

    if model and pre_pipeline:
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

    if model and fit_params:
        if "cat_features" in fit_params:
            fit_params["cat_features"] = [col for col in fit_params["cat_features"] if col in train_df.columns]

    if fit_params and isinstance(model, Pipeline):
        fit_params = prefix_dict_elems(fit_params, "est__")

    if model:
        if (not use_cache) or (not exists(model_file_name)):
            if sample_weight is not None:
                sample_weight = sample_weight.loc[train_idx].values
            display(train_df.tail(3).style.set_caption("Features sample"))
            if model_type_name in TABNET_MODEL_TYPES:
                model.fit(train_df.values, target.loc[train_idx].values, sample_weight=sample_weight, **fit_params)
            else:
                model.fit(train_df, target.loc[train_idx], sample_weight=sample_weight, **fit_params)
            if model:
                # get number of the best iteration
                try:
                    best_iter = get_model_best_iter(model)
                    if best_iter:
                        print(f"es_best_iter: {best_iter:_}")
                except Exception as e:
                    logger.warning(e)

    if verbose:
        if show_val_chart and val_idx is not None:
            if df is None:
                val_df = None
                columns = []
            else:
                df.loc[val_idx].drop(columns=real_drop_columns)
                columns = val_df.columns

            val_preds, val_probs = report_probabilistic_model_perf(
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
                print_report=True,
                show_fi=False,
                subgroups=subgroups,
                subset_index=val_idx,
                custom_ice_metric=custom_ice_metric,
            )

        if df is not None:
            report_title = f"Training {model_name} model on {train_df.shape[1]} feature(s): {', '.join(train_df.columns.to_list())}, {len(train_df):_} records"
            test_df = df.loc[test_idx].drop(columns=real_drop_columns)
            if model and pre_pipeline:
                test_df = pre_pipeline.transform(test_df)
            if model_type_name in TABNET_MODEL_TYPES:
                test_df = test_df.values
            columns = test_df.columns
        else:
            columns = []
            report_title = ""
            test_df = None

        test_preds, test_probs = report_probabilistic_model_perf(
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
            subgroups=subgroups,
            subset_index=test_idx,
            custom_ice_metric=custom_ice_metric,
        )

        if include_confidence_analysis:
            """Separate analysis: having original dataset, and test preditions made by a trained model, 
            find what original factors are the most discriminative regarding prediction accuracy. for that, 
            training a meta-model on test set could do it. use original features, as targets use prediction-ground truth, 
            train a regression boosting & check its feature importances."""

            # for (any, even multiclass) classification, targets are probs of ground truth classes
            if test_df is not None:
                confidence_clf=CatBoostRegressor(verbose=0,eval_fraction=0.1,task_type=("GPU" if CUDA_IS_AVAILABLE else "CPU"))
                
                fit_params_copy=copy.copy(fit_params)
                if 'eval_set' in fit_params_copy: del fit_params_copy['eval_set']
                fit_params_copy['plot']=False

                confidence_clf.fit(test_df,test_probs[np.arange(test_probs.shape[0]), target.loc[test_idx]], **fit_params_copy)
                plot_model_feature_importances(model=confidence_clf,columns=test_df.columns,model_name="Confidence Analysis [factors that change our accuracy]",num_factors=5,figsize=(figsize[0]/2,figsize[1]/2))

    return model, test_preds, test_probs, val_preds, val_probs, columns, pre_pipeline


def report_probabilistic_model_perf(
    targets: Union[np.ndarray, pd.Series],
    columns: Sequence,
    model_name: str,
    model: ClassifierMixin,
    subgroups: dict = None,
    subset_index: np.ndarray = None,
    digits: int = 4,
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
    show_fi: bool = True,
    custom_ice_metric: Callable = None,
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

    integral_error = custom_ice_metric(y_true=targets, y_score=probs, verbose=False)

    if not classes:
        if model:
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

        brier_loss, calibration_mae, calibration_std, calibration_coverage, roc_auc, pr_auc, fig = fast_calibration_report(
            y_true=y_true,
            y_pred=y_score,
            title=title,
            figsize=figsize,
            show_roc_auc_in_title=True,
            show_logloss_in_title=True,
            show_pr_auc_in_title=True,
            use_weights=use_weights,
            ndigits=calib_report_ndigits,
            verbose=verbose,
            nbins=nbins,
        )
        calibs.append(
            f"\t{str_class_name}: MAE{'W' if use_weights else ''}={calibration_mae * 100:.{calib_report_ndigits}f}%, STD={calibration_std * 100:.{calib_report_ndigits}f}%, COV={calibration_coverage * 100:.0f}%"
        )

        pr_aucs.append(f"{str_class_name}={pr_auc:.{digits}f}")
        roc_aucs.append(f"{str_class_name}={roc_auc:.{digits}f}")
        brs.append(f"{str_class_name}={brier_loss * 100:.{digits}f}%")

        if show_fi: plot_model_feature_importances(model=model,columns=columns,model_name=model_name,figsize=figsize)

    if print_report:
        print(report_title)
        print(classification_report(targets, preds, zero_division=0, digits=digits))
        print(f"ROC AUCs: {', '.join(roc_aucs)}")
        print(f"PR AUCs: {', '.join(pr_aucs)}")
        print(f"CALIBRATIONS: \n{', '.join(calibs)}")
        print(f"BRIER LOSS: \n\t{', '.join(brs)}")
        print(f"INTEGRAL ERROR: {integral_error:.4f}")

        if subgroups:
            robustness_report = compute_robustness_metrics(
                subgroups=subgroups, subset_index=subset_index, y_true=y_true, y_pred=probs, 
                metrics={'ICE':custom_ice_metric,'ROC AUC':fast_roc_auc},
                metrics_higher_is_better={'ICE':False,'ROC AUC':True},                
            )
            if robustness_report is not None:
                display(robustness_report)

    return preds, probs

def plot_model_feature_importances(model:object,columns:Sequence,model_name:str=None,num_factors:int=20,figsize: tuple = (15, 5),positive_fi_only:bool=False):
    
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

def compute_robustness_metrics(
    metrics: dict,
    metrics_higher_is_better: dict,
    subgroups: dict,
    subset_index: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cont_nbins: int = 3,
    top_n: int = 5,    
) -> pd.DataFrame:
    """* is added to the bin name if bin's metric is an outlier (compted using Tukey's fence & IQR)."""
    if subgroups:

        res = []
        quantile = 0.25
        quantiles_to_compute = [0.5 - quantile, 0.5, 0.5 + quantile]
        tukey_mult = get_tukey_fences_multiplier_for_quantile(quantile=quantile, sd_sigma=2.7)

        for group_name, subgrouping in subgroups.items():
            if group_name in ("**ORDER**", "**RANDOM**"):
                # settings = subgrouping.get("settings")
                l = len(y_true)
                step_size = l // cont_nbins
                bins = np.empty(shape=l, dtype=np.int16)
                start = 0
                unique_bins = range(cont_nbins)
                for i in unique_bins:
                    bins[start : start + step_size] = i
                    start += step_size
                if group_name == "**RANDOM**":
                    np.random.shuffle(bins)
            else:
                bins = subgrouping.get("bins")
                if bins is not None:
                    assert subset_index is not None
                    bins = bins.loc[subset_index]
                bins_names = subgrouping.get("bins_names")
                unique_bins = None
            
            perfs =defaultdict(dict)
            npoints = []
            if unique_bins is None:
                if isinstance(bins,pd.Series):
                    unique_bins = bins.unique()
                else:
                    unique_bins = np.unique(bins)
            for bin_name in unique_bins:
                idx = bins == bin_name
                n_points = idx.sum()
                if n_points:
                    npoints.append(n_points)
                    for metric_name,metric_func in metrics.items():
                        metric_value = metric_func(y_true[idx],y_pred[idx, :])
                        perfs[metric_name][f"{bin_name} [{n_points}]"] = metric_value

            for metric_name,metric_perf in perfs.items():

                metric_perf = sort_dict_by_value(metric_perf)
                npoints = np.array(npoints)
                line = dict(
                    factor=group_name,metric=metric_name, nbins=len(unique_bins), npoints_from=npoints.min(), npoints_median=int(np.median(npoints)), npoints_to=npoints.max()
                )

                # -----------------------------------------------------------------------------------------------------------------------------------------------------
                # Compute quantiles of the metric value.
                # -----------------------------------------------------------------------------------------------------------------------------------------------------

                performances = np.array(list(metric_perf.values()))
                quantiles = np.quantile(performances, q=quantiles_to_compute)
                iqr = quantiles[-1] - quantiles[0]
                min_boundary = quantiles[0] - tukey_mult * iqr
                max_boundary = quantiles[-1] + tukey_mult * iqr

                """
                for q, value in zip(quantiles_to_compute, quantiles):
                    line[f"q{q:.2f}"] = value
                """

                line[f"metric_mean"] = quantiles.mean()
                line[f"metric_std"] = quantiles.std()

                # -----------------------------------------------------------------------------------------------------------------------------------------------------
                # Show top-n best/worst groups. postfix * means metric value for the bin is an outlier.
                # -----------------------------------------------------------------------------------------------------------------------------------------------------

                l = len(metric_perf)
                real_top_n = min(l // 2, top_n)

                for i, (bin_name, metric_value) in enumerate(metric_perf.items()):
                    if metric_value < min_boundary or metric_value > max_boundary:
                        postfix = "*"
                    else:
                        postfix = ""
                    if i < real_top_n:
                        if metrics_higher_is_better[metric_name]:
                            line["bin-worst-"+ str(i + 1)] = f"{bin_name}: {metric_value:.3f}{postfix}"
                        else:
                            line["bin-best-"+ str(i + 1)] = f"{bin_name}: {metric_value:.3f}{postfix}"
                    elif i >= l - real_top_n:
                        if metrics_higher_is_better[metric_name]:
                            line["bin-best-" + str(l - i)] = f"{bin_name}: {metric_value:.3f}{postfix}"
                        else:
                            line["bin-worst-" + str(l - i)] = f"{bin_name}: {metric_value:.3f}{postfix}"

                res.append(line)
        if res:
            res=pd.DataFrame(res).set_index(['factor' ,'nbins','npoints_from'	,'npoints_median',	'npoints_to','metric'])
            return res.reindex(sorted(res.columns), axis=1).style.set_caption("ML performance by group")
