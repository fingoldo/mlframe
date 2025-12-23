"""
Model evaluation and reporting functions for mlframe training.

This module contains functions for evaluating trained models and generating
performance reports for both regression and classification tasks.

Functions:
    evaluate_model: High-level model evaluation interface
    report_model_perf: Unified performance report for classifiers and regressors
    report_regression_model_perf: Detailed regression performance report
    report_probabilistic_model_perf: Detailed classification performance report
    get_model_feature_importances: Extract feature importances from a model
    plot_model_feature_importances: Plot feature importances
    post_calibrate_model: Post-calibrate a model using a meta-model
"""

import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Union, Callable

# =============================================================================
# Constants
# =============================================================================

DEFAULT_RANDOM_SEED = 42
DEFAULT_BINARY_THRESHOLD = 0.5
DEFAULT_PLOT_SAMPLE_SIZE = 500
DEFAULT_REPORT_NDIGITS = 4
DEFAULT_CALIB_REPORT_NDIGITS = 2
DEFAULT_NBINS = 10
DEFAULT_FIGSIZE = (15, 5)
DEFAULT_FI_FIGSIZE = (15, 10)

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    max_error,
    r2_score,
    classification_report,
)

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
        output_errors = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight, multioutput="raw_values"))
        if isinstance(multioutput, str):
            if multioutput == "raw_values":
                return output_errors
            elif multioutput == "uniform_average":
                multioutput = None
        return np.average(output_errors, weights=multioutput)


from pyutilz.pythonlib import get_human_readable_set_size
from IPython.display import display

from mlframe.metrics import fast_calibration_report, fast_roc_auc, compute_fairness_metrics
from mlframe.feature_importance import plot_feature_importance

logger = logging.getLogger(__name__)


def report_model_perf(
    targets: Union[np.ndarray, pd.Series],
    columns: Sequence[str],
    model_name: str,
    model: Optional[Union[ClassifierMixin, RegressorMixin]],
    subgroups: Optional[Dict[str, np.ndarray]] = None,
    subset_index: Optional[np.ndarray] = None,
    report_ndigits: int = DEFAULT_REPORT_NDIGITS,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    report_title: str = "",
    use_weights: bool = True,
    calib_report_ndigits: int = DEFAULT_CALIB_REPORT_NDIGITS,
    verbose: bool = False,
    classes: Optional[Sequence] = None,
    preds: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    target_label_encoder: Optional[LabelEncoder] = None,
    nbins: int = DEFAULT_NBINS,
    print_report: bool = True,
    show_perf_chart: bool = True,
    show_fi: bool = True,
    fi_kwargs: Optional[Dict[str, Any]] = None,
    plot_file: str = "",
    custom_ice_metric: Optional[Callable] = None,
    custom_rice_metric: Optional[Callable] = None,
    metrics: Optional[Dict[str, Any]] = None,
    group_ids: Optional[np.ndarray] = None,
    n_features: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generate a unified performance report for both classifiers and regressors.

    Automatically detects model type and routes to the appropriate reporting
    function (classification or regression).

    Parameters
    ----------
    targets : np.ndarray or pd.Series
        True target values.
    columns : Sequence[str]
        Feature column names used for training.
    model_name : str
        Name of the model for display in reports.
    model : ClassifierMixin, RegressorMixin, or None
        Trained model. Can be None if preds/probs are provided.
    subgroups : dict, optional
        Dictionary mapping subgroup names to boolean arrays for fairness analysis.
    subset_index : np.ndarray, optional
        Indices to subset the data for fairness analysis.
    report_ndigits : int, default=4
        Number of decimal digits for metric reporting.
    figsize : tuple, default=(15, 5)
        Figure size for plots.
    report_title : str, default=""
        Title prefix for reports.
    use_weights : bool, default=True
        Whether to use weighted calibration metrics.
    calib_report_ndigits : int, default=2
        Decimal digits for calibration metrics.
    verbose : bool, default=False
        Enable verbose output.
    classes : Sequence, optional
        Class labels for classification.
    preds : np.ndarray, optional
        Pre-computed predictions.
    probs : np.ndarray, optional
        Pre-computed probabilities for classification.
    df : pd.DataFrame, optional
        Feature DataFrame for generating predictions.
    target_label_encoder : LabelEncoder, optional
        Encoder for target labels.
    nbins : int, default=10
        Number of bins for calibration analysis.
    print_report : bool, default=True
        Whether to print the report.
    show_perf_chart : bool, default=True
        Whether to display performance charts.
    show_fi : bool, default=True
        Whether to show feature importances.
    fi_kwargs : dict, optional
        Additional kwargs for feature importance plotting.
    plot_file : str, default=""
        Base path for saving plots.
    custom_ice_metric : Callable, optional
        Custom integral calibration error metric.
    custom_rice_metric : Callable, optional
        Custom robust integral calibration error metric.
    metrics : dict, optional
        Dictionary to store computed metrics (modified in-place).
    group_ids : np.ndarray, optional
        Group identifiers for grouped calibration analysis.

    Returns
    -------
    tuple
        (preds, probs) for classification, (preds, None) for regression.
    """
    if fi_kwargs is None:
        fi_kwargs = {}

    # Common parameters shared by both classification and regression
    common_params = dict(
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
        n_features=n_features,
    )

    if is_classifier(model) or type(model).__name__ == "NGBClassifier" or (model is None and probs is not None):
        preds, probs = report_probabilistic_model_perf(
            **common_params,
            use_weights=use_weights,
            calib_report_ndigits=calib_report_ndigits,
            classes=classes,
            probs=probs,
            target_label_encoder=target_label_encoder,
            nbins=nbins,
            custom_ice_metric=custom_ice_metric,
            custom_rice_metric=custom_rice_metric,
            group_ids=group_ids,
        )
    else:
        preds, probs = report_regression_model_perf(**common_params)

    if show_fi:
        n_cols = n_features if n_features is not None else (len(columns) if columns else 0)
        nfeatures = f"{n_cols:_}F/" if n_cols > 0 else ""
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
    columns: Sequence[str],
    model_name: str,
    model: Optional[RegressorMixin],
    subgroups: Optional[Dict[str, np.ndarray]] = None,
    subset_index: Optional[np.ndarray] = None,
    report_ndigits: int = DEFAULT_REPORT_NDIGITS,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    report_title: str = "",
    verbose: bool = False,
    preds: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    print_report: bool = True,
    show_perf_chart: bool = True,
    plot_file: str = "",
    plot_marker: str = "o",
    plot_sample_size: int = DEFAULT_PLOT_SAMPLE_SIZE,
    metrics: Optional[Dict[str, Any]] = None,
    n_features: Optional[int] = None,
) -> Tuple[np.ndarray, None]:
    """
    Generate a detailed performance report for regression models.

    Computes and optionally displays MAE, RMSE, MaxError, R² metrics,
    scatter plots of predictions vs actuals, and fairness analysis.

    Parameters
    ----------
    targets : np.ndarray or pd.Series
        True target values.
    columns : Sequence[str]
        Feature column names.
    model_name : str
        Name of the model for display.
    model : RegressorMixin or None
        Trained regression model. Can be None if preds are provided.
    subgroups : dict, optional
        Dictionary mapping subgroup names to boolean arrays for fairness analysis.
    subset_index : np.ndarray, optional
        Indices to subset the data for fairness analysis.
    report_ndigits : int, default=4
        Number of decimal digits for metric reporting.
    figsize : tuple, default=(15, 5)
        Figure size for plots.
    report_title : str, default=""
        Title prefix for reports.
    verbose : bool, default=False
        Enable verbose output.
    preds : np.ndarray, optional
        Pre-computed predictions. If None, generated from model.
    df : pd.DataFrame, optional
        Feature DataFrame for generating predictions.
    print_report : bool, default=True
        Whether to print the report.
    show_perf_chart : bool, default=True
        Whether to display performance charts.
    plot_file : str, default=""
        Path for saving the plot.
    plot_marker : str, default="o"
        Marker style for scatter plot.
    plot_sample_size : int, default=500
        Maximum number of points to plot (for performance).
    metrics : dict, optional
        Dictionary to store computed metrics (modified in-place).

    Returns
    -------
    tuple
        (preds, None) - predictions and None (no probabilities for regression).
    """
    if preds is None:
        preds = model.predict(df)

    if isinstance(targets, pd.Series):
        targets = targets.values

    MAE = mean_absolute_error(y_true=targets, y_pred=preds)
    MaxError = max_error(y_true=targets, y_pred=preds)
    R2 = r2_score(y_true=targets, y_pred=preds)
    RMSE = root_mean_squared_error(y_true=targets, y_pred=preds)

    current_metrics = dict(
        MAE=MAE,
        MaxError=MaxError,
        R2=R2,
        RMSE=RMSE,
    )
    if metrics is not None:
        metrics.update(current_metrics)

    if show_perf_chart or plot_file:
        title = report_title + " " + model_name
        n_cols = n_features if n_features is not None else (len(columns) if columns else 0)
        nfeatures = f"{n_cols:_}F/" if n_cols > 0 else ""
        title += f" [{nfeatures}{get_human_readable_set_size(len(targets))} rows]" + "\n"

        title += f" MAE={MAE:.{report_ndigits}f}"
        title += f" RMSE={RMSE:.{report_ndigits}f}"
        title += f" MaxError={MaxError:.{report_ndigits}f}"
        title += f" R2={R2:.{report_ndigits}f}"

        np.random.seed(DEFAULT_RANDOM_SEED)
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
        print(f"R2: {R2:.{report_ndigits}f}")

    if subgroups:
        fairness_report = compute_fairness_metrics(
            subgroups=subgroups,
            subset_index=subset_index,
            y_true=targets,
            y_pred=preds,
            metrics={"MAE": mean_absolute_error, "RMSE": root_mean_squared_error},
            metrics_higher_is_better={"MAE": False, "RMSE": False},
        )
        if fairness_report is not None:
            if print_report:
                display(fairness_report)
            if metrics is not None:
                metrics.update(dict(fairness_report=fairness_report))

    return preds, None


def report_probabilistic_model_perf(
    targets: Union[np.ndarray, pd.Series],
    columns: Sequence[str],
    model_name: str,
    model: Optional[ClassifierMixin],
    subgroups: Optional[Dict[str, np.ndarray]] = None,
    subset_index: Optional[np.ndarray] = None,
    report_ndigits: int = DEFAULT_REPORT_NDIGITS,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    report_title: str = "",
    use_weights: bool = True,
    calib_report_ndigits: int = DEFAULT_CALIB_REPORT_NDIGITS,
    verbose: bool = False,
    classes: Optional[Sequence] = None,
    preds: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    target_label_encoder: Optional[LabelEncoder] = None,
    nbins: int = DEFAULT_NBINS,
    print_report: bool = True,
    show_perf_chart: bool = True,
    plot_file: str = "",
    custom_ice_metric: Optional[Callable] = None,
    custom_rice_metric: Optional[Callable] = None,
    metrics: Optional[Dict[str, Any]] = None,
    group_ids: Optional[np.ndarray] = None,
    n_features: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a detailed performance report for probabilistic classification models.

    Computes and displays classification metrics including ROC AUC, PR AUC,
    calibration metrics, Brier loss, log loss, and fairness analysis.

    Parameters
    ----------
    targets : np.ndarray or pd.Series
        True target labels.
    columns : Sequence[str]
        Feature column names.
    model_name : str
        Name of the model for display.
    model : ClassifierMixin or None
        Trained classification model. Can be None if probs are provided.
    subgroups : dict, optional
        Dictionary mapping subgroup names to boolean arrays for fairness analysis.
    subset_index : np.ndarray, optional
        Indices to subset the data for fairness analysis.
    report_ndigits : int, default=4
        Number of decimal digits for metric reporting.
    figsize : tuple, default=(15, 5)
        Figure size for plots.
    report_title : str, default=""
        Title prefix for reports.
    use_weights : bool, default=True
        Whether to use weighted calibration metrics.
    calib_report_ndigits : int, default=2
        Decimal digits for calibration metrics.
    verbose : bool, default=False
        Enable verbose output.
    classes : Sequence, optional
        Class labels. If None, inferred from model or targets.
    preds : np.ndarray, optional
        Pre-computed class predictions.
    probs : np.ndarray, optional
        Pre-computed class probabilities. If None, generated from model.
    df : pd.DataFrame, optional
        Feature DataFrame for generating predictions.
    target_label_encoder : LabelEncoder, optional
        Encoder for converting numeric labels to string names.
    nbins : int, default=10
        Number of bins for calibration analysis.
    print_report : bool, default=True
        Whether to print the report.
    show_perf_chart : bool, default=True
        Whether to display calibration and performance charts.
    plot_file : str, default=""
        Base path for saving plots.
    custom_ice_metric : Callable, optional
        Custom integral calibration error metric function.
    custom_rice_metric : Callable, optional
        Custom robust integral calibration error metric function.
    metrics : dict, optional
        Dictionary to store computed metrics (modified in-place).
    group_ids : np.ndarray, optional
        Group identifiers for grouped calibration analysis.

    Returns
    -------
    tuple
        (preds, probs) - class predictions and probability arrays.
    """
    if probs is None:
        try:
            probs = model.predict_proba(df)
        except (AttributeError, TypeError, NotImplementedError) as e:
            logger.warning(f"predict_proba not available for {type(model).__name__}, using predict() instead: {e}")
            preds_fallback = model.predict(df)

            if hasattr(model, "classes_"):
                n_classes = len(model.classes_)
                class_indices = np.searchsorted(model.classes_, preds_fallback)
            else:
                n_classes = len(np.unique(preds_fallback))
                class_indices = preds_fallback.astype(int)

            probs = np.zeros((len(preds_fallback), n_classes))
            probs[np.arange(len(preds_fallback)), class_indices] = 1.0

    if preds is None:
        if probs.shape[1] == 2:
            # For binary classification, use threshold=0.5 on class 1 probability
            # This ensures consistency with calibration metrics in fast_calibration_report
            classes_ = model.classes_ if (model is not None and hasattr(model, "classes_")) else np.array([0, 1])
            preds = np.where(probs[:, 1] >= 0.5, classes_[1], classes_[0])
        else:
            preds = np.argmax(probs, axis=1)
            if model is not None and hasattr(model, "classes_"):
                preds = model.classes_[preds]

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
    robust_integral_error = None
    if custom_rice_metric and custom_rice_metric != custom_ice_metric:
        robust_integral_error = custom_rice_metric(y_true=targets, y_score=probs)

    if not classes:
        if model is not None:
            if hasattr(model, "classes_"):
                classes = model.classes_
            else:
                classes = np.unique(targets)
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
        n_cols = n_features if n_features is not None else (len(columns) if columns else 0)
        nfeatures = f"{n_cols:_}F/" if n_cols > 0 else ""
        title += f" [{nfeatures}{get_human_readable_set_size(len(y_true))} rows]"
        if custom_rice_metric and custom_rice_metric != custom_ice_metric:
            class_robust_integral_error = custom_rice_metric(y_true=y_true, y_score=y_score)
            title += f", RICE={class_robust_integral_error:.{calib_report_ndigits}f}"

        brier_loss, calibration_mae, calibration_std, calibration_coverage, roc_auc, pr_auc, ice, ll, precision, recall, f1, metrics_string, *_ = (
            fast_calibration_report(
                y_true=y_true,
                y_pred=y_score,
                use_weights=use_weights,
                nbins=nbins,
                group_ids=group_ids,
                title=title,
                figsize=figsize,
                plot_file=plot_file + "_perfplot.png" if plot_file else "",
                show_plots=show_perf_chart,
                ndigits=calib_report_ndigits,
                verbose=verbose,
            )
        )

        if print_report:
            calibs.append(
                f"\t{str_class_name}: MAE{'W' if use_weights else ''}={calibration_mae * 100:.{calib_report_ndigits}f}%, STD={calibration_std * 100:.{calib_report_ndigits}f}%, COV={calibration_coverage * 100:.0f}%"
            )
            pr_aucs.append(f"{str_class_name}={'N/A' if np.isnan(pr_auc) else f'{pr_auc:.{report_ndigits}f}'}")
            roc_aucs.append(f"{str_class_name}={'N/A' if np.isnan(roc_auc) else f'{roc_auc:.{report_ndigits}f}'}")
            brs.append(f"{str_class_name}={brier_loss * 100:.{report_ndigits}f}%")
            integral_errors.append(f"{str_class_name}={ice:.{report_ndigits}f}")
            if ll is None:
                log_losses.append(f"{str_class_name}=None")
            else:
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
                precision=precision,
                recall=recall,
                f1=f1,
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
        if robust_integral_error is not None:
            print(f"TOTAL ROBUST INTEGRAL ERROR: {robust_integral_error:.4f}")

    if subgroups:
        subgroups_metrics = {"ICE": custom_ice_metric}
        metrics_higher_is_better = {"ICE": False}

        if probs.shape[1] == 2:
            subgroups_metrics["ROC AUC"] = fast_roc_auc
            metrics_higher_is_better["ROC AUC"] = True

        fairness_report = compute_fairness_metrics(
            subgroups=subgroups,
            subset_index=subset_index,
            y_true=targets,
            y_pred=probs,
            metrics=subgroups_metrics,
            metrics_higher_is_better=metrics_higher_is_better,
        )
        if fairness_report is not None:
            if print_report:
                display(fairness_report.style.set_caption("ML perf fairness by group"))
            if metrics is not None:
                metrics.update(dict(fairness_report=fairness_report))

    return preds, probs


def get_model_feature_importances(
    model: Any,
    columns: Sequence[str],
    return_df: bool = False,
) -> Optional[Union[np.ndarray, pd.DataFrame]]:
    """
    Extract feature importances from a trained model.

    Supports models with `feature_importances_` attribute (tree-based models)
    or `coef_` attribute (linear models). For Pipeline objects, extracts
    importances from the final estimator.

    Parameters
    ----------
    model : Any
        Trained model with feature_importances_ or coef_ attribute.
    columns : Sequence[str]
        Feature column names.
    return_df : bool, default=False
        If True, return a DataFrame with feature names and importances.

    Returns
    -------
    np.ndarray, pd.DataFrame, or None
        Feature importances array, DataFrame (if return_df=True), or None
        if the model doesn't have importances.
    """
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
    model: Any,
    columns: Sequence[str],
    model_name: Optional[str] = None,
    num_factors: int = 40,
    figsize: Tuple[int, int] = DEFAULT_FI_FIGSIZE,
    positive_fi_only: bool = False,
    show_plots: bool = True,
    plot_file: str = "",
) -> Optional[np.ndarray]:
    """
    Plot feature importances for a trained model.

    Extracts and visualizes feature importances as a bar chart.

    Parameters
    ----------
    model : Any
        Trained model with extractable feature importances.
    columns : Sequence[str]
        Feature column names.
    model_name : str, optional
        Title for the plot.
    num_factors : int, default=40
        Maximum number of features to display.
    figsize : tuple, default=(15, 10)
        Figure size for the plot.
    positive_fi_only : bool, default=False
        If True, only show features with positive importance.
    plot_file : str, default=""
        Path for saving the plot.

    Returns
    -------
    np.ndarray or None
        Feature importances array, or None if extraction failed.
    """
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
                show_plots=show_plots,
            )
        except (ValueError, AttributeError, IndexError, TypeError) as e:
            logger.warning(f"Could not plot feature importances: {e}. Maybe data shape is changed within a pipeline?")

        return feature_importances


def post_calibrate_model(
    original_model: Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Sequence[str], Any, Dict],
    target_series: pd.Series,
    target_label_encoder: Optional[LabelEncoder],
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    configs: Any,
    calib_set_size: int = 2000,
    nbins: int = DEFAULT_NBINS,
    show_val: bool = False,
    meta_model: Optional[Any] = None,
    **fit_params: Any,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Sequence[str], Any, Dict]:
    """
    Post-calibrate a trained model using a meta-model.

    Trains a meta-model (CatBoost by default) on the original model's
    probability outputs to improve calibration. Uses a portion of the
    test set for calibration training.

    Parameters
    ----------
    original_model : tuple
        8-element tuple containing:
        (model, test_preds, test_probs, val_preds, val_probs, columns, pre_pipeline, metrics)
    target_series : pd.Series
        Target values indexed to match val_idx and test_idx.
    target_label_encoder : LabelEncoder or None
        Encoder for target labels.
    val_idx : np.ndarray
        Indices for validation set.
    test_idx : np.ndarray
        Indices for test set.
    configs : object
        Configuration object with `integral_calibration_error` attribute.
    calib_set_size : int, default=2000
        Number of samples from test set to use for meta-model training.
    nbins : int, default=10
        Number of bins for calibration analysis.
    show_val : bool, default=False
        Whether to display validation set results.
    meta_model : Any, optional
        Custom meta-model. If None, uses CatBoostClassifier with GPU.
    **fit_params
        Additional parameters passed to meta_model.fit().

    Returns
    -------
    tuple
        Same 8-element structure as input, but with calibrated probabilities:
        (model, test_preds, meta_test_probs, val_preds, meta_val_probs, columns, pre_pipeline, metrics)
    """
    from catboost import CatBoostClassifier
    from mlframe.metrics import ICE

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

    # Validate original_model structure
    if not isinstance(original_model, (tuple, list)) or len(original_model) != 8:
        raise ValueError(
            f"original_model must be an 8-element tuple/list containing "
            f"(model, test_preds, test_probs, val_preds, val_probs, columns, pre_pipeline, metrics), "
            f"got {type(original_model).__name__} with {len(original_model) if hasattr(original_model, '__len__') else 'unknown'} elements"
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
        preds=(meta_test_probs[calib_set_size:, 1] > DEFAULT_BINARY_THRESHOLD).astype(int),
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
        preds=(meta_test_probs[:calib_set_size, 1] > DEFAULT_BINARY_THRESHOLD).astype(int),
        probs=meta_test_probs[:calib_set_size, :],
        report_title="",
        nbins=nbins,
        print_report=True,
        show_fi=False,
        custom_ice_metric=configs.integral_calibration_error,
    )

    return model, test_preds, meta_test_probs, val_preds, meta_val_probs, columns, pre_pipeline, metrics


def evaluate_model(
    model: Union[ClassifierMixin, RegressorMixin],
    model_name: str,
    targets: Union[np.ndarray, pd.Series],
    columns: Sequence,
    preds: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    show_fi: bool = True,
    verbose: int = 1,
    **kwargs,
) -> tuple:
    """
    Evaluate a trained model and generate reports.

    Args:
        model: Trained model
        model_name: Name for reporting
        targets: True target values
        columns: Feature column names
        preds: Predictions (optional, will be generated if not provided)
        probs: Probabilities for classification (optional)
        df: DataFrame with features (optional)
        show_fi: Whether to show feature importances
        verbose: Verbosity level
        **kwargs: Additional arguments passed to report functions

    Returns:
        Tuple of (preds, probs) or (preds, None) for regression
    """
    return report_model_perf(
        targets=targets,
        columns=columns,
        model_name=model_name,
        model=model,
        preds=preds,
        probs=probs,
        df=df,
        show_fi=show_fi,
        **kwargs,
    )


__all__ = [
    "evaluate_model",
    "report_model_perf",
    "report_regression_model_perf",
    "report_probabilistic_model_perf",
    "get_model_feature_importances",
    "plot_model_feature_importances",
    "post_calibrate_model",
]
