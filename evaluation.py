# ****************************************************************************************************************************
# Imports
# ****************************************************************************************************************************

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.pythonlib import ensure_installed

ensure_installed("pandas numpy")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import warnings

import pandas as pd, numpy as np
from matplotlib import pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, precision_score, accuracy_score, recall_score, balanced_accuracy_score

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import train_test_split

try:
    from imblearn.pipeline import Pipeline
except:
    from sklearn.pipeline import Pipeline

from IPython.display import display, Markdown, Latex
from finance.backtesting import show_classifier_calibration

from pyutilz.system import tqdmu
from pyutilz.pythonlib import get_human_readable_set_size
from pyutilz.logginglib import initialize_function_log, finalize_function_log, log_loaded_rows, log_activity, log_result

from mlframe.calibration import show_custom_calibration_plot

from catboost import Pool, sum_models


def train_test_split_from_generator(gen: object, X=None, y=None, groups=None):
    for train_indices, test_indices in gen.split(X=X, y=y, groups=groups):
        if groups is not None:
            grouped_train = set(groups[train_indices])
            grouped_test = set(groups[test_indices])
            logger.info(
                f"Train: {len(grouped_train)} Groups {len(train_indices)} Rows, Test :{len(grouped_test)} Groups {len(test_indices)} Rows, overlap={len(grouped_train.intersection(grouped_test))} Groups"
            )

        return train_indices, test_indices


def get_predicted_classes(predictions: np.ndarray, thresholds: np.ndarray = np.array([0.0, 0.1, 0.5, 1.0])):
    """
    Turns scores predicted by regression into class labels, knowing thresholds used to encode labels.
    >>>_,preds=get_predicted_classes(predictions=np.array([0.83157152, 0.91605568, 0.34691267, 0.01739674]),thresholds=np.array([0.0,0.1,0.5,1.0]));preds
    >>>preds
    [3, 3, 2, 0]
    """
    distances = np.abs(thresholds - predictions.reshape(-1, 1))
    distances = np.abs(1 - distances)
    sum_dst = distances.sum(axis=1)
    probs = distances / sum_dst.reshape(-1, 1)
    preds = probs.argmax(axis=1)
    return probs, preds


def regression_stats(y_test, preds, format: str = "_.8f") -> str:
    mes = []
    for func in (mean_absolute_error, mean_squared_error, r2_score):
        res = "{:{fmt}}".format(func(y_test, preds), fmt=format)
        mes += [f"{func.__name__}: {res}"]
    return ", ".join(mes)


def evaluate_estimators(
    X_train,
    X_test,
    y_train=None,
    y_test=None,
    estimators: Sequence = [],
    pre_pipeline: Sequence = [],
    val_size: float = 0.5,
    shuffle: bool = True,
    target_names: dict = None,
    display_labels: list = None,
    show_classification_report: bool = True,
    show_confusion_matrix: bool = True,
    confusion_matrix_file: str = None,
    cfm_normalize: str = "pred",
    cfm_include_values: bool = True,
    cfm_cmap: str = "viridis",
    cfm_ax: object = None,
    cfm_xticks_rotation: str = "horizontal",
    cfm_values_format: str = ".2%",
    cfm_colorbar: bool = True,
    threshold: float = 0.5,
    pos_label: int = 1,
    classification_thresholds: list = None,
    show_calibration_plot: bool = True,
    use_sklearn_calibration: bool = False,
    calibration_nbins: int = 100,
    dpi=100,
    results_log: dict = None,
    target_wrapper: object = None,
    caption: str = None,
    figsize: tuple = (15, 5),
    competing_probs: list = [],
    stratify=None,
    plot: bool = True,
    init_model: object = None,
    groups=None,
    baseline_model=None,
):
    """
    Fit a series of estimators on the same dataset, (and, possibly, same preprocessing pipeline)
    record & compare performances.

    target_wrapper: lambda est: TransformedTargetRegressor(regressor=est,func=np.log1p,inverse_func=np.expm1)
    """

    pipe, classification_report_text, classification_report_dict, cm = None, None, None, None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        if caption:
            display(Markdown(f"**{caption.upper()}:**"))

        for est in estimators:

            # ****************************************************************************************************************************
            # Make complete pipeline
            # ****************************************************************************************************************************

            est_type = type(est).__name__

            if type(est) == tuple:
                est_name, est = est
            else:
                est_name = est_type

            if target_wrapper:
                pipe = Pipeline(pre_pipeline + [("est", target_wrapper(est))])
            else:
                pipe = Pipeline(pre_pipeline + [("est", est)])

            # ****************************************************************************************************************************
            # Fit that estimator to the train set
            # ****************************************************************************************************************************

            if val_size is not None and "CatBoost" in type(est).__name__:

                # ----------------------------------------------------------------------------------------------------------------------------
                # Just a classifier with early stopping... Need to get early stopping set for it...
                # ----------------------------------------------------------------------------------------------------------------------------

                if type(val_size) == float:
                    X_test_test, X_test_val, y_test_test, y_test_val = train_test_split(X_test, y_test, test_size=val_size, shuffle=shuffle, stratify=stratify)
                else:
                    train_indices, test_indices = train_test_split_from_generator(gen=val_size, X=X_test, groups=groups)

                    X_test_test = X_test.iloc[train_indices, :]
                    X_test_val = X_test.iloc[test_indices, :]
                    y_test_test = y_test[train_indices]
                    y_test_val = y_test[test_indices]

                if baseline_model is not None:
                    eval_set = Pool(X_test_val, y_test_val)
                    eval_set.set_baseline(baseline_model.predict(X_test_val).astype(int))
                else:
                    eval_set = (X_test_val, y_test_val)

                if type(X_train) in (Pool, str):
                    pipe.fit(X_train, est__eval_set=eval_set, est__plot=plot, est__init_model=init_model)
                else:
                    pipe.fit(X_train, y_train, est__eval_set=eval_set, est__plot=plot, est__init_model=init_model)

            else:
                if type(X_train) in (Pool, str):
                    pipe.fit(X_train)
                else:
                    pipe.fit(X_train, y_train)
                X_test_test = X_test
                y_test_test = y_test

            # ****************************************************************************************************************************
            # Get predictions for the test set
            # ****************************************************************************************************************************

            test_size = len(y_test_test)

            if test_size > 0:
                print("test_size=", test_size)
                test_size = get_human_readable_set_size(test_size)
                is_classification = is_classifier(est)

                if is_classification:

                    # ----------------------------------------------------------------------------------------------------------------------------
                    # Classifier
                    # ----------------------------------------------------------------------------------------------------------------------------

                    probs = pipe.predict_proba(X_test_test)
                    nclasses = probs.shape[1]

                    if nclasses == 2:
                        if threshold is None:
                            threshold = 1 / nclasses
                        preds = (probs > threshold).astype(np.int8)[:, pos_label]
                    else:
                        preds = np.argmax(probs, axis=1)

                    mes = f"Balanced accuracy on {test_size} samples: {balanced_accuracy_score(y_test_test,preds):.2%}"
                    if classification_thresholds is not None:
                        mes += "\n" + regression_stats(pd.Series(y_test_test).map(classification_thresholds), pd.Series(preds).map(classification_thresholds))

                else:

                    # ----------------------------------------------------------------------------------------------------------------------------
                    # Regressor
                    # ----------------------------------------------------------------------------------------------------------------------------

                    preds = pipe.predict(X_test_test)

                    mes = regression_stats(y_test_test, preds)

                    if classification_thresholds is not None:
                        _, y_test_test = get_predicted_classes(y_test_test.values, thresholds=classification_thresholds)
                        probs, preds = get_predicted_classes(preds, thresholds=classification_thresholds)
                        nclasses = probs.shape[1]

                # ----------------------------------------------------------------------------------------------------------------------------
                # Report accuracy & other metrics
                # ----------------------------------------------------------------------------------------------------------------------------

                if mes:
                    # print(mes)
                    # logger.info(mes)
                    display(Markdown(f"*Model*: **{est_name}**, {mes}"))

                # ----------------------------------------------------------------------------------------------------------------------------
                # Compute additional metrics & visualisations
                # ----------------------------------------------------------------------------------------------------------------------------

                if is_classification or (classification_thresholds is not None):

                    if "classes_" in dir(est):
                        labels_to_use = est.classes_
                        target_names_to_use = {key: value for key, value in target_names.items() if value in labels_to_use}
                    else:
                        labels_to_use = None
                        target_names_to_use = target_names

                    if show_classification_report:

                        classification_report_text = classification_report(y_test_test, preds, labels=labels_to_use, target_names=target_names_to_use)
                        print(classification_report_text)

                        if results_log:
                            classification_report_dict = classification_report(
                                y_test_test, preds, labels=labels_to_use, target_names=target_names_to_use, output_dict=True
                            )

                            log_result(results_log, f"classification_report_dict", classification_report_dict)
                            log_result(results_log, f"classification_report_text", classification_report_text)

                    if show_confusion_matrix:

                        # plot_confusion_matrix(pipe, X_test_test, y_test_test, display_labels=display_labels,normalize ='pred',values_format='.2%');

                        cm = confusion_matrix(y_test_test, preds, labels=labels_to_use, normalize=cfm_normalize)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
                        disp.plot(
                            include_values=cfm_include_values,
                            cmap=cfm_cmap,
                            ax=cfm_ax,
                            xticks_rotation=cfm_xticks_rotation,
                            values_format=cfm_values_format,
                            colorbar=cfm_colorbar,
                        )

                        # plt.rcParams['axes.grid'] = False
                        plt.grid(b=None)
                        if confusion_matrix_file:
                            plt.savefig(confusion_matrix_file, dpi=dpi, bbox_inches="tight")

                        plt.show()

                    if show_calibration_plot:

                        if use_sklearn_calibration:

                            """Standard sklearn code"""

                            for pos_label in range(nclasses):
                                prob_pos = probs[:, pos_label]
                                prob_true, prob_pred = calibration_curve(y_test_test == pos_label, prob_pos, n_bins=calibration_nbins)

                                plt.figure(figsize=figsize)
                                ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
                                ax2 = plt.subplot2grid((3, 1), (2, 0))

                                ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

                                ax1.plot(prob_pred, prob_true, "s-", label="%s" % (est_name,))

                                ax2.hist(prob_pos, range=(0, 1), bins=10, label=est_name, histtype="step", lw=2)

                                ax1.set_ylabel("Fraction of positives")
                                ax1.set_ylim([-0.05, 1.05])
                                ax1.legend(loc="lower right")
                                ax1.set_title(f"Calibration plot for {display_labels[pos_label]}")

                                ax2.set_xlabel("Mean predicted value")
                                ax2.set_ylabel("Count")
                                ax2.legend(loc="upper center", ncol=2)

                                plt.tight_layout()
                                plt.show()
                        else:
                            show_custom_calibration_plot(
                                X=X_test_test,
                                y=y_test_test,
                                probs=probs,
                                competing_probs=competing_probs,
                                nclasses=nclasses,
                                nbins=calibration_nbins,
                                display_labels=display_labels,
                                figsize=figsize,
                            )

    return pipe, classification_report_text, classification_report_dict, cm


def evaluate_grouped(
    pipe: object, X_test: object, y_test: object, by_column: str = "Вакансия_Должность", ntop: int = 100, min_population=100, all_cats: dict = {}
):
    """
    Показать точность обученной простой модели в разбивке по Должностям
    """
    from sklearn.metrics import classification_report

    target_names = {k: v for k, v in sorted(all_cats.items(), key=lambda item: item[1])}

    res = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        for position, qty in tqdmu(X_test[by_column].value_counts().head(ntop).to_dict().items()):
            idx = X_test[by_column] == position

            preds = pipe.predict(X_test[idx])
            rp = classification_report(y_test[idx], preds, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=True, zero_division="warn")

            failed = False
            for label, stats in rp.items():
                if type(stats) == dict:
                    if stats.get("support", 0) < min_population:
                        failed = True
                        break
            if not failed:
                stats = rp["weighted avg"]
                if stats:
                    res.append({by_column: position, "Откликов": qty, "Точность": stats["precision"], "Полнота": stats["recall"]})
                # res.append({'Должность':position,'Откликов':qty,'Точность':precision_score(y_test[idx], preds, average='macro'),'Полнота':recall_score(y_test[idx], preds, average='macro')})

    by_position = pd.DataFrame(res)

    return by_position.sort_values(by="Точность", ascending=False).reset_index(drop=True)
