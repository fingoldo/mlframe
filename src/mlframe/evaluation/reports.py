
from __future__ import annotations

# ****************************************************************************************************************************
# Imports
# ****************************************************************************************************************************

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Optional, Sequence

import warnings

import pandas as pd, numpy as np
from matplotlib import pyplot as plt

from sklearn.calibration import calibration_curve
from mlframe.reporting.charts import confusion_matrix_counts, plot_confusion_matrix
from mlframe.metrics.core import (
    fast_mean_absolute_error,
    fast_mean_squared_error,
    fast_r2_score,
    fast_classification_report,
    format_classification_report,
    balanced_accuracy_binary,
)

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.base import is_classifier
from sklearn.model_selection import train_test_split

try:
    from imblearn.pipeline import Pipeline
except (ImportError, ModuleNotFoundError):
    from sklearn.pipeline import Pipeline

from IPython.display import display, Markdown

# `finance` is a sibling private library (not on PyPI). The
# show_classifier_calibration helper is only used by a single optional
# reporting path; guard the import so consumers without finance installed
# can still load mlframe.evaluation.reports for the rest of its surface.
try:
    from finance.backtesting import show_classifier_calibration  # type: ignore
except ImportError:  # pragma: no cover - depends on private sibling pkg
    def show_classifier_calibration(*args, **kwargs):  # type: ignore[no-redef]
        raise ImportError(
            "show_classifier_calibration() requires the private `finance` package "
            "(github.com/fingoldo/finance); install it from source before calling."
        )

from pyutilz.system import tqdmu
from pyutilz.pythonlib import get_human_readable_set_size
from pyutilz.logginglib import log_result

from mlframe.calibration.quality import make_custom_calibration_plot

from catboost import Pool


def train_test_split_from_generator(gen: object, X=None, y=None, groups=None):
    for train_indices, test_indices in gen.split(X=X, y=y, groups=groups):
        if groups is not None:
            grouped_train = set(groups[train_indices])
            grouped_test = set(groups[test_indices])
            logger.info(
                "Train: %s Groups %s Rows, Test :%s Groups %s Rows, overlap=%s Groups",
                len(grouped_train),
                len(train_indices),
                len(grouped_test),
                len(test_indices),
                len(grouped_train.intersection(grouped_test)),
            )

        return train_indices, test_indices


def get_predicted_classes(predictions: np.ndarray, thresholds: np.ndarray = None):
    """
    Turns scores predicted by regression into class labels, knowing thresholds used to encode labels.
    >>>_,preds=get_predicted_classes(predictions=np.array([0.83157152, 0.91605568, 0.34691267, 0.01739674]),thresholds=np.array([0.0,0.1,0.5,1.0]));preds
    >>>preds
    [3, 3, 2, 0]
    """
    if thresholds is None:
        thresholds = np.array([0.0, 0.1, 0.5, 1.0])
    distances = np.abs(thresholds - predictions.reshape(-1, 1))
    distances = np.abs(1 - distances)
    sum_dst = distances.sum(axis=1)
    probs = distances / sum_dst.reshape(-1, 1)
    preds = probs.argmax(axis=1)
    return probs, preds


def regression_stats(y_test, preds, format: str = "_.8f") -> str:
    mes = []
    for func in (fast_mean_absolute_error, fast_mean_squared_error, fast_r2_score):
        res = "{:{fmt}}".format(func(y_test, preds), fmt=format)
        mes += [f"{func.__name__}: {res}"]
    return ", ".join(mes)


def evaluate_estimators(
    X_train,
    X_test,
    y_train=None,
    y_test=None,
    estimators: Sequence = None,
    pre_pipeline: Sequence = None,
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
    competing_probs: list = None,
    stratify=None,
    plot: bool = True,
    init_model: object = None,
    groups=None,
    baseline_model=None,
):
    """
    Fit a series of estimators to the same dataset, (and, possibly, same preprocessing pipeline)
    record & compare performances.

    target_wrapper: lambda est: TransformedTargetRegressor(regressor=est,func=np.log1p,inverse_func=np.expm1)
    """
    if competing_probs is None:
        competing_probs = []
    if estimators is None:
        estimators = []
    if pre_pipeline is None:
        pre_pipeline = []

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

            if type(est) is tuple:
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

            if val_size is not None and (("CatBoost" in type(est).__name__) or ('TransformedTargetRegressor' in type(est).__name__ and ("CatBoost" in type(est.regressor).__name__))):

                # ----------------------------------------------------------------------------------------------------------------------------
                # Just a classifier with early stopping... Need to get early stopping set for it...
                # ----------------------------------------------------------------------------------------------------------------------------

                if type(val_size) is float:
                    X_test_test, X_test_val, y_test_test, y_test_val = train_test_split(X_test, y_test, test_size=val_size, shuffle=shuffle, stratify=stratify)
                else:
                    train_indices, test_indices = train_test_split_from_generator(gen=val_size, X=X_test, groups=groups)

                    X_test_test = X_test.iloc[train_indices, :]
                    X_test_val = X_test.iloc[test_indices, :]
                    y_test_test = y_test.iloc[train_indices] if hasattr(y_test, "iloc") else y_test[train_indices]
                    y_test_val = y_test.iloc[test_indices] if hasattr(y_test, "iloc") else y_test[test_indices]

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
                logger.info("test_size=%s", test_size)
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
                        # Previous: `(probs > threshold).astype(int8)[:, pos_label]` returned a
                        # 0/1 vector that was "1 iff prob of pos_label > threshold" — which is a
                        # per-column thresholding semantic distinct from argmax. When pos_label=1
                        # that collapses to the intended behavior; for pos_label=0 it silently
                        # inverted the threshold's meaning. Use an explicit column threshold:
                        preds = (probs[:, pos_label] > threshold).astype(np.int8)
                    else:
                        # Wave 21 P2: nan-safe argmax.
                        from ..utils.nan_safe import argmax_classes_safe
                        preds = argmax_classes_safe(probs, context="evaluation.reports")

                    if nclasses == 2:
                        _bal_acc = balanced_accuracy_binary(y_test_test, preds)
                    else:
                        # balanced_accuracy_binary is binary-only; multiclass recall-macro has no fast kernel.
                        from sklearn.metrics import balanced_accuracy_score
                        _bal_acc = balanced_accuracy_score(y_test_test, preds)
                    mes = f"Balanced accuracy on {test_size} samples: {_bal_acc:.2%}"
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

                    # Ordered target-name list for the fast report/plot (their kernels index by
                    # class 0..nclasses-1). target_names_to_use is a {label: display-name} dict.
                    if isinstance(target_names_to_use, dict):
                        _tn = [target_names_to_use.get(i, str(i)) for i in range(nclasses)]
                    elif target_names_to_use is not None:
                        _tn = list(target_names_to_use)
                    else:
                        _tn = None

                    if show_classification_report:

                        classification_report_text = format_classification_report(
                            y_test_test, preds, nclasses=nclasses, target_names=_tn
                        )
                        logger.info("classification report:\n%s", classification_report_text)

                        if results_log:
                            hits, misses, accuracy, balanced_accuracy, supports, precisions, recalls, f1s, macro_avgs, weighted_avgs = (
                                fast_classification_report(y_test_test, preds, nclasses=nclasses)
                            )
                            classification_report_dict = {
                                str((_tn[i] if _tn else i)): {
                                    "precision": float(precisions[i]),
                                    "recall": float(recalls[i]),
                                    "f1-score": float(f1s[i]),
                                    "support": int(supports[i]),
                                }
                                for i in range(nclasses)
                            }
                            classification_report_dict["accuracy"] = float(accuracy)

                            log_result(results_log, "classification_report_dict", classification_report_dict)
                            log_result(results_log, "classification_report_text", classification_report_text)

                    if show_confusion_matrix:

                        cm, _cm_labels = confusion_matrix_counts(y_test_test, preds, labels=labels_to_use)
                        fig_cm, ax_cm = plot_confusion_matrix(
                            y_test_test,
                            preds,
                            labels=labels_to_use,
                            display_labels=display_labels,
                            normalize=cfm_normalize,
                            cmap=cfm_cmap,
                            ax=cfm_ax,
                            values_format=cfm_values_format,
                            colorbar=cfm_colorbar,
                            include_values=cfm_include_values,
                            xticks_rotation=cfm_xticks_rotation,
                        )

                        ax_cm.grid(visible=None)
                        if confusion_matrix_file:
                            fig_cm.savefig(confusion_matrix_file, dpi=dpi, bbox_inches="tight")

                        # Library code must not leave figures open (memory leak under repeated evaluation) nor block on plt.show().
                        plt.close(fig_cm or plt.gcf())

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
                                # Close instead of show: library code must not leak per-class figures nor block.
                                plt.close()
                        else:
                            fig, _cal_metrics = make_custom_calibration_plot(
                                y=y_test_test,
                                probs=probs,
                                nclasses=nclasses,
                                nbins=calibration_nbins,
                                display_labels=display_labels,
                                figsize=figsize,
                                competing_probs=competing_probs,
                                X=X_test_test,
                            )
                            if fig is not None:
                                # Close instead of show: library code must not leak the calibration figure nor block.
                                plt.close(fig)

    return pipe, classification_report_text, classification_report_dict, cm


def evaluate_grouped(
    pipe: object, X_test: object, y_test: object, by_column: str = "Вакансия_Должность", ntop: int = 100, min_population=100, all_cats: dict = None
):
    """
    Показать точность обученной простой модели в разбивке по Должностям
    """
    if all_cats is None:
        all_cats = {}

    def _report_dict(y_true, y_pred) -> dict:
        """output_dict-style report (per-label + 'weighted avg') via our fast kernel.

        Remaps the observed labels to 0..K-1 (fast_classification_report indexes by
        contiguous class id), computes per-class precision/recall/f1/support, and
        assembles the sklearn-shaped dict the callers below read ('support', 'precision',
        'recall' per label plus the 'weighted avg' aggregate)."""
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        labels_arr = np.unique(np.concatenate([yt.ravel(), yp.ravel()])) if yt.size or yp.size else np.array([])
        K = len(labels_arr)
        if K == 0:
            return {}
        pos = {lab: i for i, lab in enumerate(labels_arr)}
        yt_pos = np.array([pos[v] for v in yt.ravel()], dtype=np.int64)
        yp_pos = np.array([pos[v] for v in yp.ravel()], dtype=np.int64)
        (hits, misses, accuracy, balanced_accuracy, supports, precisions,
         recalls, f1s, macro_avgs, weighted_avgs) = fast_classification_report(yt_pos, yp_pos, nclasses=K)
        out: dict = {}
        for i, lab in enumerate(labels_arr):
            out[str(lab)] = {
                "precision": float(precisions[i]),
                "recall": float(recalls[i]),
                "f1-score": float(f1s[i]),
                "support": int(supports[i]),
            }
        out["accuracy"] = float(accuracy)
        out["weighted avg"] = {
            "precision": float(weighted_avgs[0]),
            "recall": float(weighted_avgs[1]),
            "f1-score": float(weighted_avgs[2]),
            "support": int(supports.sum()),
        }
        return out

    res = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        for position, qty in tqdmu(X_test[by_column].value_counts().head(ntop).to_dict().items()):
            idx = X_test[by_column] == position

            preds = pipe.predict(X_test[idx])
            rp = _report_dict(y_test[idx], preds)

            failed = False
            for label, stats in rp.items():
                if type(stats) is dict:
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


# ****************************************************************************************************************************
# Salvaged legacy evaluation helpers (copied from training_old.py / OldEnsembling.py)
# ****************************************************************************************************************************


def predictions_beautify_linear(preds: np.ndarray, known_outcomes: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Adjust probabilities linearly toward ground-truth labels by mixing weight alpha.

    preds: 1D array of floats in [0,1]
    known_outcomes: 1D array of ints {0,1}
    alpha: mixing weight toward truth (0=no change, 1=fully corrected)

    Business use case: visualise the business-lift effect of a more accurate
    forecast without actually improving the model — useful for sensitivity /
    counterfactual analysis on thresholded decisions.

    Returns the adjusted probabilities as a numpy float array.
    """
    preds = np.asarray(preds, dtype=float)
    y = np.asarray(known_outcomes, dtype=float)
    return (1 - alpha) * preds + alpha * y


def _precision_at_top_decile(y_true: np.ndarray, preds: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    preds = np.asarray(preds, dtype=float)
    n = len(preds)
    if n == 0:
        return 0.0
    k = max(1, n // 10)
    # Wave 58 (2026-05-20): lexsort with row-position tiebreak so tied
    # predictions don't make the top-decile pick depend on input row order.
    idx = np.lexsort((np.arange(n), -preds))[:k]
    return float(np.mean(y_true[idx]))


def plot_beautified_lift(
    preds: np.ndarray,
    y: np.ndarray,
    alphas: Sequence[float] = (0.0, 0.01, 0.05, 0.1, 0.2),
    metric: str = "precision_at_top_decile",
    ax: Optional[object] = None,
):
    """Plot metric(preds_beautified, y) as a function of alpha mixing weight.

    Supported metrics: 'auroc', 'auprc', 'precision_at_top_decile', 'brier'.
    Returns the matplotlib Figure (does not call plt.show()).
    """
    from mlframe.metrics.core import fast_roc_auc, fast_brier_score_loss, average_precision_score

    y = np.asarray(y)
    preds = np.asarray(preds, dtype=float)

    if metric == "auroc":
        _score = fast_roc_auc
    elif metric == "auprc":
        _score = average_precision_score
    elif metric == "brier":
        _score = lambda y_, p_: fast_brier_score_loss(y_, p_)
    elif metric == "precision_at_top_decile":
        _score = _precision_at_top_decile
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    xs = list(alphas)
    ys = [_score(y, predictions_beautify_linear(preds, y, alpha=a)) for a in xs]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("alpha (mixing weight toward truth)")
    ax.set_ylabel(metric)
    ax.set_title(f"Beautified-lift: {metric} vs alpha")
    return fig


def _decimate_curve_vertices(arrays: tuple, max_points: int = 2000) -> tuple:
    """Uniform-stride decimation of parallel curve-vertex arrays for plotting; both endpoints are always kept.
    Display-only: metrics (AP / AUC / Brier) stay computed on the full data."""
    n = len(arrays[0])
    if n <= max_points:
        return arrays
    idx = np.linspace(0, n - 1, max_points).round().astype(np.int64)
    return tuple(a[idx] for a in arrays)


def plot_pr_curve(
    y: np.ndarray,
    preds: np.ndarray,
    show_calibration: bool = False,
    save_as: Optional[str] = None,
    thresh: float = 0.5,
    ax: Optional[object] = None,
):
    """Dual PR + ROC on same axes with PR baseline and classification_report logged at thresh.

    Standalone axes-based shim (returns a Figure; no production suite caller). The canonical
    binary curve charts in the suite are the FigureSpec panels in reporting/charts/binary.py
    (ROC / PR / SCORE_DIST / KS / THRESHOLD / GAIN); this helper stays for direct notebook use.
    Curve vertices are decimated to ~2000 points for plotting; all displayed metrics use full data.
    """
    from sklearn.metrics import precision_recall_curve, auc
    from mlframe.metrics.core import fast_brier_score_loss, average_precision_score, fast_roc_curve

    y = np.asarray(y)
    preds = np.asarray(preds, dtype=float)

    # A single-class y makes PR/ROC degenerate (prevalence 0 or 1, undefined AUC) and the baseline misleading. Refuse explicitly.
    if np.unique(y).size < 2:
        raise ValueError(f"plot_pr_curve: y must contain both classes; got a single class {np.unique(y).tolist()}.")

    precision, recall, _ = precision_recall_curve(y, preds)
    recall, precision = _decimate_curve_vertices((recall, precision))
    # The dummy (constant-prediction) PR baseline is analytic: a single threshold gives precision [prevalence, 1], recall [1, 0],
    # and AP = positive-class prevalence. Running precision_recall_curve on np.full(n, mode) wasted a full-n pass for 2 points.
    classes, counts = np.unique(y, return_counts=True)
    prevalence = float(counts[-1]) / float(counts.sum())
    dummy_precision = np.array([prevalence, 1.0])
    dummy_recall = np.array([1.0, 0.0])
    pr_auc = average_precision_score(y, preds)
    dummy_pr_auc = prevalence

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.set_title("PRC/ROC, BrierLoss=%.3f" % fast_brier_score_loss(y, preds))
    ax.step(recall, precision, color="b", alpha=0.4, label="PR AUC=%.2f/%.2fR" % (pr_auc, dummy_pr_auc), where="post")
    ax.fill_between(recall, precision, alpha=0.2, color="b", step="post")
    ax.step(dummy_recall, dummy_precision, color="r", alpha=0.1, where="post")
    ax.fill_between(dummy_recall, dummy_precision, alpha=0.1, color="r", step="post")
    ax.set_xlabel("Recall/Fall-out")
    ax.set_ylabel("Precision/Recall")
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])

    fpr, tpr, _ = fast_roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    fpr_d, tpr_d = _decimate_curve_vertices((fpr, tpr))
    ax.plot(fpr_d, tpr_d, "b", label="ROC AUC=%0.3f" % roc_auc)
    ax.plot([0, 1], [0, 1], "r--")
    ax.legend(loc="lower right")

    if show_calibration:
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(y, preds, n_bins=50)
            ax.plot(mean_predicted_value, fraction_of_positives, "--g")
        except Exception as exc:
            logger.warning("calibration overlay failed: %s", exc)

    if save_as:
        fig.savefig(save_as, bbox_inches="tight")

    try:
        logger.info("classification report at thresh=%s:\n%s", thresh, format_classification_report(y, (preds > thresh).astype(np.int64), nclasses=2))
    except Exception as exc:
        logger.warning("classification_report failed: %s", exc)

    return fig


def plot_roc_curve(
    y: np.ndarray,
    preds: np.ndarray,
    show_calibration: bool = False,
    save_as: Optional[str] = None,
    ax: Optional[object] = None,
):
    """Simple ROC + diagonal, calibration overlay optional. Returns Figure.

    Source: OldEnsembling.plot_roc (preferred over the Models.py duplicate
    because it includes the calibration overlay).
    """
    from sklearn.metrics import auc
    from mlframe.metrics.core import fast_roc_curve

    y = np.asarray(y)
    preds = np.asarray(preds, dtype=float)

    fpr, tpr, _ = fast_roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    fpr_d, tpr_d = _decimate_curve_vertices((fpr, tpr))

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.set_title("ROC")
    ax.plot(fpr_d, tpr_d, "b", label="AUC = %0.2f" % roc_auc)
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_ylabel("Recall")
    ax.set_xlabel("Fall-out")
    ax.legend(loc="lower right")

    if show_calibration:
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(y, preds, n_bins=50)
            ax.plot(mean_predicted_value, fraction_of_positives)
        except Exception as exc:
            logger.warning("calibration overlay failed: %s", exc)

    if save_as:
        fig.savefig(save_as, bbox_inches="tight")

    return fig
