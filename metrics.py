# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *
from numba import njit
from math import floor
import numpy as np, pandas as pd
from matplotlib import pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """np.argsort needs to stay out of njitted func."""
    desc_score_indices = np.argsort(y_score)[::-1]
    return fast_numba_auc_nonw(y_true=y_true, y_score=y_score, desc_score_indices=desc_score_indices)


@njit()
def fast_numba_auc_nonw(y_true: np.ndarray, y_score: np.ndarray, desc_score_indices: np.ndarray) -> float:
    """code taken from fastauc lib."""
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    last_counted_fps = 0
    last_counted_tps = 0
    tps, fps = 0, 0
    auc = 0

    l = len(y_true) - 1
    for i in range(l + 1):
        tps += y_true[i]
        fps += 1 - y_true[i]
        if i == l or y_score[i + 1] != y_score[i]:
            auc += (fps - last_counted_fps) * (last_counted_tps + tps)
            last_counted_fps = fps
            last_counted_tps = tps
    tmp=tps * fps* 2
    if tmp>0:
        return auc / tmp
    else:
        return 0


@njit()
def fast_precision(y_true: np.ndarray, y_pred: np.ndarray, nclasses: int = 2, zero_division: int = 0):
    # storage inits
    allpreds = np.zeros(nclasses, dtype=np.int64)
    hits = np.zeros(nclasses, dtype=np.int64)
    # count stats
    for true_class, predicted_class in zip(y_true, y_pred):
        allpreds[predicted_class] += 1
        if predicted_class == true_class:
            hits[predicted_class] += 1
    precisions = hits / allpreds
    return precisions[-1]


@njit()
def fast_classification_report(y_true: np.ndarray, y_pred: np.ndarray, nclasses: int = 2, zero_division: int = 0):
    """Custom classification report, proof of concept."""

    N_AVG_ARRAYS = 3  # precisions, recalls, f1s

    # storage inits
    weighted_averages = np.empty(N_AVG_ARRAYS, dtype=np.float64)
    macro_averages = np.empty(N_AVG_ARRAYS, dtype=np.float64)
    supports = np.zeros(nclasses, dtype=np.int64)
    allpreds = np.zeros(nclasses, dtype=np.int64)
    misses = np.zeros(nclasses, dtype=np.int64)
    hits = np.zeros(nclasses, dtype=np.int64)

    # count stats
    for true_class, predicted_class in zip(y_true, y_pred):
        supports[true_class] += 1
        allpreds[predicted_class] += 1
        if predicted_class == true_class:
            hits[predicted_class] += 1
        else:
            misses[predicted_class] += 1

    # main calcs
    accuracy = hits.sum() / len(y_true)
    balanced_accuracy = np.nan_to_num(hits / supports, copy=True, nan=zero_division).mean()

    recalls = hits / supports
    precisions = hits / allpreds
    f1s = 2 * (precisions * recalls) / (precisions + recalls)

    # fix nans & compute averages
    i = 0
    for arr in (precisions, recalls, f1s):
        np.nan_to_num(arr, copy=False, nan=zero_division)
        weighted_averages[i] = (arr * supports).sum() / len(y_true)
        macro_averages[i] = arr.mean()
        i += 1

    return hits, misses, accuracy, balanced_accuracy, supports, precisions, recalls, f1s, macro_averages, weighted_averages


@njit()
def fast_calibration_binning(y_true: np.ndarray, y_pred: np.ndarray, nbins: int = 100):
    """Computes bins of predicted vs actual events frequencies. Corresponds to sklearn's UNIFORM strategy."""

    pockets_predicted = np.zeros(nbins, dtype=np.int64)
    pockets_true = np.zeros(nbins, dtype=np.int64)

    # compute span

    min_val, max_val = 1.0, 0.0
    for predicted_prob in y_pred:
        if predicted_prob > max_val:
            max_val = predicted_prob
        elif predicted_prob < min_val:
            min_val = predicted_prob
    span = max_val - min_val

    if span>0:
        multiplier = nbins / span
        for true_class, predicted_prob in zip(y_true, y_pred):
            ind = floor((predicted_prob - min_val) * multiplier)
            pockets_predicted[ind] += 1
            pockets_true[ind] += true_class   
    else:
        ind =0
        for true_class, predicted_prob in zip(y_true, y_pred):
            pockets_predicted[ind] += 1
            pockets_true[ind] += true_class

    idx = np.nonzero(pockets_predicted > 0)[0]

    hits = pockets_predicted[idx]
    if len(hits) > 0:
        freqs_predicted, freqs_true = (min_val + (np.arange(nbins)[idx] + 0.5) * span / nbins).astype(np.float64), pockets_true[idx] / pockets_predicted[idx]
    else:
        freqs_predicted, freqs_true = np.array((), dtype=np.float64), np.array((), dtype=np.float64)

    return freqs_predicted, freqs_true, hits


def show_calibration_plot(
    freqs_predicted: np.ndarray,
    freqs_true: np.ndarray,
    hits: np.ndarray,
    show_plots: bool = True,
    plot_file: str = "",
    plot_title: str = "",
    figsize: tuple = (12, 6),
):
    """Plots reliability digaram from the binned predictions."""
    fig = plt.figure(figsize=figsize)
    plt.scatter(freqs_predicted, freqs_true, marker="o", s=5000 * hits / hits.sum(), c=hits, label="Real")
    x_min, x_max = np.min(freqs_predicted), np.max(freqs_predicted)
    plt.plot([x_min, x_max], [x_min, x_max], "g--", label="Perfect")
    if plot_title:
        plt.title(plot_title)
    if plot_file:
        fig.savefig(plot_file)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

njit()
def maximum_absolute_percentage_error(y_true:np.ndarray, y_pred:np.ndarray)->float:    
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    return np.nanmax(mape)

@njit()
def calibration_metrics_from_freqs(freqs_predicted: np.ndarray, freqs_true: np.ndarray, hits: np.ndarray, nbins: int,array_size:int):
    calibration_coverage=len(hits)/nbins
    if len(hits)>0:
        diffs = np.abs((freqs_predicted - freqs_true))   
        weights=hits/array_size
        calibration_mae =np.sum(diffs*weights)
        calibration_std=np.sqrt(np.sum(((diffs-calibration_mae)**2)*weights))
    else:
        calibration_mae, calibration_std=1.0,1.0
    
    return calibration_mae, calibration_std,calibration_coverage


@njit()
def fast_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray, nbins: int = 100):
    freqs_predicted, freqs_true, hits = fast_calibration_binning(y_true=y_true, y_pred=y_pred, nbins=nbins)
    return calibration_metrics_from_freqs(freqs_predicted=freqs_predicted, freqs_true=freqs_true, hits=hits, nbins=nbins,array_size=len(y_true))


def fast_calibration_report(y_true: np.ndarray, y_pred: np.ndarray, nbins: int = 100, show_plots: bool = True, plot_file: str = "", figsize: tuple = (12, 6),ndigits:int=4):
    """Bins predictions, then computes regresison-like error metrics between desired and real binned probs."""

    freqs_predicted, freqs_true, hits = fast_calibration_binning(y_true=y_true, y_pred=y_pred, nbins=nbins)
    calibration_mae, calibration_std,calibration_coverage = calibration_metrics_from_freqs(freqs_predicted=freqs_predicted, freqs_true=freqs_true, hits=hits, nbins=nbins,array_size=len(y_true))

    if plot_file or show_plots:
        show_calibration_plot(
            freqs_predicted=freqs_predicted,
            freqs_true=freqs_true,
            hits=hits,
            plot_title=f"Calibration MAE={calibration_mae:.{ndigits}f} ± {calibration_std:.{ndigits}f}, cov. {calibration_coverage*100:.{ndigits}f}%",
            show_plots=show_plots,
            plot_file=plot_file,
            figsize=figsize,
        )

    return calibration_mae, calibration_std, calibration_coverage


def predictions_time_instability(preds: pd.Series) -> float:
    """Computes how stable are true values or predictions over time.
    It's hard to use predictions that change upside down from point to point.
    For binary classification instability ranges from 0 to 1, for regression from 0 to any value depending on the target stats.
    """
    return np.abs(np.diff(preds)).mean()


# ----------------------------------------------------------------------------------------------------------------------------
# Errors & scorers
# ----------------------------------------------------------------------------------------------------------------------------


class CB_CALIB_ERROR:
    def is_max_optimal(self):
        return False  # greater is better

    def evaluate(self, approxes, target, weight):
        output_weight = 1  # weight is not used

        # predictions=expit(approxes[0])
        y_pred = 1 / (1 + np.exp(-approxes[0]))
        calibration_mae, calibration_std, calibration_coverage = fast_calibration_metrics(y_true=target, y_pred=y_pred)

        return njitted_calib_error(calibration_mae=calibration_mae, calibration_std=calibration_std, calibration_coverage=calibration_coverage), output_weight

    def get_final_error(self, error, weight):
        return error


class CB_PRECISION:
    def is_max_optimal(self):
        return False  # greater is better

    def evaluate(self, approxes, target, weight):
        output_weight = 1  # weight is not used

        # y_pred=expit(approxes[0])
        y_pred = 1 / (1 + np.exp(-approxes[0]))

        return fast_precision(y_true=target, y_pred=(y_pred >= 0.5).astype(np.int8), zero_division=0), output_weight

    def get_final_error(self, error, weight):
        return error

#@njit()
def calib_error(calibration_mae:float , calibration_std:float, calibration_coverage:float,std_weight:float=0.5,cov_degree:float=0.5) -> float:
    """Integral calibration error."""

    if calibration_coverage==0.0:
        return 1e5
    else:
        return (calibration_mae + calibration_std * std_weight)/(calibration_coverage**cov_degree)
njitted_calib_error=njit(calib_error)

def calib_error_xgboost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    calibration_mae, calibration_std, calibration_coverage = fast_calibration_metrics(y_true=y_true, y_pred=y_pred)
    return calib_error(calibration_mae=calibration_mae, calibration_std=calibration_std, calibration_coverage=calibration_coverage)


def calib_error_keras(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    calibration_mae, calibration_std, calibration_coverage = fast_calibration_metrics(y_true=y_true.numpy()[:, -1], y_pred=y_pred.numpy()[:, -1])
    return calib_error(calibration_mae=calibration_mae, calibration_std=calibration_std, calibration_coverage=calibration_coverage)

@njit()
def brier_score_loss(y_true:np.ndarray,y_prob:np.ndarray)->float:
    return np.mean((y_true - y_prob) ** 2)

@njit()
def probability_separation_score(y_true: np.ndarray, y_prob: np.ndarray, class_label: int = 1, std_weight: float = 0.5) -> float:
    idx = y_true == class_label
    if idx.sum()==0:
        return np.nan
        if class_label == 1:
            res=0.0
        else:
            res=1.0
    else:
        res = np.mean(y_prob[idx])
        if std_weight != 0.0:
            addend = np.std(y_prob[idx]) * std_weight
            if class_label == 1:
                res = res - addend
            else:
                res = res + addend
    return res
