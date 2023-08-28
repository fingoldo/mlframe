import numpy as np, pandas as pd
from numba import njit
from math import floor
from matplotlib import pyplot as plt


def fast_auc(y_true: np.array, y_score: np.array) -> float:
    """np.argsort needs to stay out of njitted func."""
    desc_score_indices = np.argsort(y_score)[::-1]
    return fast_numba_auc_nonw(y_true=y_true, y_score=y_score, desc_score_indices=desc_score_indices)


@njit()
def fast_numba_auc_nonw(y_true: np.array, y_score: np.array, desc_score_indices: np.array) -> float:
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
    return auc / (tps * fps * 2)

@njit()
def fast_precision(y_true: np.ndarray, y_pred: np.ndarray, nclasses: int = 2, zero_division: int = 0):
    # storage inits
    misses = np.zeros(nclasses, dtype=np.int64)
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

    min_val, max_val = 1.0, 0.0
    for predicted_prob in y_pred:
        if predicted_prob > max_val:
            max_val = predicted_prob
        elif predicted_prob < min_val:
            min_val = predicted_prob
    span = max_val - min_val
    multiplier = nbins / span
    for true_class, predicted_prob in zip(y_true, y_pred):
        ind = floor((predicted_prob - min_val) * multiplier)
        pockets_predicted[ind] += 1
        pockets_true[ind] += true_class

    idx = np.nonzero(pockets_predicted > 0)[0]

    hits = pockets_predicted[idx]
    if len(hits)>0:
        freqs_predicted, freqs_true = (min_val + (np.arange(nbins)[idx] + 0.5) * span / nbins).astype(np.float64), pockets_true[idx] / pockets_predicted[idx]
    else:
        freqs_predicted, freqs_true=np.array((),dtype=np.float64),np.array((),dtype=np.float64)

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


@njit()
def calibration_metrics_from_freqs(freqs_predicted: np.ndarray, freqs_true: np.ndarray, hits: np.ndarray):
    if len(hits)>0:
        diffs = np.abs((freqs_predicted - freqs_true))
        calibration_mae, calibration_std = np.mean(diffs), np.std(diffs)
    else:
        calibration_mae, calibration_std=1.0,1.0
    
    return calibration_mae, calibration_std    

@njit()
def fast_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray, nbins: int = 100):
    freqs_predicted, freqs_true, hits = fast_calibration_binning(y_true=y_true, y_pred=y_pred, nbins=nbins)
    return calibration_metrics_from_freqs(freqs_predicted=freqs_predicted, freqs_true=freqs_true, hits=hits)


def fast_calibration_report(y_true: np.ndarray, y_pred: np.ndarray, nbins: int = 100, show_plots: bool = True, plot_file: str = "", figsize: tuple = (12, 6)):
    """Bins predictions, then computes regresison-like error metrics between desired and real binned probs."""

    freqs_predicted, freqs_true, hits = fast_calibration_binning(y_true=y_true, y_pred=y_pred, nbins=nbins)
    calibration_mae, calibration_std = calibration_metrics_from_freqs(freqs_predicted=freqs_predicted, freqs_true=freqs_true, hits=hits)

    if plot_file or show_plots:
        show_calibration_plot(
            freqs_predicted=freqs_predicted,
            freqs_true=freqs_true,
            hits=hits,
            plot_title=f"Calibration MAE={calibration_mae:.4f} ± {calibration_std:.4f}",
            show_plots=show_plots,
            plot_file=plot_file,
            figsize=figsize,
        )

    return calibration_mae, calibration_std


def predictions_time_instability(preds: pd.Series) -> float:
    """Computes how stable are true values or predictions over time.
    It's hard to use predictions that change upside down from point to point.
    For binary classification instability ranges from 0 to 1, for regression from 0 to any value depending on the target stats.
    """
    return preds.diff().abs().mean()
