"""Assesing quality of a classifier in terms of how often probabilities predicted by it convert into real events.
"""
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

ensure_installed("pandas numpy properscoring scikit-learn")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import pandas as pd, numpy as np
from matplotlib import pyplot as plt

from sklearn.feature_selection import mutual_info_regression
from properscoring import brier_score, crps_ensemble
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    explained_variance_score,
    r2_score,
    mean_squared_log_error,
    mean_absolute_percentage_error,
)  # ,mean_pinball_loss

from sklearn.metrics import brier_score_loss  # , log_loss

# ----------------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------------


def hyvarinen_score(y: np.ndarray, y_preds: np.ndarray) -> float:
    return mutual_info_regression(y.reshape(-1, 1), y_preds.reshape(-1, 1), n_neighbors=2)[0]


def crps(y: np.ndarray, y_preds: np.ndarray) -> float:
    """Computes mean Continuous Ranked Probability Score of true binary outcomes versus predicted probabilities."""
    return crps_ensemble(observations=y, forecasts=y_preds).mean()


# ----------------------------------------------------------------------------------------------------------------------------
# Core functionality
# ----------------------------------------------------------------------------------------------------------------------------
METRICS_TO_SHOW = {
    #
    "R2": r2_score,
    # "EV": explained_variance_score,
    #
    # "MSE": mean_squared_error,
    # "MSLE":mean_squared_log_error,
    # "MAE": mean_absolute_error,
    # "MAPE": mean_absolute_percentage_error,
    # "MEAE": median_absolute_error,
    "BR": brier_score_loss,
    # "HS": hyvarinen_score,
    "CRPS": crps,
    #
    # "MPL": mean_pinball_loss,
    # "SEP": get_separation_percent,
    # "BL": get_betting_loss,
}


def make_custom_calibration_plot(
    y: np.ndarray,
    probs: np.ndarray,
    nclasses: int,
    classes=[],
    nbins: int = 100,
    competing_probs: list = [],
    X: np.ndarray = None,
    display_labels: dict = {},
    figsize: tuple = (15, 5),
):
    """Custom implementanion of calibration plot"""
    
    metrics={}
    if not classes:
        classes=range(nclasses)
    else:
        nclasses=len(classes)
    fig, ax_probs = plt.subplots(ncols=nclasses, nrows=1, sharex=False, sharey=False, figsize=figsize)
    for pos_label in classes:

        title = f"Calibration plot for {display_labels.get(pos_label,'class '+str(pos_label))}:"
        # fig.suptitle(title)

        if type(probs) == np.ndarray:
            prob_pos = probs[:, pos_label]
        else:
            prob_pos = probs.iloc[:, pos_label].values

        if isinstance(y, np.ndarray):
            y_true = (y == pos_label).astype(np.int8)
        elif isinstance(y, (pd.DataFrame, pd.Series)):
            y_true = (y.values == pos_label).astype(np.int8)
        else:
            raise TypeError("Unexpected y type: %s", type(y))

        class_performance_metrics=show_classifier_calibration(y_true, prob_pos, legend_label="Model Probs", ax=ax_probs if nclasses==1 else ax_probs[pos_label], title=title, append=False, nbins=nbins)
        metrics[pos_label]=class_performance_metrics

        # Same axis, competing probs, if any

        for competing_vars in competing_probs:
            if len(competing_vars[pos_label]) > 0:
                var_name = competing_vars[pos_label]
                prob_pos = X[var_name]
            else:
                named_vars = [var for var in competing_vars if len(var) > 0]

                prob_pos = 1.0 - X[named_vars].sum(axis=1)
                var_name = named_vars[0]  # any of them

            if type(prob_pos) != np.ndarray:
                prob_pos = prob_pos.values
            var_name = "_".join(var_name.split("_")[1:])
            show_classifier_calibration(y_true, prob_pos, legend_label=var_name, ax=ax_probs[pos_label], title=title, append=True, nbins=nbins)    
    return fig,metrics


# @njit()
def estimate_calibration_quality_binned(
    y_true: np.array,
    y_predicted: np.array,
    nbins: int = 20,
    metrics_to_show: dict = METRICS_TO_SHOW,
):
    indices = np.argsort(y_predicted)
    pockets_predicted, pockets_true, data = [], [], []
    s = len(y_predicted)
    l = 0
    bin_size = s // nbins
    for i in range(nbins):
        if i == nbins - 1:
            r = s
        else:
            r = l + bin_size
        avg_x = np.mean(y_predicted[indices[l:r]])
        avg_y = np.mean(y_true[indices[l:r]])
        pockets_predicted.append(avg_x)
        pockets_true.append(avg_y)
        data.append([avg_x, avg_y * (r - l), r - l, avg_y])
        l = r
    pockets_predicted, pockets_true = np.array(pockets_predicted), np.array(pockets_true)
    # r2 = np.corrcoef(pockets_predicted, pockets_true)[0, 1] ** 2

    return (
        pockets_predicted,
        pockets_true,
        data,
        {fname: (f(y_true, y_predicted) if f == brier_score_loss else f(pockets_true, pockets_predicted)) for fname, f in metrics_to_show.items()},
    )


def show_classifier_calibration(
    y_true,
    y_predicted,
    title,
    nbins=20,
    alpha=0.40,
    show_table=False,
    nintervals=1,
    ax=None,
    marker_size: int = 15,
    metrics_digits: int = 4,
    connected: bool = True,
    legend_label: str = None,
    append: bool = False,
    metrics_to_show: dict = METRICS_TO_SHOW,
):

    s = len(y_true)
    step = s // nintervals
    l = 0

    if ax is None:
        ax = plt

    for i in range(nintervals):
        if i == nintervals - 1:
            r = s
        else:
            r = l + step

        try:
            x, y, data, performances = estimate_calibration_quality_binned(y_true[l:r], y_predicted[l:r], nbins=nbins, metrics_to_show=metrics_to_show)
        except Exception as e:
            logging.exception(e)
            return
        metrics_formatted = " ".join([f"{metric_name}: {round(metric_value,metrics_digits)}" for metric_name, metric_value in performances.items()])

        if legend_label:
            metrics_formatted = legend_label + ": " + metrics_formatted

        if connected:
            ax.plot(x, y, alpha=alpha, label=metrics_formatted, markersize=marker_size, marker="o")
        else:
            ax.scatter(x, y, alpha=alpha, label=metrics_formatted, s=marker_size)
        l = r
    x_min, x_max = np.min(x), np.max(x)
    #y_min, y_max = np.min(y), np.max(y)
    is_profit = "profit" in title.lower()
    ax.legend(loc="lower right")
    if not append:
        # Set general params for the first time
        ax.plot([x_min, x_max], [x_min, x_max], "g--", label="Perfect")
        try:
            ax.set_xlabel("Expected")
            ax.set_ylabel("Real")
            ax.set_title("%s, %d bins, %d points" % (title, nbins, len(y_true)))
        except:
            pass

        if is_profit:
            ax.axhline(0.0, color="g", linestyle="--")
            ax.axvline(0.0, color="g", linestyle="--")
        # if x_max>=1:
        #    ax.ylim([-.10, 1])
        #    ax.xlim([-.10, 1])
    # plt.show(block=False)
    # plt.pause(0.001)
    if show_table:
        if is_profit:
            return pd.DataFrame(data, columns=["Predicted ROI", "TotalWinnings", "NBets", "Real ROI"])
        else:
            return pd.DataFrame(data, columns=["Prob", "Won", "Predicted", "Freq"])
    else:
        return performances