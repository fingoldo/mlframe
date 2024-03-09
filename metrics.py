# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *
from numba import njit
from math import floor
from scipy.special import expit
import numpy as np, pandas as pd
from matplotlib import pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.io import write_image

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
    tmp = tps * fps * 2
    if tmp > 0:
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

    if span > 0:
        multiplier = nbins / span
        for true_class, predicted_prob in zip(y_true, y_pred):
            ind = floor((predicted_prob - min_val) * multiplier)
            pockets_predicted[ind] += 1
            pockets_true[ind] += true_class
    else:
        ind = 0
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
    backend: str = "matplotlib",
    label_freq: str = "Frequency",
    label_perfect: str = "Perfect",
    label_real: str = "Real",
    label_prob: str = "Probability",
    use_size: bool = False,
):
    """Plots reliability digaram from the binned predictions."""

    assert backend in ("plotly", "matplotlib")

    x_min, x_max = np.min(freqs_predicted), np.max(freqs_predicted)

    if backend == "matplotlib":
        fig = plt.figure(figsize=figsize)
        plt.scatter(x=freqs_predicted, y=freqs_true, marker="o", s=5000 * hits / hits.sum(), c=hits, label=label_freq)
        plt.plot([x_min, x_max], [x_min, x_max], "g--", label=label_perfect)
        plt.xlabel(label_prob)
        plt.ylabel(label_freq)
        if plot_title:
            plt.title(plot_title)

        if plot_file:
            fig.savefig(plot_file)

        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    else:

        df = pd.DataFrame(
            {
                label_prob: freqs_predicted,
                label_freq: freqs_true,
                "NCases": hits,
            }
        )
        hover_data = {label_prob: ":.2%", label_freq: ":.2%", "NCases": True}
        print(hover_data)

        if use_size:
            df["size"] = 5000 * hits / hits.sum()
            hover_data["size"] = False

        fig = go.Figure()
        # fig = px.scatter(data_frame=df ,x=label_prob,y=label_freq,size="size" if use_size else None, color="NCases", labels={'x':label_prob, 'y':label_freq},hover_data=hover_data)
        fig.add_trace(
            go.Scatter(
                data_frame=df,
                x=label_prob,
                y=label_freq,
                size="size" if use_size else None,
                color="NCases",
                labels={"x": label_prob, "y": label_freq},
                hover_data=hover_data,
                name=label_real,
            )
        )
        fig.add_trace(go.Scatter(x=[x_min, x_max], y=[x_min, x_max], line={"color": "green", "dash": "dash"}, name=label_perfect, mode="lines"))
        fig.update(layout_coloraxis_showscale=False)
        if plot_title:
            fig.update_layout(title=plot_title)

        if plot_file:
            ext = plot_file.split(".")[-1]
            if not ext:
                ext = "png"
            write_image(fig, file=plot_file, format=ext)

        if show_plots:
            fig.show()
    return fig


@njit()
def maximum_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    return np.nanmax(mape)


@njit()
def calibration_metrics_from_freqs(
    freqs_predicted: np.ndarray, freqs_true: np.ndarray, hits: np.ndarray, nbins: int, array_size: int, use_weights: bool = True
):
    calibration_coverage = len(set(np.round(freqs_predicted, int(np.log10(nbins))))) / nbins
    if len(hits) > 0:
        diffs = np.abs((freqs_predicted - freqs_true))
        if use_weights:
            weights = hits / array_size
            calibration_mae = np.sum(diffs * weights)
            # print(np.sum(diffs),hits,array_size,weights,calibration_mae,weights.sum())
            calibration_std = np.sqrt(np.sum(((diffs - calibration_mae) ** 2) * weights))
        else:
            calibration_mae = np.mean(diffs)
            calibration_std = np.sqrt(np.mean(((diffs - calibration_mae) ** 2)))
    else:
        calibration_mae, calibration_std = 1.0, 1.0

    return calibration_mae, calibration_std, calibration_coverage


@njit()
def fast_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray, nbins: int = 100, use_weights: bool = False, verbose: int = 0):
    freqs_predicted, freqs_true, hits = fast_calibration_binning(y_true=y_true, y_pred=y_pred, nbins=nbins)
    if verbose:
        print(freqs_predicted, freqs_true)
    return calibration_metrics_from_freqs(
        freqs_predicted=freqs_predicted, freqs_true=freqs_true, hits=hits, nbins=nbins, array_size=len(y_true), use_weights=use_weights
    )


def fast_calibration_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nbins: int = 100,
    show_plots: bool = True,
    show_points_density_in_title: bool = False,
    show_roc_auc_in_title: bool = False,
    show_coverage_in_title: bool = False,
    plot_file: str = "",
    figsize: tuple = (12, 6),
    ndigits: int = 2,
    backend: str = "matplotlib",
    title: str = "",
    use_weights=True,
    verbose: bool = False,
):
    """Bins predictions, then computes regresison-like error metrics between desired and real binned probs."""

    assert backend in ("plotly", "matplotlib")

    brier_loss = brier_score_loss(y_true=y_true, y_prob=y_pred)

    freqs_predicted, freqs_true, hits = fast_calibration_binning(y_true=y_true, y_pred=y_pred, nbins=nbins)
    if verbose:
        print("freqs_predicted", freqs_predicted)
        print("freqs_true", freqs_true)
    min_hits, max_hits = np.min(hits), np.max(hits)
    calibration_mae, calibration_std, calibration_coverage = calibration_metrics_from_freqs(
        freqs_predicted=freqs_predicted, freqs_true=freqs_true, hits=hits, nbins=nbins, array_size=len(y_true), use_weights=use_weights
    )

    fig = None
    if plot_file or show_plots:
        plot_title = f"BR={brier_loss*100:.{ndigits}f}% Calibration MAE{'W' if use_weights else ''}={calibration_mae*100:.{ndigits}f}%±{calibration_std*100:.{ndigits}f}%"

        if show_coverage_in_title:
            plot_title += f", cov.={calibration_coverage*100:.{int(np.log10(nbins))}f}%"
        if show_roc_auc_in_title:
            plot_title += f", ROC AUC={fast_auc(y_true=y_true, y_score=y_pred):.3f}"
        if show_points_density_in_title:
            plot_title += f", dens.=[{max_hits:_};{min_hits:_}]"
        if title:
            plot_title = title.strip() + " " + plot_title
        fig = show_calibration_plot(
            freqs_predicted=freqs_predicted,
            freqs_true=freqs_true,
            hits=hits,
            plot_title=plot_title,
            show_plots=show_plots,
            plot_file=plot_file,
            figsize=figsize,
            backend=backend,
        )

    return brier_loss, calibration_mae, calibration_std, calibration_coverage, fig


def predictions_time_instability(preds: pd.Series) -> float:
    """Computes how stable are true values or predictions over time.
    It's hard to use predictions that change upside down from point to point.
    For binary classification instability ranges from 0 to 1, for regression from 0 to any value depending on the target stats.
    """
    return np.abs(np.diff(preds)).mean()


# ----------------------------------------------------------------------------------------------------------------------------
# Errors & scorers
# ----------------------------------------------------------------------------------------------------------------------------


class CB_INTEGRAL_CALIB_ERROR:
    """Custom probabilistic prediction error metric balancing predictive power with calibration.
    Can regularly create a calibration plot.
    """

    def __init__(
        self,
        method: str = "multicrit",
        std_weight: float = 0.9,
        brier_loss_weight: float = 0.5,
        roc_auc_weight=0.5,
        use_weighted_calibration: bool = True,
        weight_by_class_npositives: bool = False,
        calibration_plot_period: int = 0,
    ) -> None:

        assert method in ("multicrit", "precision", "brier_score")

        self.method = method
        self.std_weight = std_weight
        self.roc_auc_weight = roc_auc_weight
        self.brier_loss_weight = brier_loss_weight
        self.use_weighted_calibration = use_weighted_calibration
        self.weight_by_class_npositives = weight_by_class_npositives

        self.calibration_plot_period = calibration_plot_period
        self.nruns = 0

    def is_max_optimal(self):
        return False  # greater is better?

    def evaluate(self, approxes, target, weight):
        output_weight = 1  # weight is not used

        if len(approxes) == 1:
            y_pred = expit(approxes[0])
            probs = [1 - y_pred, y_pred]
            class_id = 1
        else:

            probs = []
            tot_sum = np.zeros_like(approxes[0])
            for class_id in range(len(approxes)):
                y_pred = np.exp(approxes[class_id])
                probs.append(y_pred)
                tot_sum += y_pred
            for class_id in range(len(approxes)):
                probs[class_id] /= tot_sum

        total_error = compute_integral_calibration_error(
            probs=probs,
            target=target,
            method=self.method,
            std_weight=self.std_weight,
            brier_loss_weight=self.brier_loss_weight,
            roc_auc_weight=self.roc_auc_weight,
            use_weighted_calibration=self.use_weighted_calibration,
            weight_by_class_npositives=self.weight_by_class_npositives,
        )

        self.nruns += 1

        if self.calibration_plot_period and (self.nruns % self.calibration_plot_period == 0):
            y_true = (target == class_id).astype(np.int8)
            brier_loss, calibration_mae, calibration_std, calibration_coverage, _ = fast_calibration_report(
                y_true=y_true,
                y_pred=y_pred,
                title=f"{len(approxes[0]):_} records of class {class_id}, integral error={total_error:.4f}, nruns={self.nruns:_}\r\n",
                show_roc_auc_in_title=True,
                use_weights=self.use_weighted_calibration,
                verbose=False,
            )

        return total_error, output_weight

    def get_final_error(self, error, weight):
        return error


def compute_integral_calibration_error(
    probs: Sequence,
    target,
    labels=None,
    method: str = "multicrit",
    std_weight: float = 0.9,
    brier_loss_weight: float = 0.5,
    roc_auc_weight=0.5,
    use_weighted_calibration: bool = True,
    weight_by_class_npositives: bool = False,
    verbose: bool = False,
    ndigits: int = 4,
):
    total_error = 0.0
    weights_sum = 0

    for class_id in range(len(probs)):

        if len(probs) == 2 and class_id == 0:
            continue

        y_pred = probs[class_id]
        if labels is not None:
            y_true = (target == labels[class_id]).astype(np.int8)
        else:
            y_true = (target == class_id).astype(np.int8)

        if method == "multicrit":
            calibration_mae, calibration_std, calibration_coverage = fast_calibration_metrics(
                y_true=y_true, y_pred=y_pred, use_weights=use_weighted_calibration
            )
            brier_loss = brier_score_loss(y_true=y_true, y_prob=y_pred)

            desc_score_indices = np.argsort(y_pred)[::-1]
            roc_auc = fast_numba_auc_nonw(y_true=y_true, y_score=y_pred, desc_score_indices=desc_score_indices)

            if verbose:
                print(
                    f"\t class_id={class_id}, BR={brier_loss:.{ndigits}f}, calibration_mae={calibration_mae:.{ndigits}f} ± {calibration_std:.{ndigits}f}, roc_auc={roc_auc:.{ndigits}f}"
                )

            multicrit_class_error = integral_calibration_error(
                calibration_mae=calibration_mae,
                calibration_std=calibration_std,
                calibration_coverage=calibration_coverage,
                brier_loss=brier_loss,
                roc_auc=roc_auc,
                std_weight=std_weight,
                brier_loss_weight=brier_loss_weight,
                roc_auc_weight=roc_auc_weight,
            )
        if method == "brier_score":
            multicrit_class_error = brier_score_loss(y_true=y_true, y_prob=y_pred)
        elif method == "precision":
            multicrit_class_error = fast_precision(y_true=y_true, y_pred=(y_pred >= 0.5).astype(np.int8), zero_division=0)

        if weight_by_class_npositives:
            weight = y_true.sum()
        else:
            weight = 1

        total_error += multicrit_class_error * weight
        weights_sum += weight

    total_error /= weights_sum

    return total_error


@njit()
def integral_calibration_error(
    calibration_mae: float,
    calibration_std: float,
    calibration_coverage: float,
    brier_loss: float,
    roc_auc: float,
    std_weight: float = 0.9,
    brier_loss_weight: float = 0.5,
    roc_auc_weight: float = 0.5,
) -> float:
    """Integral calibration error."""
    return brier_loss * brier_loss_weight + calibration_mae + calibration_std * std_weight - np.abs(roc_auc - 0.5) * roc_auc_weight


def sklearn_integral_calibration_error(
    y_true,
    y_score,
    labels=None,
    method: str = "multicrit",
    std_weight: float = 0.9,
    brier_loss_weight: float = 0.5,
    roc_auc_weight=0.5,
    use_weighted_calibration: bool = True,
    weight_by_class_npositives: bool = False,
    verbose: bool = False,
):
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    if isinstance(y_score, (pd.Series, pd.DataFrame)):
        y_score = y_score.values
    if labels is not None and isinstance(labels, (pd.Series, pd.DataFrame)):
        labels = labels.values
    return compute_integral_calibration_error(
        probs=[y_score[:, i] for i in range(y_score.shape[1])],
        target=y_true,
        labels=labels,
        method=method,
        std_weight=std_weight,
        brier_loss_weight=brier_loss_weight,
        roc_auc_weight=roc_auc_weight,
        use_weighted_calibration=use_weighted_calibration,
        weight_by_class_npositives=weight_by_class_npositives,
        verbose=verbose,
    )


def integral_calibration_error_xgboost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    calibration_mae, calibration_std, calibration_coverage = fast_calibration_metrics(y_true=y_true, y_pred=y_pred)
    return integral_calibration_error(calibration_mae=calibration_mae, calibration_std=calibration_std, calibration_coverage=calibration_coverage)


def integral_calibration_error_keras(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    calibration_mae, calibration_std, calibration_coverage = fast_calibration_metrics(y_true=y_true.numpy()[:, -1], y_pred=y_pred.numpy()[:, -1])
    return integral_calibration_error(calibration_mae=calibration_mae, calibration_std=calibration_std, calibration_coverage=calibration_coverage)


@njit()
def brier_score_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return np.mean((y_true - y_prob) ** 2)


@njit()
def probability_separation_score(y_true: np.ndarray, y_prob: np.ndarray, class_label: int = 1, std_weight: float = 0.5) -> float:
    idx = y_true == class_label
    if idx.sum() == 0:
        return np.nan
        if class_label == 1:
            res = 0.0
        else:
            res = 1.0
    else:
        res = np.mean(y_prob[idx])
        if std_weight != 0.0:
            addend = np.std(y_prob[idx]) * std_weight
            if class_label == 1:
                res = res - addend
            else:
                res = res + addend
    return res
