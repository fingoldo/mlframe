"""Scoring a spec's y-scale reconstruction on every evaluation row, the way the shipped estimator predicts them.

A gate that scores ``rmse(y_eval[finite], y_hat[finite])`` is measuring a different predictor from the one that ships:
``CompositeTargetEstimator.predict`` never returns a non-finite value, it falls back to the train median (and to the lag
route) for a row whose inverse blows up. Dropping those rows scores the spec only where it behaves, so a spec that
collapses on the unseen-base tail these gates exist to catch - up to half the holdout - passed on the remainder.

The fill is the train-fold median, matching the estimator's fallback, and the RMSE is taken over the full evaluation
population. The finite-fraction reject stays where it is: a spec that collapses on most rows is still rejected outright,
before its median-filled RMSE is looked at.
"""

from __future__ import annotations

import numpy as np

__all__ = ["median_filled_predictions", "median_filled_with_std"]


def median_filled_predictions(y_hat: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
    """``y_hat`` with every non-finite entry replaced by the median of the finite ``y_fit`` values.

    Parameters
    ----------
    y_hat
        The spec's reconstructed y on the evaluation rows.
    y_fit
        The fit-fold y, whose median the estimator falls back to at predict time.

    Returns
    -------
    np.ndarray
        A finite float array of ``y_hat``'s shape (all-non-finite ``y_fit`` leaves the entries as they were).
    """
    out: np.ndarray = np.asarray(y_hat, dtype=np.float64).reshape(-1).copy()
    bad = ~np.isfinite(out)
    if not bad.any():
        return out
    fit = np.asarray(y_fit, dtype=np.float64).reshape(-1)
    fit = fit[np.isfinite(fit)]
    if fit.size:
        out[bad] = float(np.median(fit))
    return out


def median_filled_with_std(y_hat: np.ndarray, y_fit: np.ndarray) -> tuple[np.ndarray, float]:
    """``median_filled_predictions(y_hat, y_fit)`` and the standard deviation of the result, which the collapse checks read.

    Parameters
    ----------
    y_hat
        The spec's reconstructed y on the evaluation rows.
    y_fit
        The fit-fold y, whose median fills a row the inverse could not produce.

    Returns
    -------
    tuple[np.ndarray, float]
        The filled predictions and their spread; a spec that reconstructs few rows now reads as the near-constant it is.
    """
    filled = median_filled_predictions(y_hat, y_fit)
    ok = np.isfinite(filled)
    return filled, (float(np.std(filled[ok])) if ok.any() else 0.0)
