"""``find_prediction_band_shift``/``apply_prediction_band_correction``: targeted sub-band multiplicative fix.

Source: 4th_home-credit-default-risk.md -- "if you correct your prediction for revolving loan that is over
0.4 by 0.8, it will boost your auc" -- a targeted post-hoc multiplicative correction applied only to
predictions above a threshold, tied to a discovered train/OOF prevalence mismatch for a subpopulation.
Distinct from :func:`mlframe.calibration.asymmetric_rescale` (a reciprocal two-sided correction pivoted at 0)
and from the group-key-based `group_bias_correction`/`apply_smoothed_override`: this targets an arbitrary
VALUE-RANGE band, gated explicitly on measured OOF evidence (mean(y_true)/mean(y_pred) inside the band) rather
than blind leaderboard-probing -- production-usable wherever detectable feature/prevalence drift concentrates
in a specific prediction sub-range.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def find_prediction_band_shift(y_true: np.ndarray, y_pred: np.ndarray, lo: float, hi: float) -> float:
    """Measure the multiplicative correction factor implied by OOF evidence within ``(lo, hi]``.

    Parameters
    ----------
    y_true, y_pred
        ``(n,)`` OOF ground truth and predictions.
    lo, hi
        Prediction-value band, ``(lo, hi]``.

    Returns
    -------
    float
        ``mean(y_true) / mean(y_pred)`` among rows with ``lo < y_pred <= hi``. ``1.0`` (no correction) if the
        band is empty or ``mean(y_pred)`` in the band is ``0``.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    mask = (y_pred_arr > lo) & (y_pred_arr <= hi)
    if not mask.any():
        return 1.0
    pred_mean = float(y_pred_arr[mask].mean())
    if pred_mean == 0.0:
        return 1.0
    return float(y_true_arr[mask].mean() / pred_mean)


def apply_prediction_band_correction(y_pred: np.ndarray, lo: float, hi: float, factor: float, clip: Optional[Tuple[float, float]] = (0.0, 1.0)) -> np.ndarray:
    """Multiply predictions within ``(lo, hi]`` by ``factor``; predictions outside the band are unchanged.

    Parameters
    ----------
    y_pred
        ``(n,)`` predictions to correct.
    lo, hi
        Prediction-value band, ``(lo, hi]``.
    factor
        Multiplicative correction applied to in-band predictions.
    clip
        ``(low, high)`` bounds applied AFTER correction (default ``(0, 1)``, appropriate for probability
        predictions), or ``None`` to skip clipping.

    Returns
    -------
    np.ndarray
        ``(n,)`` corrected predictions.
    """
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    mask = (pred_arr > lo) & (pred_arr <= hi)
    corrected = np.where(mask, pred_arr * factor, pred_arr)
    if clip is not None:
        corrected = np.clip(corrected, clip[0], clip[1])
    return np.asarray(corrected)


__all__ = ["find_prediction_band_shift", "apply_prediction_band_correction"]
