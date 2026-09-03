"""``find_prediction_band_shift``/``apply_prediction_band_correction``: targeted sub-band multiplicative fix.

Source: 4th_home-credit-default-risk.md - "if you correct your prediction for revolving loan that is over
0.4 by 0.8, it will boost your auc" - a targeted post-hoc multiplicative correction applied only to
predictions above a threshold, tied to a discovered train/OOF prevalence mismatch for a subpopulation.
Distinct from :func:`mlframe.calibration.asymmetric_rescale` (a reciprocal two-sided correction pivoted at 0)
and from the group-key-based `group_bias_correction`/`apply_smoothed_override`: this targets an arbitrary
VALUE-RANGE band, gated explicitly on measured OOF evidence (mean(y_true)/mean(y_pred) inside the band) rather
than blind leaderboard-probing - production-usable wherever detectable feature/prevalence drift concentrates
in a specific prediction sub-range. ``assess_prediction_band_stability`` is an opt-in bootstrap-resample
reliability check: a band fit on few OOF rows risks the shift factor overfitting sampling noise, so this
reports how much the factor moves under resampling before a caller trusts and ships it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class BandStabilityReport:
    """Bootstrap-resample stability report for a ``find_prediction_band_shift`` fit.

    A single hard-edged band fit on OOF data risks overfitting noise when the band holds few samples --
    ``factor`` alone gives no signal for that. This report resamples (with replacement) the in-band rows
    ``n_bootstrap`` times, recomputes the shift each time, and summarizes the resulting distribution so a
    caller can tell a genuine subpopulation miscalibration (tight bootstrap spread) from a sparse-band
    correction that is mostly sampling noise (wide spread relative to the fitted factor).
    """

    factor: float
    band_n: int
    bootstrap_mean: float
    bootstrap_std: float
    ci_lo: float
    ci_hi: float
    relative_std: float
    is_stable: bool


def assess_prediction_band_stability(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lo: float,
    hi: float,
    n_bootstrap: int = 500,
    ci: float = 0.90,
    max_relative_std: float = 0.15,
    min_band_n: int = 30,
    random_state: Optional[int] = None,
) -> BandStabilityReport:
    """Bootstrap the ``find_prediction_band_shift`` estimate to gauge how trustworthy it is.

    Opt-in companion to :func:`find_prediction_band_shift` - does not alter that function or
    :func:`apply_prediction_band_correction` in any way; call this separately once a candidate band has
    been found, before deciding whether to apply the correction in production.

    Parameters
    ----------
    y_true, y_pred
        ``(n,)`` OOF ground truth and predictions.
    lo, hi
        Prediction-value band, ``(lo, hi]``.
    n_bootstrap
        Number of bootstrap resamples drawn (with replacement) from the in-band rows.
    ci
        Two-sided bootstrap confidence-interval mass reported via ``ci_lo``/``ci_hi`` (default ``0.90``).
    max_relative_std
        ``is_stable`` requires ``bootstrap_std / |factor|`` at or below this fraction (default ``0.15``).
    min_band_n
        ``is_stable`` also requires at least this many in-band rows, since a tiny band can look
        artificially tight under bootstrap (few distinct values to resample from).
    random_state
        Optional seed for the bootstrap resampling RNG.

    Returns
    -------
    BandStabilityReport
        ``factor`` mirrors :func:`find_prediction_band_shift`'s point estimate on the full band; the rest
        characterizes the bootstrap distribution of that estimate. ``is_stable`` is ``False`` when the band
        is empty or too small to resample meaningfully.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    mask = (y_pred_arr > lo) & (y_pred_arr <= hi)
    band_n = int(mask.sum())
    factor = find_prediction_band_shift(y_true_arr, y_pred_arr, lo, hi)

    if band_n < 2:
        return BandStabilityReport(
            factor=factor, band_n=band_n, bootstrap_mean=factor, bootstrap_std=0.0, ci_lo=factor, ci_hi=factor, relative_std=0.0, is_stable=False
        )

    band_true = y_true_arr[mask]
    band_pred = y_pred_arr[mask]
    rng = np.random.default_rng(random_state)
    # A resample whose in-band mean prediction is exactly zero is SKIPPED, not recorded as 1.0. Substituting the
    # meaningful value "no correction" puts a point that is not a draw from the estimator sampling distribution
    # into that distribution: over a band like (-1e-9, 1e-9], or any band where positive and negative predictions
    # cancel, a meaningful fraction of the resamples land there, `bootstrap_std` then measures the spread between
    # the real factor and a pile of 1.0s, and `is_stable` is decided from that mixture -- invisibly, because the
    # point estimate uses the same 1.0 convention. This mirrors the skip-and-count discipline in
    # `evaluation/bootstrap.py`.
    collected = []
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, band_n, size=band_n)
        sample_pred_mean = band_pred[sample_idx].mean()
        if sample_pred_mean == 0.0:
            continue
        collected.append(band_true[sample_idx].mean() / sample_pred_mean)
    n_degenerate = n_bootstrap - len(collected)
    if n_degenerate:
        logger.warning(
            "assess_prediction_band_stability: %d/%d bootstrap resample(s) had an exactly-zero in-band mean "
            "prediction and were skipped; the band straddles zero, so the correction factor is poorly determined.",
            n_degenerate,
            n_bootstrap,
        )
    if len(collected) < 2:
        return BandStabilityReport(
            factor=factor,
            band_n=band_n,
            bootstrap_mean=factor,
            bootstrap_std=float("nan"),
            ci_lo=float("nan"),
            ci_hi=float("nan"),
            relative_std=float("inf"),
            is_stable=False,
        )
    boot_factors = np.asarray(collected, dtype=np.float64)

    bootstrap_mean = float(boot_factors.mean())
    bootstrap_std = float(boot_factors.std(ddof=1))
    alpha = (1.0 - ci) / 2.0
    ci_lo = float(np.quantile(boot_factors, alpha))
    ci_hi = float(np.quantile(boot_factors, 1.0 - alpha))
    relative_std = bootstrap_std / abs(factor) if factor != 0.0 else float("inf")
    # More than a tenth of the resamples degenerate means the estimate rests on a shrinking, biased subsample.
    is_stable = band_n >= min_band_n and relative_std <= max_relative_std and n_degenerate <= n_bootstrap // 10

    return BandStabilityReport(
        factor=factor,
        band_n=band_n,
        bootstrap_mean=bootstrap_mean,
        bootstrap_std=bootstrap_std,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        relative_std=relative_std,
        is_stable=is_stable,
    )


__all__ = ["find_prediction_band_shift", "apply_prediction_band_correction", "BandStabilityReport", "assess_prediction_band_stability"]
