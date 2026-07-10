"""Cheap overfit-risk flag for isotonic-regression calibration, complementary to full nested-CV selection.

Isotonic regression fits a free-form monotone step function, so its in-sample calibration error can be
driven arbitrarily close to zero simply by adding more breakpoints — the exact failure mode the Elo Merchant
7th-place team hit ("Isotonic regression give me 0.005~0.006 boost both on cv and lb but pb score become
worse... Isotonic regression overfit both on cv and lb"). ``policy.pick_best_calibrator`` already offers a
full nested-CV selection safeguard (fit on a fold, score on a disjoint fold) for choosing AMONG calibrators;
this module adds a much cheaper, isotonic-specific RED FLAG that needs no extra CV split: count how many
distinct step segments the fit uses relative to the sample size. A step function with a breakpoint every few
samples is fitting noise, not a genuine monotone relationship, regardless of how good its in-sample ECE looks.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def isotonic_overfit_risk(
    calib_p: np.ndarray,
    calib_y: np.ndarray,
    segment_ratio_threshold: float = 0.05,
) -> dict:
    """Fit isotonic regression on ``(calib_p, calib_y)`` and flag it as overfit-risky by segment density.

    Parameters
    ----------
    calib_p
        ``(n,)`` predicted probabilities to calibrate.
    calib_y
        ``(n,)`` binary outcomes aligned to ``calib_p``.
    segment_ratio_threshold
        A fit is flagged when ``n_segments / n`` exceeds this. ``n_segments`` counts the distinct constant
        pieces of the fitted step function (i.e. how many times the fitted value changes when walking the
        calibration points in sorted-``calib_p`` order) — a genuine monotone relationship in real data
        needs far fewer breakpoints than one sample per few points; a high ratio means the fit is tracking
        individual points' noise rather than a smooth underlying trend.

    Returns
    -------
    dict
        ``{"n_samples", "n_segments", "segment_ratio", "flagged", "isotonic_fit"}`` — ``isotonic_fit`` is
        the fitted ``sklearn.isotonic.IsotonicRegression`` instance (reusable by the caller; refitting it
        would be wasted work).
    """
    p = np.asarray(calib_p, dtype=np.float64).ravel()
    y = np.asarray(calib_y, dtype=np.float64).ravel()
    n = p.shape[0]
    if n != y.shape[0]:
        raise ValueError(f"isotonic_overfit_risk: calib_p length {n} != calib_y length {y.shape[0]}")
    if n < 2:
        raise ValueError(f"isotonic_overfit_risk: need at least 2 samples, got {n}")

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)

    order = np.argsort(p, kind="stable")
    fitted_sorted = iso.predict(p[order])
    # Count distinct constant pieces: a "segment" boundary is any place the fitted value strictly changes.
    n_segments = 1 + int(np.count_nonzero(np.diff(fitted_sorted) > 1e-12))
    segment_ratio = n_segments / n

    return {
        "n_samples": n,
        "n_segments": n_segments,
        "segment_ratio": segment_ratio,
        "flagged": segment_ratio > segment_ratio_threshold,
        "isotonic_fit": iso,
    }


__all__ = ["isotonic_overfit_risk"]
