"""Cheap overfit-risk flag for isotonic-regression calibration, complementary to full nested-CV selection.

Isotonic regression fits a free-form monotone step function, so its in-sample calibration error can be
driven arbitrarily close to zero simply by adding more breakpoints — the exact failure mode the Elo Merchant
7th-place team hit ("Isotonic regression give me 0.005~0.006 boost both on cv and lb but pb score become
worse... Isotonic regression overfit both on cv and lb"). ``policy.pick_best_calibrator`` already offers a
full nested-CV selection safeguard (fit on a fold, score on a disjoint fold) for choosing AMONG calibrators;
this module adds a much cheaper, isotonic-specific RED FLAG that needs no extra CV split: count how many
distinct step segments the fit uses relative to the sample size. A step function with a breakpoint every few
samples is fitting noise, not a genuine monotone relationship, regardless of how good its in-sample ECE looks.

``remediate=True`` closes the loop from diagnostic to corrective action: when the flag trips, a Platt
(logistic) fit is trained alongside isotonic and a blended predictor is returned that weights isotonic vs.
Platt per-query by LOCAL sample density around that probability value — sparse segments (few training points
nearby, where isotonic is provably overfitting) fall back toward the smooth, low-variance Platt fit; dense
segments keep trusting isotonic's more flexible shape. This is opt-in and changes nothing about the existing
diagnostic-only return keys/values when omitted.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _make_blended_predict(
    iso: IsotonicRegression,
    platt: LogisticRegression,
    train_p_sorted: np.ndarray,
    density_window: float,
    min_local_points: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a closure that blends isotonic and Platt predictions, weighted by local training density.

    For each query point ``x``, ``local_count`` is the number of training ``calib_p`` values within
    ``density_window`` of ``x`` (found via searchsorted on the pre-sorted training array — O(log n) per
    query). The isotonic weight is ``clip(local_count / min_local_points, 0, 1)``: a query landing in a
    segment with at least ``min_local_points`` nearby training samples trusts isotonic fully; a query in a
    sparse segment blends toward Platt in proportion to how starved of local data it is.
    """

    def _predict(query_p: np.ndarray) -> np.ndarray:
        """Blend isotonic and Platt predictions for ``query_p``, weighted by local training density."""
        q = np.asarray(query_p, dtype=np.float64).ravel()
        lo_idx = np.searchsorted(train_p_sorted, q - density_window, side="left")
        hi_idx = np.searchsorted(train_p_sorted, q + density_window, side="right")
        local_count = (hi_idx - lo_idx).astype(np.float64)
        weight_iso = np.clip(local_count / min_local_points, 0.0, 1.0)

        iso_pred = iso.predict(q)
        platt_pred = platt.predict_proba(q.reshape(-1, 1))[:, 1]
        blended: np.ndarray = weight_iso * iso_pred + (1.0 - weight_iso) * platt_pred
        return blended

    return _predict


def isotonic_overfit_risk(
    calib_p: np.ndarray,
    calib_y: np.ndarray,
    segment_ratio_threshold: float = 0.05,
    remediate: bool = False,
    density_window: float = 0.05,
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
    remediate
        Opt-in. When ``True`` AND the fit is flagged, also fits a Platt (logistic) calibrator and returns
        a ``predict`` closure blending isotonic with Platt, weighted by local segment density — the
        corrective counterpart to the diagnostic flag. Default ``False`` reproduces the original
        diagnostic-only behavior exactly (no extra fitting work, no new non-``None`` keys populated).
    density_window
        Half-width, in probability units, of the local-density window used to compute the isotonic-vs-Platt
        blend weight when ``remediate=True``. Only used when remediation actually fires.

    Returns
    -------
    dict
        ``{"n_samples", "n_segments", "segment_ratio", "flagged", "isotonic_fit", "remediated",
        "platt_fit", "predict"}``. ``isotonic_fit`` is the fitted ``sklearn.isotonic.IsotonicRegression``
        instance (reusable by the caller; refitting it would be wasted work). ``remediated`` is ``True``
        only when ``remediate=True`` AND ``flagged`` is ``True``; otherwise ``platt_fit``/``predict`` stay
        ``None`` and callers should keep using ``isotonic_fit`` directly, exactly as before this parameter
        existed.
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
    p_sorted = p[order]
    fitted_sorted = iso.predict(p_sorted)
    # Count distinct constant pieces: a "segment" boundary is any place the fitted value strictly changes.
    n_segments = 1 + int(np.count_nonzero(np.diff(fitted_sorted) > 1e-12))
    segment_ratio = n_segments / n
    flagged = segment_ratio > segment_ratio_threshold

    result: dict = {
        "n_samples": n,
        "n_segments": n_segments,
        "segment_ratio": segment_ratio,
        "flagged": flagged,
        "isotonic_fit": iso,
        "remediated": False,
        "platt_fit": None,
        "predict": None,
    }

    if remediate and flagged:
        platt = LogisticRegression()
        platt.fit(p.reshape(-1, 1), y)
        # A segment needs at least "1 / segment_ratio_threshold" neighbors to be considered dense enough
        # for isotonic's per-point flexibility to be trustworthy under the SAME rule used to flag it.
        min_local_points = max(1.0, 1.0 / segment_ratio_threshold)
        result["remediated"] = True
        result["platt_fit"] = platt
        result["predict"] = _make_blended_predict(iso, platt, p_sorted, density_window, min_local_points)

    return result


__all__ = ["isotonic_overfit_risk"]
