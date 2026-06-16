"""Point recalibration of regression predictions: a monotone ``g(yhat) ~= E[y|yhat]``.

The classification analog (Platt/isotonic) is standard; the regression analog is underused
even though regression is routinely miscalibrated -- shrinkage to the centre (ridge,
early-stopped GBMs, tree averaging compress the tails so ``E[y|yhat] != yhat`` on the
extremes), back-transform bias, distribution shift. Fitting a MONOTONE map on a held-out
calibration slice corrects the shrinkage without ever changing the ranking, so it can only
help a shrunk model and is ~identity (a no-op) on an already-calibrated one.

Fit on the disjoint calibration slice (``calib_size``); the conformal residuals are then taken
on the recalibrated predictor on the SEPARATE ``conformal_size`` slice. Monotone-preserving by
construction (isotonic increasing / positive-slope linear). Ship only when it measurably beats
identity on honest holdout -- ``recalibration_rmse_gain`` is the gate (REJECTED != DELETED).
"""

from __future__ import annotations

import numpy as np

_VALID_METHODS = ("isotonic", "linear")


class PointRecalibrator:
    """Monotone ``g(yhat) ~= E[y|yhat]`` fit on a held-out slice; picklable (module-level class).

    ``method="isotonic"`` -> non-parametric increasing map (``sklearn.isotonic`` with clip
    out-of-bounds), the most flexible monotone correction. ``method="linear"`` -> affine
    ``a*yhat + b`` (slope clamped non-negative to preserve ranking). ``transform`` applies the
    fitted map; ``fit`` returns self. Falls back to identity when the slice is too small or
    degenerate (constant predictions) so it never raises on edge inputs.
    """

    def __init__(self, method: str = "isotonic") -> None:
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
        self.method = method
        self._fitted = False
        self._identity = False

    def fit(self, y_pred_cal: np.ndarray, y_cal: np.ndarray) -> "PointRecalibrator":
        yp = np.asarray(y_pred_cal, dtype=np.float64).reshape(-1)
        yt = np.asarray(y_cal, dtype=np.float64).reshape(-1)
        if yp.shape != yt.shape:
            raise ValueError(f"y_pred_cal {yp.shape} and y_cal {yt.shape} must match")
        finite = np.isfinite(yp) & np.isfinite(yt)
        yp, yt = yp[finite], yt[finite]
        # Too few points or a degenerate (constant) predictor -> nothing to learn; stay identity.
        if yp.size < 10 or float(np.ptp(yp)) == 0.0:
            self._identity = True
            self._fitted = True
            return self
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self._model = IsotonicRegression(increasing=True, out_of_bounds="clip")
            self._model.fit(yp, yt)
        else:
            # Affine fit; clamp slope to >= 0 so the map is monotone non-decreasing (ranking-safe).
            slope, intercept = np.polyfit(yp, yt, deg=1)
            self._slope = max(0.0, float(slope))
            self._intercept = float(intercept)
        self._fitted = True
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PointRecalibrator.transform called before fit")
        yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        if self._identity:
            return yp
        if self.method == "isotonic":
            return np.asarray(self._model.predict(yp), dtype=np.float64)
        return self._slope * yp + self._intercept


def fit_point_recalibrator(y_pred_cal: np.ndarray, y_cal: np.ndarray, method: str = "isotonic") -> PointRecalibrator:
    """Fit a :class:`PointRecalibrator` on a held-out calibration slice."""
    return PointRecalibrator(method=method).fit(y_pred_cal, y_cal)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_true, dtype=np.float64).reshape(-1) - np.asarray(y_pred, dtype=np.float64).reshape(-1)
    finite = np.isfinite(d)
    if not finite.any():
        return float("inf")
    return float(np.sqrt(np.mean(d[finite] ** 2)))


def recalibration_rmse_gain(
    recal: PointRecalibrator,
    y_pred_test: np.ndarray,
    y_true_test: np.ndarray,
) -> float:
    """Honest-holdout RMSE improvement from recalibration: ``rmse(raw) - rmse(g(raw))``.

    Positive -> recalibration helps (ship); ~0 / negative -> already calibrated or harmful (keep
    identity). This is the gate that decides whether to apply ``g``, never assume it helps.
    """
    raw = _rmse(y_true_test, y_pred_test)
    recalibrated = _rmse(y_true_test, recal.transform(y_pred_test))
    return raw - recalibrated
