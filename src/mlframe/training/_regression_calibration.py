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


def cv2_recalibration_gain(y_pred_cal: np.ndarray, y_cal: np.ndarray, method: str = "isotonic") -> float:
    """Honest 2-fold gain estimate WITHIN the calib slice (no extra holdout spent).

    Fits ``g`` on each half and scores the other, averaging ``rmse(raw) - rmse(g(raw))`` -- so the
    apply/skip decision never measures gain on the rows ``g`` was fit on (which would be optimistic).
    Used as the ship-gate before applying recalibration to the model. Returns 0.0 when the slice is
    too small to split.
    """
    yp = np.asarray(y_pred_cal, dtype=np.float64).reshape(-1)
    yt = np.asarray(y_cal, dtype=np.float64).reshape(-1)
    finite = np.isfinite(yp) & np.isfinite(yt)
    yp, yt = yp[finite], yt[finite]
    n = yp.size
    if n < 40:
        return 0.0
    mid = n // 2
    g_a = PointRecalibrator(method).fit(yp[:mid], yt[:mid])
    g_b = PointRecalibrator(method).fit(yp[mid:], yt[mid:])
    gain_b = _rmse(yt[mid:], yp[mid:]) - _rmse(yt[mid:], g_a.transform(yp[mid:]))
    gain_a = _rmse(yt[:mid], yp[:mid]) - _rmse(yt[:mid], g_b.transform(yp[:mid]))
    return 0.5 * (gain_a + gain_b)


def duan_log_smearing_factor(residuals_cal: np.ndarray) -> float:
    """Duan (1983) smearing factor for a log-transformed target: ``mean(exp(resid))`` on the log scale.

    Naive back-transform ``exp(pred)`` is biased low (Jensen) when the model is fit on ``log(y)``; the
    unbiased mean estimate is ``exp(pred) * factor`` with this factor computed from held-out log-scale
    residuals. Returns 1.0 (no correction) on too-few / non-finite residuals.
    """
    r = np.asarray(residuals_cal, dtype=np.float64).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size < 5:
        return 1.0
    return float(np.mean(np.exp(r)))


def smearing_predict(pred_transformed: np.ndarray, residuals_cal: np.ndarray, inverse_fn, *, max_cal: int = 2000, seed: int = 0) -> np.ndarray:
    """General Duan smearing: ``yhat(x) = mean_i inverse_fn(pred(x) + resid_i)`` over calib residuals.

    Unbiased back-transform for an arbitrary monotone ``inverse_fn`` (e.g. ``np.expm1`` for log1p). The
    calib residual set is subsampled to ``max_cal`` to bound the O(n_test * n_cal) cost.
    """
    p = np.asarray(pred_transformed, dtype=np.float64).reshape(-1)
    r = np.asarray(residuals_cal, dtype=np.float64).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.asarray(inverse_fn(p), dtype=np.float64)
    if r.size > max_cal:
        r = np.random.default_rng(seed).choice(r, size=max_cal, replace=False)
    # (n_test, n_cal) broadcast then mean over calib; inverse_fn applied elementwise.
    grid = inverse_fn(p[:, None] + r[None, :])
    return np.asarray(np.mean(grid, axis=1), dtype=np.float64)


class DistributionalRecalibrator:
    """Kuleshov-2018 distributional recalibration: make the predicted CDF's PIT uniform.

    Fits the empirical CDF ``R`` of the calibration PIT values (probability-integral transforms of the
    held-out targets through the model's predictive CDF). Applying ``R`` to any PIT maps it toward uniform
    (the PIT theorem), so a quantile predicted at nominal level ``p`` is recalibrated to level ``R(p)`` --
    restoring calibrated coverage for models that emit quantiles/a CDF. Monotone (isotonic), picklable.
    """

    def fit(self, pit_cal: np.ndarray) -> "DistributionalRecalibrator":
        from sklearn.isotonic import IsotonicRegression

        p = np.asarray(pit_cal, dtype=np.float64).reshape(-1)
        p = p[np.isfinite(p)]
        if p.size < 10:
            self._identity = True
            return self
        self._identity = False
        order = np.argsort(p)
        ps = p[order]
        ecdf = (np.arange(1, ps.size + 1)) / (ps.size + 1)  # empirical CDF levels in (0,1)
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(ps, ecdf)
        return self

    def recalibrate(self, pit: np.ndarray) -> np.ndarray:
        """Map PIT (or a nominal level) through R toward a calibrated/uniform scale."""
        p = np.asarray(pit, dtype=np.float64).reshape(-1)
        if getattr(self, "_identity", True):
            return p
        return np.asarray(self._iso.predict(p), dtype=np.float64)


class RecalibratedRegressor:
    """Picklable wrapper that ships ``g(base.predict(X))`` -- the recalibrated predictor.

    Holds the fitted base estimator + a fitted :class:`PointRecalibrator`; ``predict`` applies the
    base model then the monotone map. Module-level (not a closure) so ``dill``/``pickle`` and
    ``sklearn.clone`` round-trip. ``predict_proba`` is intentionally absent (regression only), which
    also lets downstream code keep treating the entry as a regressor.
    """

    def __init__(self, base_model: object, recalibrator: PointRecalibrator) -> None:
        self.base_model = base_model
        self.recalibrator = recalibrator

    def predict(self, X, *args, **kwargs) -> np.ndarray:
        raw = np.asarray(self.base_model.predict(X, *args, **kwargs)).reshape(-1)
        return self.recalibrator.transform(raw)

    def __getattr__(self, name: str):
        # Delegate unknown attributes (feature_names_in_, n_features_in_, ...) to the base model so the
        # wrapper is a drop-in. Guard dunders + the pre-state-restore window during unpickling (when
        # __dict__ has no base_model yet) so we raise AttributeError instead of recursing.
        if name.startswith("__"):
            raise AttributeError(name)
        base = self.__dict__.get("base_model")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)
