"""Post-hoc recalibration of a cross-target ensemble's blended output.

The cross-target ensemble blends K component predictions with NNLS / Ridge / gain-weighted weights. Those weights minimise a (weighted) squared error on the OOF surface but leave NO guarantee that the blended output is *calibrated*: a least-squares blend of biased components is itself biased, and the bias is often S-shaped (the blend over-predicts in one range and under-predicts in another). A monotone recalibration map fit on the OOF blended output vs the truth removes that systematic, order-preserving distortion without touching the ranking the ensemble learned.

This module provides :class:`OutputCalibrator`, a one-feature monotone regressor fit on ``(raw_ensemble_pred, y)`` pairs. Three methods:

- ``"isotonic"`` (default): a free-form monotone non-decreasing step map (``sklearn.isotonic.IsotonicRegression``). Corrects an arbitrary S-shaped / saturating miscalibration. Out-of-range inputs are clipped to the fitted edge values (``out_of_bounds="clip"``).
- ``"sigmoid"``: Platt-style 1-D logistic-link affine map fit by least squares on a centred-logit basis -- a smooth monotone S-correction with only 2 parameters (robust when OOF is small / noisy). Falls back to linear when the link fit is degenerate.
- ``"linear"``: a 1-D ordinary least squares ``a * pred + b`` -- corrects only a global scale + offset bias (the cheapest map; use when the miscalibration is affine, e.g. NNLS shrinkage).

Leakage contract
----------------
The calibrator MUST be fit on OUT-OF-FOLD ensemble predictions (predictions made on rows the components were not trained on). Fitting on in-sample blended predictions would learn the components' overfit and inflate the apparent gain while hurting honest holdout. The ensemble wires this by fitting on the same OOF holdout matrix the NNLS / gain weights were derived from -- never on a re-prediction of the training frame. ``fit`` itself is agnostic to where the predictions came from; the *caller* owns the OOF guarantee, exactly like the weight solvers.

Default OFF / bit-identity
--------------------------
``CompositeCrossTargetEnsemble`` ships with ``calibrate_output=False``; with no calibrator attached ``predict`` returns the raw blend bit-for-bit. The recalibration is opt-in and only ever applied as a final monotone post-map, so enabling it cannot change which component dominates a prediction -- only the y-scale it lands on.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_VALID_METHODS = ("isotonic", "sigmoid", "linear")


class OutputCalibrator:
    """Fitted monotone recalibration map for a 1-D ensemble output.

    Provenance / formula
    --------------------
    Given raw ensemble predictions ``p_i`` and truths ``y_i`` on an OOF surface, fit a monotone non-decreasing map ``g`` minimising ``sum_i w_i (g(p_i) - y_i)^2`` and apply ``y_hat = g(raw_pred)`` at predict.

    - ``isotonic``: ``g`` is the pool-adjacent-violators isotonic fit (free-form, monotone non-decreasing), clipped to the fitted ``[p_min, p_max]`` range at the edges.
    - ``linear``: ``g(p) = a*p + b`` with ``(a, b)`` the weighted OLS slope/intercept; ``a`` is clamped to ``>= 0`` so the map stays monotone (a degenerate negative-slope fit collapses to ``a=0`` i.e. the weighted-mean constant).
    - ``sigmoid``: ``g(p) = lo + (hi-lo) * sigmoid(A*z + B)`` where ``z`` is the min-max-normalised raw pred and ``(A, B)`` are the OLS fit of the centred logit of the normalised truth on ``z``; a smooth 2-parameter monotone S-map. Falls back to ``linear`` when the link basis is degenerate (constant pred, zero target range).

    The map is always monotone non-decreasing, so it preserves the ensemble's ordering of any two rows; it only re-spaces predictions onto the truth scale.
    """

    def __init__(self, method: str = "isotonic") -> None:
        if method not in _VALID_METHODS:
            raise ValueError(
                f"OutputCalibrator: method must be one of {_VALID_METHODS}; got {method!r}."
            )
        self.method = method
        self._fitted = False
        # Per-method state (set in fit).
        self._iso: Any = None
        self._lin_a: float = 1.0
        self._lin_b: float = 0.0
        self._sig_A: float = 1.0
        self._sig_B: float = 0.0
        self._sig_lo: float = 0.0
        self._sig_hi: float = 1.0
        self._sig_pmin: float = 0.0
        self._sig_pspan: float = 1.0
        self._sig_degenerate: bool = False

    # ------------------------------------------------------------------
    @staticmethod
    def _clean(
        raw_pred: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Coerce to 1-D float64 + drop non-finite rows, keeping arrays aligned."""
        p = np.asarray(raw_pred, dtype=np.float64).reshape(-1)
        t = np.asarray(y, dtype=np.float64).reshape(-1)
        if p.shape[0] != t.shape[0]:
            raise ValueError(
                f"OutputCalibrator: raw_pred length {p.shape[0]} != y length {t.shape[0]}."
            )
        finite = np.isfinite(p) & np.isfinite(t)
        w = None
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if w.shape[0] != p.shape[0]:
                raise ValueError(
                    f"OutputCalibrator: sample_weight length {w.shape[0]} != n {p.shape[0]}."
                )
            if not np.all(np.isfinite(w)) or (w < 0).any():
                raise ValueError("OutputCalibrator: sample_weight must be finite and non-negative.")
            finite &= np.isfinite(w)
            w = w[finite]
        return p[finite], t[finite], w

    def fit(
        self,
        raw_pred: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "OutputCalibrator":
        """Fit the monotone map on ``(raw_pred, y)`` OOF pairs.

        Needs at least 3 finite rows; below that (or a degenerate constant-prediction column) it fits the identity map so ``predict`` is a safe pass-through. The caller is responsible for ensuring ``raw_pred`` are OUT-OF-FOLD predictions (no leakage).
        """
        p, t, w = self._clean(raw_pred, y, sample_weight)
        if p.shape[0] < 3 or np.ptp(p) <= 0.0:
            # Too few rows or constant predictions: identity map (pass-through).
            self.method = "linear"
            self._lin_a, self._lin_b = 1.0, 0.0
            self._fitted = True
            logger.debug(
                "OutputCalibrator.fit: degenerate input (n=%d, ptp=%.3g); using identity map.",
                p.shape[0], float(np.ptp(p)) if p.shape[0] else 0.0,
            )
            return self
        if self.method == "isotonic":
            self._fit_isotonic(p, t, w)
        elif self.method == "linear":
            self._fit_linear(p, t, w)
        else:
            self._fit_sigmoid(p, t, w)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def _fit_isotonic(self, p: np.ndarray, t: np.ndarray, w: np.ndarray | None) -> None:
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(p, t, sample_weight=w)
        self._iso = iso

    def _fit_linear(self, p: np.ndarray, t: np.ndarray, w: np.ndarray | None) -> None:
        a, b = _weighted_ols_1d(p, t, w)
        # Keep the map monotone non-decreasing: a negative slope would invert the
        # ensemble's ranking, which a recalibration map must never do. Collapse to
        # the (constant) weighted mean of the target instead.
        if a < 0.0:
            a = 0.0
            b = float(np.average(t, weights=w))
        self._lin_a, self._lin_b = float(a), float(b)

    def _fit_sigmoid(self, p: np.ndarray, t: np.ndarray, w: np.ndarray | None) -> None:
        pmin = float(p.min())
        pspan = float(p.max() - p.min())
        if pspan <= 0.0:
            self.method = "linear"
            return self._fit_linear(p, t, w)
        z = (p - pmin) / pspan  # normalised raw pred in [0, 1]
        lo = float(np.min(t))
        hi = float(np.max(t))
        tspan = hi - lo
        if tspan <= 0.0:
            self.method = "linear"
            return self._fit_linear(p, t, w)
        # Logit of the min-max-normalised truth, clipped off the asymptotes so the
        # link stays finite; fit (A, B) by weighted OLS of logit(t_norm) on z.
        tn = np.clip((t - lo) / tspan, 1e-4, 1.0 - 1e-4)
        link = np.log(tn / (1.0 - tn))
        A, B = _weighted_ols_1d(z, link, w)
        if not (np.isfinite(A) and np.isfinite(B)) or A <= 0.0:
            # Degenerate / non-monotone link fit -> fall back to the affine map.
            self.method = "linear"
            return self._fit_linear(p, t, w)
        self._sig_A, self._sig_B = float(A), float(B)
        self._sig_lo, self._sig_hi = lo, hi
        self._sig_pmin, self._sig_pspan = pmin, pspan

    # ------------------------------------------------------------------
    def predict(self, raw_pred: np.ndarray) -> np.ndarray:
        """Apply the fitted monotone map to ``raw_pred`` (1-D y-scale output)."""
        if not self._fitted:
            raise RuntimeError("OutputCalibrator.predict: not fitted.")
        p = np.asarray(raw_pred, dtype=np.float64).reshape(-1)
        if self.method == "isotonic":
            # sklearn's IsotonicRegression.predict rejects NaN/inf, so map only the
            # finite rows and pass non-finite raw preds straight through unchanged
            # (a single bad row must not raise nor corrupt to an edge constant).
            bad = ~np.isfinite(p)
            out = p.astype(np.float64, copy=True)
            good = ~bad
            if good.any():
                out[good] = np.asarray(
                    self._iso.predict(p[good]), dtype=np.float64,
                ).reshape(-1)
            return out
        if self.method == "linear":
            return self._lin_a * p + self._lin_b
        # sigmoid
        z = (p - self._sig_pmin) / self._sig_pspan
        s = 1.0 / (1.0 + np.exp(-(self._sig_A * z + self._sig_B)))
        out = self._sig_lo + (self._sig_hi - self._sig_lo) * s
        bad = ~np.isfinite(p)
        if bad.any():
            out[bad] = p[bad]
        return out

    # ------------------------------------------------------------------
    def export(self) -> dict[str, Any]:
        """Plain-dict snapshot for metadata storage."""
        d: dict[str, Any] = {"method": self.method, "fitted": bool(self._fitted)}
        if self.method == "linear":
            d["linear"] = {"a": self._lin_a, "b": self._lin_b}
        elif self.method == "sigmoid":
            d["sigmoid"] = {
                "A": self._sig_A, "B": self._sig_B,
                "lo": self._sig_lo, "hi": self._sig_hi,
                "pmin": self._sig_pmin, "pspan": self._sig_pspan,
            }
        elif self.method == "isotonic" and self._iso is not None:
            d["isotonic"] = {
                "x_min": float(getattr(self._iso, "X_min_", np.nan)),
                "x_max": float(getattr(self._iso, "X_max_", np.nan)),
            }
        return d


def _weighted_ols_1d(
    x: np.ndarray, y: np.ndarray, w: np.ndarray | None,
) -> tuple[float, float]:
    """Weighted ordinary least squares slope/intercept for ``y ~ a*x + b``.

    Returns ``(a, b)``. Falls back to ``(0, weighted_mean(y))`` when ``x`` has zero (weighted) variance.
    """
    if w is None:
        xm = float(np.mean(x))
        ym = float(np.mean(y))
        cov = float(np.mean((x - xm) * (y - ym)))
        var = float(np.mean((x - xm) ** 2))
    else:
        wsum = float(w.sum())
        if wsum <= 0:
            return 0.0, float(np.mean(y))
        xm = float(np.sum(w * x) / wsum)
        ym = float(np.sum(w * y) / wsum)
        cov = float(np.sum(w * (x - xm) * (y - ym)) / wsum)
        var = float(np.sum(w * (x - xm) ** 2) / wsum)
    if var <= 0.0:
        return 0.0, ym
    a = cov / var
    b = ym - a * xm
    return a, b


def fit_output_calibrator(
    raw_oof_pred: np.ndarray,
    y_oof: np.ndarray,
    method: str = "isotonic",
    sample_weight: np.ndarray | None = None,
) -> OutputCalibrator | None:
    """Fit an :class:`OutputCalibrator` on OOF blended predictions vs truth.

    Returns the fitted calibrator, or ``None`` when the inputs are too small / degenerate to fit anything useful (caller then keeps the raw blend, i.e. behaves as if calibration were off). The ``raw_oof_pred`` MUST be out-of-fold ensemble predictions to avoid leakage; this function does not (and cannot) verify that -- it is the caller's contract, mirroring the NNLS / Ridge weight solvers.
    """
    try:
        p = np.asarray(raw_oof_pred, dtype=np.float64).reshape(-1)
        t = np.asarray(y_oof, dtype=np.float64).reshape(-1)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("fit_output_calibrator: could not coerce inputs (%s); skipping calibration.", exc)
        return None
    if p.shape[0] != t.shape[0] or p.shape[0] < 3:
        logger.debug(
            "fit_output_calibrator: too few / mismatched rows (pred=%d, y=%d); skipping calibration.",
            p.shape[0], t.shape[0],
        )
        return None
    try:
        return OutputCalibrator(method=method).fit(p, t, sample_weight=sample_weight)
    except Exception as exc:
        logger.warning(
            "fit_output_calibrator: calibrator fit failed (%s); skipping calibration (raw blend kept).",
            exc,
        )
        return None
