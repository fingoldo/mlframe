"""Production drift monitoring for a deployed ``CompositeTargetEstimator``.

:class:`CompositeDriftMonitor` is a READ-ONLY companion to a fitted
``CompositeTargetEstimator``. Given a stream / batch of new ``(X, y?)`` it
computes cheap drift signals against the train-time fitted state and emits
a structured report ``{signal -> {value, alert}}`` with configurable
thresholds. It never copies the frame -- only narrow column pulls (the same
``_extract_base`` one-ndarray path the estimator uses) plus the y-scale
``predict`` the estimator already exposes.

Signals
-------
1. ``base_psi[col]`` / ``base_ks[col]`` -- per base-column distribution drift
   (Population Stability Index + Kolmogorov-Smirnov) between the train-time
   base quantile sketch and the new batch base.
2. ``prediction_psi`` -- PSI of the new batch ``y_hat`` vs a stored train
   ``y_hat`` sketch.
3. ``residual_mean_shift`` / ``residual_scale_shift`` -- when ``y`` is given,
   the standardised shift of ``(y - y_hat)`` mean and the log-ratio of its
   scale vs the train residual, plus a rolling-window ``residual_rmse``.

The train-time sketch (small per-column quantile knots + the train residual
mean / std) is stored once on the estimator under
``fitted_params_["_drift_sketch"]`` so a second monitor reuses it; when the
estimator was loaded without a sketch the caller supplies a reference batch
(held-out / recent-clean rows) and the sketch is built from that. The sketch
is fit ONLY on the rows the caller passes as reference -- never on the
monitored stream -- so the comparison stays honest (a drifted stream cannot
define its own "normal").

A high-drift alert sets ``report["recommend_update"] = True``, the hook the
``online_refit`` (``CompositeTargetEstimator.update``) streaming path keys on.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default number of quantile knots in a column sketch. 10 deciles is the
# standard PSI bucketisation -- cheap to store, robust on a few-hundred batch.
_DEFAULT_SKETCH_QUANTILES: int = 10
# PSI laplace floor so an empty bin does not blow the log to +inf.
_PSI_EPS: float = 1e-6


def _finite_1d(arr: Any) -> np.ndarray:
    """Return a finite 1-D float64 view (non-finite rows dropped)."""
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    return a[np.isfinite(a)]


def _quantile_knots(values: np.ndarray, n_quantiles: int) -> np.ndarray:
    """Unique, sorted interior quantile edges of ``values``.

    Returns up to ``n_quantiles - 1`` interior edges; duplicates collapse so a
    near-constant column does not manufacture zero-width PSI bins.
    """
    v = _finite_1d(values)
    if v.size == 0:
        return np.array([0.0], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_quantiles + 1)[1:-1]
    edges = np.quantile(v, qs)
    return np.unique(edges)


def _bin_fractions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Fraction of ``values`` falling in each bin defined by ``edges``.

    ``edges`` are the interior cut points; the result has ``len(edges) + 1``
    bins. A Laplace ``_PSI_EPS`` floor is added then re-normalised so PSI and
    the KS step never hit a zero bin.
    """
    v = _finite_1d(values)
    n_bins = edges.size + 1
    if v.size == 0:
        return np.full(n_bins, 1.0 / n_bins, dtype=np.float64)
    counts = np.histogram(v, bins=np.concatenate(([-np.inf], edges, [np.inf])))[0]
    frac = counts.astype(np.float64) + _PSI_EPS
    return frac / frac.sum()


def _uniform_ref_fractions(edges: np.ndarray) -> np.ndarray:
    """Reference bin fractions for a QUANTILE sketch: uniform by construction.

    The stored edges are interior quantile knots, so each bin holds an equal
    share of the train mass (1 / n_bins). We use this analytic uniform vector
    as the PSI reference rather than re-binning the few knot points (which is
    degenerate -- K points cannot populate K+1 bins evenly).
    """
    n_bins = edges.size + 1
    return np.full(n_bins, 1.0 / n_bins, dtype=np.float64)


def _psi(ref_frac: np.ndarray, new_frac: np.ndarray) -> float:
    """Population Stability Index between two binned fraction vectors.

    ``sum((new - ref) * ln(new / ref))``. Both inputs are already floored +
    normalised, so the log is finite.
    """
    return float(np.sum((new_frac - ref_frac) * np.log(new_frac / ref_frac)))


def _ks_statistic(ref_knots: np.ndarray, new_values: np.ndarray) -> float:
    """Two-sample KS supremum distance between the train sketch CDF and the
    new batch ECDF, evaluated at the stored knots + the new sample points.

    The train side is represented by its quantile knots (a piecewise-uniform
    CDF approximation): the knot at position ``i`` carries CDF level
    ``(i + 1) / (K + 1)``. The new side uses the exact ECDF. The supremum is
    taken over the union of evaluation points -- cheap and sketch-only on the
    reference side (no train rows retained).
    """
    new = np.sort(_finite_1d(new_values))
    if new.size == 0 or ref_knots.size == 0:
        return 0.0
    k = ref_knots.size
    # Midpoint plotting positions (i + 0.5)/k span (0,1) symmetrically and reach ~1.0 at the top knot; the prior (i + 1)/(k + 1) capped the reference CDF at k/(k+1) < 1, adding a constant ~1/(k+1) positive bias to every KS stat -> false drift alerts even on a no-drift batch.
    ref_levels = (np.arange(k, dtype=np.float64) + 0.5) / k
    points = np.unique(np.concatenate([ref_knots, new]))
    # Train CDF at each point: fraction of knots <= point, scaled to the knot
    # level grid (right-continuous step through the stored quantile levels).
    ref_idx = np.searchsorted(ref_knots, points, side="right")
    ref_cdf = np.where(ref_idx > 0, ref_levels[np.clip(ref_idx - 1, 0, k - 1)], 0.0)
    new_cdf = np.searchsorted(new, points, side="right") / new.size
    return float(np.max(np.abs(ref_cdf - new_cdf)))


class CompositeDriftMonitor:
    """Read-only drift monitor for a fitted ``CompositeTargetEstimator``.

    Parameters
    ----------
    estimator
        A fitted ``CompositeTargetEstimator`` (has ``fitted_params_`` +
        ``predict``). The monitor never mutates it except to memoise the
        train-time sketch under ``fitted_params_["_drift_sketch"]``.
    base_psi_threshold, prediction_psi_threshold
        PSI levels above which the corresponding signal alerts. PSI > 0.25 is
        the classic "large population shift" line; > 0.1 is "moderate".
    base_ks_threshold
        KS supremum distance above which a base column alerts.
    residual_mean_z_threshold
        Absolute standardised residual-mean shift above which the residual
        signal alerts (``|mean(y - y_hat)| / train_resid_std``).
    residual_scale_log_threshold
        ``|ln(new_resid_std / train_resid_std)|`` above which the residual
        scale signal alerts (ln 1.5 ~= 0.405 -> a 50% scale change).
    rolling_rmse_window
        Number of recent residual RMSE observations kept in a rolling deque
        for the ``residual_rmse`` trend (FIFO).
    sketch_quantiles
        Quantile-knot count per column sketch (default 10 deciles).
    recommend_update_on
        Which signals, when alerting, set ``recommend_update=True`` (the hook
        the ``online_refit`` ``update()`` path consumes). Default: any base or
        residual alert (prediction PSI alone is advisory).
    """

    def __init__(
        self,
        estimator: Any,
        *,
        base_psi_threshold: float = 0.25,
        prediction_psi_threshold: float = 0.25,
        base_ks_threshold: float = 0.2,
        residual_mean_z_threshold: float = 3.0,
        residual_scale_log_threshold: float = 0.405,
        rolling_rmse_window: int = 20,
        sketch_quantiles: int = _DEFAULT_SKETCH_QUANTILES,
        recommend_update_on: Sequence[str] = ("base", "residual"),
    ) -> None:
        if not hasattr(estimator, "fitted_params_"):
            raise ValueError(
                "CompositeDriftMonitor: estimator is not fitted "
                "(no fitted_params_); fit it before monitoring."
            )
        self.estimator = estimator
        self.base_psi_threshold = float(base_psi_threshold)
        self.prediction_psi_threshold = float(prediction_psi_threshold)
        self.base_ks_threshold = float(base_ks_threshold)
        self.residual_mean_z_threshold = float(residual_mean_z_threshold)
        self.residual_scale_log_threshold = float(residual_scale_log_threshold)
        self.sketch_quantiles = int(sketch_quantiles)
        self.recommend_update_on = tuple(recommend_update_on)
        self._rolling_rmse: Deque[float] = deque(maxlen=int(rolling_rmse_window))

    # ------------------------------------------------------------------
    # Sketch construction (train-time reference state)
    # ------------------------------------------------------------------

    def _base_columns(self) -> Tuple[str, ...]:
        """Resolve the estimator's base column(s); empty for unary transforms."""
        resolver = getattr(self.estimator, "_resolve_base_columns", None)
        if resolver is not None:
            try:
                return tuple(resolver())
            except Exception:  # pragma: no cover - defensive
                pass
        bc = getattr(self.estimator, "base_columns", None)
        if bc:
            return tuple(bc)
        single = getattr(self.estimator, "base_column", "")
        return (single,) if single else ()

    def ensure_sketch(self, reference: Any = None, y_reference: Any = None) -> Dict[str, Any]:
        """Return the train-time drift sketch, building + memoising it if absent.

        The sketch carries per-base-column quantile knots, a ``y_hat`` quantile
        sketch, and the train residual ``(mean, std)``. It is stored on the
        estimator under ``fitted_params_["_drift_sketch"]`` so repeat monitors
        reuse it. When the fitted estimator carries no sketch (loaded from an
        old pickle), ``reference`` (a clean / held-out batch) is required and
        the sketch is built from it -- NEVER from the monitored stream.
        """
        fp = self.estimator.fitted_params_
        existing = fp.get("_drift_sketch")
        if existing is not None:
            return existing
        if reference is None:
            raise ValueError(
                "CompositeDriftMonitor: the estimator carries no stored drift "
                "sketch; pass a clean `reference` batch (held-out / recent rows) "
                "so the train-time reference distribution can be sketched. The "
                "monitored stream must NOT be used as its own reference."
            )
        sketch = self._build_sketch(reference, y_reference)
        fp["_drift_sketch"] = sketch
        return sketch

    def _build_sketch(self, reference: Any, y_reference: Any) -> Dict[str, Any]:
        """Build the sketch from a reference batch (no row retention -- knots only)."""
        from .estimator import _extract_base

        base_knots: Dict[str, np.ndarray] = {}
        for col in self._base_columns():
            base_knots[col] = _quantile_knots(
                _extract_base(reference, col), self.sketch_quantiles,
            )
        y_hat = np.asarray(self.estimator.predict(reference), dtype=np.float64).reshape(-1)
        pred_knots = _quantile_knots(y_hat, self.sketch_quantiles)
        resid_mean, resid_std = float("nan"), float("nan")
        if y_reference is not None:
            y_arr = np.asarray(y_reference, dtype=np.float64).reshape(-1)
            resid = y_arr - y_hat
            resid = resid[np.isfinite(resid)]
            if resid.size > 1:
                resid_mean = float(np.mean(resid))
                resid_std = float(np.std(resid))
        return {
            "base_knots": base_knots,
            "pred_knots": pred_knots,
            "resid_mean": resid_mean,
            "resid_std": resid_std,
            "n_reference": int(np.asarray(y_hat).size),
        }

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def monitor(
        self,
        X: Any,
        y: Any = None,
        *,
        reference: Any = None,
        y_reference: Any = None,
    ) -> Dict[str, Any]:
        """Compute drift signals for a new batch ``(X, y?)`` vs the train sketch.

        Returns a structured report::

            {
              "signals": {name -> {"value": float, "alert": bool}},
              "alert": bool,                  # any signal alerted
              "recommend_update": bool,       # a `recommend_update_on` alerted
              "n": int,                       # rows scored
              "has_y": bool,
            }

        ``reference`` / ``y_reference`` are forwarded to :meth:`ensure_sketch`
        only when the estimator carries no stored sketch.
        """
        sketch = self.ensure_sketch(reference=reference, y_reference=y_reference)
        signals: Dict[str, Dict[str, Any]] = {}
        from .estimator import _extract_base

        # (1) base-column distribution drift (PSI + KS per column).
        for col, knots in sketch["base_knots"].items():
            new_base = _extract_base(X, col)
            ref_frac = _uniform_ref_fractions(knots)
            new_frac = _bin_fractions(new_base, knots)
            psi_val = _psi(ref_frac, new_frac)
            ks_val = _ks_statistic(knots, new_base)
            signals[f"base_psi[{col}]"] = {
                "value": psi_val, "alert": psi_val > self.base_psi_threshold,
            }
            signals[f"base_ks[{col}]"] = {
                "value": ks_val, "alert": ks_val > self.base_ks_threshold,
            }

        # (2) prediction drift (PSI of y_hat vs the train y_hat sketch).
        y_hat = np.asarray(self.estimator.predict(X), dtype=np.float64).reshape(-1)
        pred_knots = sketch["pred_knots"]
        pred_psi = _psi(_uniform_ref_fractions(pred_knots), _bin_fractions(y_hat, pred_knots))
        signals["prediction_psi"] = {
            "value": pred_psi, "alert": pred_psi > self.prediction_psi_threshold,
        }

        has_y = y is not None
        # (3) residual drift (only when y is available).
        if has_y:
            self._residual_signals(y, y_hat, sketch, signals)

        any_alert = any(s["alert"] for s in signals.values())
        recommend = self._recommend_update(signals)
        return {
            "signals": signals,
            "alert": bool(any_alert),
            "recommend_update": bool(recommend),
            "n": int(y_hat.size),
            "has_y": bool(has_y),
        }

    def _residual_signals(
        self, y: Any, y_hat: np.ndarray, sketch: Dict[str, Any],
        signals: Dict[str, Dict[str, Any]],
    ) -> None:
        """Populate residual mean / scale shift + rolling RMSE signals in place."""
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        resid = y_arr - y_hat
        finite = np.isfinite(resid)
        resid = resid[finite]
        train_mean = sketch.get("resid_mean", float("nan"))
        train_std = sketch.get("resid_std", float("nan"))
        if resid.size == 0:
            return
        new_mean = float(np.mean(resid))
        new_std = float(np.std(resid))
        rmse = float(np.sqrt(np.mean(resid * resid)))
        self._rolling_rmse.append(rmse)
        # Standardise the mean shift by the TRAIN residual scale (the deployed
        # reference); fall back to the new-batch scale when the sketch lacks a
        # train residual (no y_reference was given at sketch time).
        denom = train_std if (np.isfinite(train_std) and train_std > 1e-12) else new_std
        if denom <= 1e-12:
            denom = 1.0
        ref_mean = train_mean if np.isfinite(train_mean) else 0.0
        mean_z = abs(new_mean - ref_mean) / denom
        signals["residual_mean_shift"] = {
            "value": float(mean_z),
            "alert": mean_z > self.residual_mean_z_threshold,
        }
        if np.isfinite(train_std) and train_std > 1e-12 and new_std > 1e-12:
            scale_log = abs(float(np.log(new_std / train_std)))
        else:
            scale_log = 0.0
        signals["residual_scale_shift"] = {
            "value": float(scale_log),
            "alert": scale_log > self.residual_scale_log_threshold,
        }
        signals["residual_rmse"] = {
            "value": rmse,
            "alert": False,  # trend signal, not a hard gate
            "rolling_mean": float(np.mean(self._rolling_rmse)) if self._rolling_rmse else rmse,
        }

    def _recommend_update(self, signals: Dict[str, Dict[str, Any]]) -> bool:
        """True when an alerting signal belongs to a ``recommend_update_on`` family."""
        for name, sig in signals.items():
            if not sig["alert"]:
                continue
            family = "residual" if name.startswith("residual") else (
                "prediction" if name.startswith("prediction") else "base"
            )
            if family in self.recommend_update_on:
                return True
        return False

    @property
    def rolling_rmse(self) -> Tuple[float, ...]:
        """Snapshot of the rolling residual-RMSE history (oldest -> newest)."""
        return tuple(self._rolling_rmse)
