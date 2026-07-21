"""``GatedRegressionMixture``: gate classifier routes to branch regressors, gate probability stacked as a feature.

Source: Elo Merchant Category Recommendation 5th place -- an outlier-probability binary classifier used for
(a) routing rows into a "low probability" vs "high probability" regression model by threshold, (b) as an
input feature to the high-probability regressor (stacked), and (c) downweighting rare outliers in the
low-probability regressor's loss.

Distinct from :class:`GatedOutlierEstimator` (this session's earlier addition): that class BLENDS a
classifier probability with ONE regressor's prediction (``p*constant + (1-p)*regressor.predict(X)``) and
never exposes the probability to the regressor itself. This class instead does HARD threshold ROUTING to N
independently-configured branch regressors (each can be a different estimator, tuned differently, matching
the source's "low prob" vs "high prob" specialist split), and -- the source's distinguishing "stacked
feature" trick -- appends the gate's own OOF probability as an EXTRA INPUT COLUMN to each branch regressor,
so a branch regressor can use "how confident the gate was" as a smooth signal on top of the hard route.
Per-branch ``sample_weight`` multipliers implement the source's rare-outlier downweighting.

Leakage discipline: the gate classifier's probabilities used to route/feature-stack the TRAINING rows are
OOF (via ``sklearn.model_selection.cross_val_predict(method="predict_proba")``) -- no row's own label leaks
into its own gate probability.

Opt-in soft routing (``soft_routing=True``): hard 0/1 threshold routing means two test rows with near-
identical gate probability straddling ``threshold`` can land on two independently-fit branch regressors and
jump discontinuously. Soft routing blends both branches' predictions by gate probability for rows within
``soft_routing_bandwidth`` of the threshold, smoothing that boundary without touching training-time branch
assignment or the default (``soft_routing=False``) predict path.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

logger = logging.getLogger(__name__)

_LOW = "low"
_HIGH = "high"


def _concat_feature(X: Any, col_name: str, values: np.ndarray) -> Any:
    """Append ``values`` as a new column named ``col_name`` to ``X``, matching its frame type."""
    if isinstance(X, pd.DataFrame):
        out = X.copy()
        out[col_name] = values
        return out
    try:
        import polars as pl
        if isinstance(X, pl.DataFrame):
            return X.with_columns(pl.Series(col_name, values))
    except ImportError:
        pass
    return np.concatenate([np.asarray(X, dtype=np.float64), values.reshape(-1, 1)], axis=1)


class GatedRegressionMixture(BaseEstimator, RegressorMixin):
    """Gate classifier hard-routes rows to branch regressors; gate probability is stacked as a feature.

    Parameters
    ----------
    gate_classifier
        sklearn-compatible probabilistic classifier prototype, cloned and OOF-fit against a binary
        subpopulation label (e.g. "is this row an outlier").
    low_regressor, high_regressor
        sklearn-compatible regressor prototypes for the below-threshold ("low" gate probability) and
        at-or-above-threshold ("high") branches respectively.
    threshold
        Gate probability cutoff for routing (default 0.5).
    use_gate_feature
        If True (default), the gate's OOF probability is appended as an extra input column to BOTH branch
        regressors at fit time, and the fitted gate's probability likewise at predict time.
    branch_sample_weight
        ``{"low": w, "high": w}`` multipliers applied to that branch's rows in its regressor's
        ``sample_weight`` (matches the source's rare-outlier downweighting; default ``{"low": 1.0, "high":
        1.0}``, i.e. no reweighting).
    n_splits, random_state
        Gate classifier's OOF CV configuration.
    soft_routing
        Opt-in (default ``False`` -- predict-time behavior is then bit-identical to hard threshold routing).
        When ``True``, rows whose gate probability falls within ``soft_routing_bandwidth`` of ``threshold``
        get a PROBABILITY-WEIGHTED BLEND of both branch predictions instead of a hard 0/1 route, removing the
        prediction discontinuity a training rows would otherwise see at the cut point (two rows with near-
        identical gate probability straddling the threshold can currently land on differently-fit branch
        regressors and jump). Rows outside the band are still hard-routed to a single branch (same as
        ``soft_routing=False``), so this only touches the transition zone. Training/routing of rows into
        branches at ``fit`` time is unaffected -- branches are still assigned by hard OOF threshold routing.
    soft_routing_bandwidth
        Half-width, in gate-probability units, of the blend zone straddling ``threshold`` (default 0.1, i.e.
        rows with ``proba`` in ``[threshold - 0.1, threshold + 0.1]`` are blended). Ignored unless
        ``soft_routing=True``.

    Attributes
    ----------
    gate_model_, branch_models_
        The fitted gate classifier and ``{"low": regressor, "high": regressor}``.
    """

    def __init__(
        self,
        gate_classifier: Any,
        low_regressor: Any,
        high_regressor: Any,
        threshold: float = 0.5,
        use_gate_feature: bool = True,
        branch_sample_weight: Optional[Dict[str, float]] = None,
        n_splits: int = 5,
        random_state: int = 42,
        soft_routing: bool = False,
        soft_routing_bandwidth: float = 0.1,
    ) -> None:
        self.gate_classifier = gate_classifier
        self.low_regressor = low_regressor
        self.high_regressor = high_regressor
        self.threshold = threshold
        self.use_gate_feature = use_gate_feature
        self.branch_sample_weight = branch_sample_weight
        self.n_splits = n_splits
        self.random_state = random_state
        self.soft_routing = soft_routing
        self.soft_routing_bandwidth = soft_routing_bandwidth

    def _predict_proba_1(self, model: Any, X: Any) -> np.ndarray:
        """Return the positive-class probability column from a classifier's ``predict_proba``."""
        return np.asarray(model.predict_proba(X), dtype=np.float64)[:, 1]

    def fit(self, X: Any, y: Any, subpop_label: np.ndarray) -> "GatedRegressionMixture":
        """Fit the gate classifier via OOF probabilities, then fit a branch regressor per routed subpopulation."""
        y_arr = np.asarray(y, dtype=np.float64)
        label_arr = np.asarray(subpop_label)
        weights = self.branch_sample_weight or {_LOW: 1.0, _HIGH: 1.0}

        # composite_oof_predictions calls .predict(), a poor proxy for a probability (a classifier's hard-
        # label predict discards calibration); use cross_val_predict(method="predict_proba") directly so the
        # gate feature/route reflects the classifier's true probabilistic output.
        from sklearn.model_selection import KFold, cross_val_predict

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_proba = cross_val_predict(clone(self.gate_classifier), X, label_arr, cv=kf, method="predict_proba")[:, 1]

        self.gate_model_ = clone(self.gate_classifier)
        self.gate_model_.fit(X, label_arr)

        route = np.where(oof_proba >= self.threshold, _HIGH, _LOW)
        self.branch_models_: Dict[str, Any] = {}
        for branch, regressor in ((_LOW, self.low_regressor), (_HIGH, self.high_regressor)):
            mask = route == branch
            if not mask.any():
                logger.warning("GatedRegressionMixture: branch %s has no routed rows at fit time.", branch)
                continue
            X_branch = X.iloc[np.flatnonzero(mask)] if hasattr(X, "iloc") else np.asarray(X)[mask]
            if self.use_gate_feature:
                X_branch = _concat_feature(X_branch, "gate_proba", oof_proba[mask])
            model = clone(regressor)
            sw = np.full(int(mask.sum()), weights.get(branch, 1.0), dtype=np.float64)
            try:
                model.fit(X_branch, y_arr[mask], sample_weight=sw)
            except TypeError:
                model.fit(X_branch, y_arr[mask])
            self.branch_models_[branch] = model
        return self

    def _predict_branch(self, branch: str, X: Any, proba: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Predict with the given branch's model on the rows selected by ``mask``."""
        X_branch = X.iloc[np.flatnonzero(mask)] if hasattr(X, "iloc") else np.asarray(X)[mask]
        if self.use_gate_feature:
            X_branch = _concat_feature(X_branch, "gate_proba", proba[mask])
        return np.asarray(self.branch_models_[branch].predict(X_branch), dtype=np.float64)

    def predict(self, X: Any) -> np.ndarray:
        """Route each row by gate probability and predict via the routed (or blended) branch model(s)."""
        proba = self._predict_proba_1(self.gate_model_, X)
        route = np.where(proba >= self.threshold, _HIGH, _LOW)
        n = proba.shape[0]
        out = np.zeros(n, dtype=np.float64)

        # Soft-routing blend zone: rows whose gate probability sits within `soft_routing_bandwidth` of
        # `threshold`. Disabled by default (band is empty) so hard-routed rows below take the exact same
        # single-branch code path as before this feature existed -- bit-identical when unused.
        band = np.zeros(n, dtype=bool)
        bw = self.soft_routing_bandwidth
        lo = self.threshold - bw
        hi = self.threshold + bw
        if self.soft_routing:
            band = (proba >= lo) & (proba <= hi)

        for branch in (_LOW, _HIGH):
            mask = (route == branch) & ~band
            if not mask.any():
                continue
            if branch not in self.branch_models_:
                # This branch had zero routed rows at FIT time (so no model exists), but genuinely does
                # receive rows at PREDICT time (an imbalanced subpop_label distribution or an extreme
                # threshold makes this realistic, not just theoretical). Falling straight through left
                # out[mask] at its np.zeros init value -- a silent, undetectable wrong prediction. Fall back to the OTHER
                # branch's model (mirrors the soft-routing band's own have_low/have_high fallback below),
                # or raise if neither branch was ever fitted.
                other = _HIGH if branch == _LOW else _LOW
                if other not in self.branch_models_:
                    raise RuntimeError(
                        f"GatedRegressionMixture.predict: neither branch has a fitted model; cannot predict "
                        f"for {int(mask.sum())} row(s) routed to branch {branch!r}."
                    )
                logger.warning(
                    "GatedRegressionMixture: branch %s has no fitted model (zero rows routed to it at fit "
                    "time); falling back to branch %s for %d predict-time row(s).",
                    branch, other, int(mask.sum()),
                )
                out[mask] = self._predict_branch(other, X, proba, mask)
                continue
            out[mask] = self._predict_branch(branch, X, proba, mask)

        if band.any():
            have_low = _LOW in self.branch_models_
            have_high = _HIGH in self.branch_models_
            if have_low and have_high:
                low_pred = self._predict_branch(_LOW, X, proba, band)
                high_pred = self._predict_branch(_HIGH, X, proba, band)
                # Linear ramp across the band: 0 at the low edge (pure low-branch), 1 at the high edge (pure
                # high-branch), matching the hard-routing decision exactly at the two band edges.
                span = hi - lo
                weight = (proba[band] - lo) / span if span > 0 else np.where(proba[band] >= self.threshold, 1.0, 0.0)
                weight = np.clip(weight, 0.0, 1.0)
                out[band] = weight * high_pred + (1.0 - weight) * low_pred
            elif have_low:
                out[band] = self._predict_branch(_LOW, X, proba, band)
            elif have_high:
                out[band] = self._predict_branch(_HIGH, X, proba, band)
        return out


__all__ = ["GatedRegressionMixture"]
