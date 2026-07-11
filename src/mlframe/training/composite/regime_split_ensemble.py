"""``RegimeSplitEnsemble``: one model per market/operating regime, combined by routing or averaging.

Source: G-Research Crypto Forecasting 9th place -- "3 different LightGBM models that were trained for
different market conditions. Up market, down market and relatively more stable market. Then I get the
average of them." Generalizes the "train separate models per condition" pattern (also echoed by the LTFS
finhack2 segment-specific modeling): a single global model trained across regimes with genuinely different
underlying dynamics is forced to compromise, often fitting none of them well; per-regime specialists avoid
that compromise.

Leakage discipline: ``regime_fn`` must be a deterministic function of ``X`` alone (e.g. a rolling trend sign
or volatility bucket computed from PAST data only) -- never derived from the target or from future
information. Both ``fit`` and ``predict`` call the SAME ``regime_fn``, so there is no separate "regime
router" model to train or leak through; this is the direct way to avoid the regime-label-leakage risk the
source idea's own critique calls out.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Literal

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

logger = logging.getLogger(__name__)


class RegimeSplitEnsemble(BaseEstimator, RegressorMixin):
    """One model per detected regime, combined at predict time by routing or averaging.

    Parameters
    ----------
    estimator_factory
        Zero-arg callable returning a fresh unfitted estimator, called once per detected regime plus once
        more for the fallback global model (trained on all rows, used for regimes seen at predict time but
        absent from training).
    regime_fn
        ``callable(X) -> (n,) array`` of regime labels, a deterministic function of ``X`` alone (see module
        docstring's leakage note).
    combine
        ``"route"`` (default): each row is predicted by its OWN regime's model (the fallback global model
        for a regime unseen at fit time). ``"average"``: every row is predicted by ALL regime models and
        averaged (the source technique's literal mechanism -- can help when regime detection is itself noisy
        and a wrong-regime model still carries some signal). ``"blend"``: every row is predicted by a
        confidence-weighted mix of the regime models, per ``regime_proba_fn`` -- smooths the prediction
        surface near regime boundaries where hard routing (``"route"``) causes a discontinuous jump as a
        row crosses the threshold from one regime's model to another's. Requires ``regime_proba_fn``.
    regime_proba_fn
        Only used when ``combine="blend"``. ``callable(X) -> {regime_label: (n,) weight array}``, a
        deterministic function of ``X`` alone (same leakage discipline as ``regime_fn``). Weights need not
        sum to 1 per row -- they are renormalized over the regimes with a fitted model; rows with zero total
        weight over known regimes fall back to the global model.

    Attributes
    ----------
    regime_models_
        ``{regime_label: fitted estimator}``.
    global_model_
        Fallback model fit on all rows, used by ``combine="route"`` for regimes not seen at fit time, and by
        ``combine="blend"`` for rows with zero weight over all known regimes.
    """

    def __init__(
        self,
        estimator_factory: Callable[[], Any],
        regime_fn: Callable[[Any], np.ndarray],
        combine: Literal["route", "average", "blend"] = "route",
        regime_proba_fn: Callable[[Any], dict[Any, np.ndarray]] | None = None,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.regime_fn = regime_fn
        self.combine = combine
        self.regime_proba_fn = regime_proba_fn

    def fit(self, X: Any, y: Any) -> "RegimeSplitEnsemble":
        if self.combine == "blend" and self.regime_proba_fn is None:
            raise ValueError('combine="blend" requires regime_proba_fn')

        y_arr = np.asarray(y, dtype=np.float64)
        regimes = np.asarray(self.regime_fn(X))
        if regimes.shape[0] != y_arr.shape[0]:
            raise ValueError(f"regime_fn returned {regimes.shape[0]} labels, expected {y_arr.shape[0]}")

        self.regime_models_: dict[Any, Any] = {}
        for regime in np.unique(regimes):
            mask = regimes == regime
            model = clone(self.estimator_factory())
            X_regime = X.loc[mask] if hasattr(X, "loc") else np.asarray(X)[mask]
            model.fit(X_regime, y_arr[mask])
            self.regime_models_[regime] = model

        self.global_model_ = clone(self.estimator_factory())
        self.global_model_.fit(X, y_arr)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.combine == "average":
            preds = [np.asarray(model.predict(X), dtype=np.float64) for model in self.regime_models_.values()]
            return np.asarray(np.mean(preds, axis=0))

        if self.combine == "blend":
            return self._predict_blend(X)

        regimes = np.asarray(self.regime_fn(X))
        n = regimes.shape[0]
        out = np.zeros(n, dtype=np.float64)
        unseen_mask = ~np.isin(regimes, list(self.regime_models_))
        if unseen_mask.any():
            X_unseen = X.loc[unseen_mask] if hasattr(X, "loc") else np.asarray(X)[unseen_mask]
            out[unseen_mask] = np.asarray(self.global_model_.predict(X_unseen), dtype=np.float64)
        for regime, model in self.regime_models_.items():
            mask = regimes == regime
            if not mask.any():
                continue
            X_regime = X.loc[mask] if hasattr(X, "loc") else np.asarray(X)[mask]
            out[mask] = np.asarray(model.predict(X_regime), dtype=np.float64)
        return out

    def _predict_blend(self, X: Any) -> np.ndarray:
        assert self.regime_proba_fn is not None  # enforced in fit()
        proba = self.regime_proba_fn(X)
        n = len(X)
        weighted_sum = np.zeros(n, dtype=np.float64)
        weight_total = np.zeros(n, dtype=np.float64)
        for regime, model in self.regime_models_.items():
            weight = np.asarray(proba.get(regime, np.zeros(n)), dtype=np.float64)
            if not weight.any():
                continue
            pred = np.asarray(model.predict(X), dtype=np.float64)
            weighted_sum += weight * pred
            weight_total += weight

        out = np.zeros(n, dtype=np.float64)
        has_weight = weight_total > 0
        out[has_weight] = weighted_sum[has_weight] / weight_total[has_weight]
        no_weight = ~has_weight
        if no_weight.any():
            X_fallback = X.loc[no_weight] if hasattr(X, "loc") else np.asarray(X)[no_weight]
            out[no_weight] = np.asarray(self.global_model_.predict(X_fallback), dtype=np.float64)
        return out


__all__ = ["RegimeSplitEnsemble"]
