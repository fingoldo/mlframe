"""``HurdleRegressor``: model a zero-inflated target as P(event) x E[size | event].

Why this exists
---------------
Many business targets are a point mass at zero plus a heavy-tailed positive part: an amount that is 0 when nothing
happened (no hire, no purchase, no claim) and a skewed magnitude when something did. A production run modelled one
such target -- ``target_total_charge``, median 0, ~71% of rows exactly zero, excess kurtosis 2415 -- with a single
regressor and with log / cbrt composite targets. The composites COLLAPSED:

    [regression-collapse-sensor:std-collapse] ... target_total_charge-logY  pred_std=6.44 (1.8% of target_std=349)

A log transform over a target that is mostly an exact zero maps the whole mass onto one point, the model predicts
near it for most rows, and the convex inverse squeezes the spread away. The same run trained the matching event
target (``target_total_hired_above_1``, test ROC AUC 0.81) separately -- the first half of a hurdle model was already
there, unused.

The decomposition
-----------------
With ``e = (y != zero_value)``::

    E[y | x] = zero_value + P(e | x) * (E[y | x, e] - zero_value)

A classifier learns ``P(e | x)`` on every row; a regressor learns ``E[y | x, e]`` on the event rows only, where there
is no point mass, so a log target is well-posed there. Each half is a smoother function than the discontinuous whole.

Measured on a synthetic reproducing the production shape (71% zeros, event and magnitude driven by DIFFERENT
features, log-normal magnitudes of increasing tail weight sigma; 20k train rows, 5 seeds):

=========  ============  ===============  ================
  sigma    single log1p   single raw-MSE   hurdle (default)
=========  ============  ===============  ================
   0.6      R2 0.130       R2 0.318         R2 0.343
   1.2      R2 0.019       R2 0.084         R2 0.096
   1.8      R2 -0.003      R2 0.002         R2 0.015
=========  ============  ===============  ================

At sigma = 1.8 the single log1p model's prediction spread is 1.8% of the target's -- the production figure exactly --
while the hurdle's is 12.1%. Against a single raw-MSE regressor the hurdle's edge is real but modest (+8..14% relative
R2 on moderate tails) and shrinks as n grows, because a flexible booster eventually learns E[y|x] directly. The
reliable win is over the transform-based composites, whose failure it removes.

Magnitude target and back-transform correction
----------------------------------------------
``magnitude_target="log"`` (default) fits the magnitude model on ``log(y - zero_value)`` and inverts with a smearing
factor, because ``exp(E[log y]) < E[y]``. It beat ``"raw"`` in all 8 measured cells, by a margin growing with tail
weight (+0.01 R2 at sigma 0.3, +0.08 at sigma 1.8), and ``"raw"`` went NEGATIVE (R2 -0.06) at sigma 1.8. ``"log"``
needs every event magnitude above ``zero_value``; otherwise the fit falls back to ``"raw"`` and says so.

``smearing`` picks the factor, and no single choice wins both objectives a caller might hold:

=========  ======================  =====================  ======================
  sigma     "none"  R2 / total      "insample" (default)   "oof"
=========  ======================  =====================  ======================
   0.3      0.4700 / -5.0%          0.4715 / -2.7%         0.4725 / -0.0%
   0.6      0.3342 / -16.6%         0.3433 / -8.1%         0.3459 / +1.9%
   1.2      0.0764 / -49.4%         0.0958 / -25.5%        0.0852 / +11.6%
   1.8      0.0061 / -77.8%         0.0150 / -46.6%        -0.0166 / +29.0%
=========  ======================  =====================  ======================

"total" is ``sum(pred) / sum(y) - 1``: the bias of the predicted TOTAL. ``"insample"`` (Duan's estimator on the
training residuals) is best-or-within-noise on per-row R2 everywhere and never negative, and costs no extra fits --
it is the default because R2 / RMSE is what the suite reports and ranks by. Its in-sample residuals are shrunk by
the booster's own fit, so it UNDER-states the true factor and the predicted total runs low on heavy tails. When the
aggregate matters more than the per-row error (forecasting total revenue across jobs), use ``"oof"``: K-fold
held-out residuals estimate the factor correctly, which makes light-tailed totals essentially unbiased, at the cost
of ``smearing_cv`` extra magnitude fits and worse per-row R2 on very heavy tails. ``"none"`` loses on both and exists
only for comparison.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

__all__ = ["HurdleRegressor"]

MagnitudeTarget = Literal["log", "raw"]
Smearing = Literal["insample", "oof", "none"]


def _take_rows(X: Any, mask: np.ndarray) -> Any:
    """``X`` restricted to the rows where ``mask`` is True, in X's own format (pandas / polars / array).

    The format is the caller's decision; subsetting in place keeps it, so a polars frame reaches a polars-native
    estimator as polars and never takes a silent pandas round-trip here.
    """
    if hasattr(X, "iloc"):
        return X.iloc[np.flatnonzero(mask)]
    if hasattr(X, "filter") and hasattr(X, "schema"):
        import polars as pl

        return X.filter(pl.Series(mask))
    return np.asarray(X)[mask]


from mlframe.utils.frame_rows import n_rows as _n_rows


def _default_classifier() -> Any:
    """The event classifier used when none is given: a default HistGradientBoostingClassifier."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier()


def _default_regressor() -> Any:
    """The event-only regressor used when none is given: a default HistGradientBoostingRegressor."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor()


class HurdleRegressor(RegressorMixin, BaseEstimator):
    """Predict ``zero_value + P(event) * (E[y | event] - zero_value)`` from a classifier and an event-only regressor.

    Parameters
    ----------
    classifier
        sklearn-compatible classifier prototype with ``predict_proba``, fit on EVERY row against ``y != zero_value``.
        ``None`` uses ``HistGradientBoostingClassifier`` (native NaN handling, no extra dependency).
    regressor
        sklearn-compatible regressor prototype fit on the EVENT rows only. ``None`` uses
        ``HistGradientBoostingRegressor``.
    zero_value
        The point mass that means "no event". Default ``0.0``.
    magnitude_target
        ``"log"`` (default) fits the magnitude on ``log(y - zero_value)`` and inverts with ``smearing``; ``"raw"``
        fits it on ``y`` directly. See the module docstring for the measurements behind the default.
    smearing
        Back-transform correction for ``magnitude_target="log"``: ``"insample"`` (default), ``"oof"`` or ``"none"``.
        See the module docstring -- this is a per-row-error versus aggregate-bias trade-off, not a free choice.
    smearing_cv
        Folds for ``smearing="oof"``.
    random_state
        Seed for the ``smearing="oof"`` fold shuffle.
    below_zero
        What a row below ``zero_value`` is: ``"event"`` (default) keeps it an event, which forces
        ``magnitude_target="log"`` down to ``"raw"``; ``"no_event"`` counts it as the point mass. The zero-inflation
        dispatch uses ``"no_event"`` for a target with a sliver below the atom (refunds of a few cents under a 74% point
        mass at 0), so a handful of rows neither blocks the hurdle nor costs the magnitude model its log scale.

    Attributes
    ----------
    classifier_, regressor_
        Fitted clones; ``None`` when the training data made that half unnecessary (no events, or no non-events).
    magnitude_target_
        The magnitude target actually used -- ``"raw"`` when ``"log"`` was requested but some event magnitude was not
        above ``zero_value``.
    smearing_factor_
        Multiplier applied to ``exp(prediction)`` under ``magnitude_target_="log"``; ``1.0`` otherwise.
    event_rate_
        Fraction of training rows that were events.
    """

    def __init__(
        self,
        classifier: Any = None,
        regressor: Any = None,
        zero_value: float = 0.0,
        magnitude_target: MagnitudeTarget = "log",
        smearing: Smearing = "insample",
        smearing_cv: int = 5,
        random_state: Optional[int] = None,
        below_zero: str = "event",
    ) -> None:
        self.classifier = classifier
        self.regressor = regressor
        self.zero_value = zero_value
        self.magnitude_target = magnitude_target
        self.smearing = smearing
        self.smearing_cv = smearing_cv
        self.random_state = random_state
        self.below_zero = below_zero

    # ------------------------------------------------------------------------------------------------ fit
    def fit(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> "HurdleRegressor":
        """Fit the event classifier on every row and the magnitude regressor on the event rows."""
        if self.magnitude_target not in ("log", "raw"):
            raise ValueError(f"magnitude_target must be 'log' or 'raw', got {self.magnitude_target!r}.")
        if self.smearing not in ("insample", "oof", "none"):
            raise ValueError(f"smearing must be 'insample', 'oof' or 'none', got {self.smearing!r}.")
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if y_arr.size != _n_rows(X):
            raise ValueError(f"HurdleRegressor.fit: X has {_n_rows(X)} rows but y has {y_arr.size}.")
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("HurdleRegressor.fit: y contains non-finite values.")
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).reshape(-1)

        if self.below_zero not in ("event", "no_event"):
            raise ValueError(f"below_zero must be 'event' or 'no_event', got {self.below_zero!r}.")
        z = float(self.zero_value)
        event = (y_arr > z) if self.below_zero == "no_event" else (y_arr != z)
        _shape = getattr(X, "shape", None)
        if _shape is not None and len(_shape) == 2:
            self.n_features_in_ = int(_shape[1])
        self.event_rate_ = float(event.mean()) if y_arr.size else 0.0
        self.classifier_ = None
        self.regressor_ = None
        self.magnitude_target_ = self.magnitude_target
        self.smearing_factor_ = 1.0
        self.constant_magnitude_ = None

        n_event = int(event.sum())
        if n_event == 0:
            # Nothing ever happened in training: every prediction is the point mass, and there is no magnitude
            # to learn. Said aloud because a hurdle over a target with no events is almost always a data mistake.
            logger.warning("[hurdle] no training row differs from zero_value=%s; every prediction will be %s.", z, z)
            return self

        if 0 < n_event < y_arr.size:
            self.classifier_ = clone(self.classifier if self.classifier is not None else _default_classifier())
            fit_kw = {"sample_weight": w} if w is not None else {}
            self.classifier_.fit(X, event.astype(np.int64), **fit_kw)

        X_ev = _take_rows(X, event)
        y_ev = y_arr[event]
        w_ev = None if w is None else w[event]

        if self.magnitude_target_ == "log" and np.any(y_ev <= z):
            # Negative magnitudes relative to the point mass have no log: fall back rather than fail, and say so.
            logger.info(
                "[hurdle] magnitude_target='log' needs every event magnitude above zero_value=%s, but %d of %d are "
                "not; fitting the magnitude on the raw scale instead.",
                z, int(np.sum(y_ev <= z)), n_event,
            )
            self.magnitude_target_ = "raw"

        if n_event < 2:
            # One event cannot train a regressor; its value is the only estimate of E[y | event] there is.
            self.constant_magnitude_ = float(y_ev[0])
            return self

        target = np.log(y_ev - z) if self.magnitude_target_ == "log" else y_ev
        self.regressor_ = clone(self.regressor if self.regressor is not None else _default_regressor())
        reg_kw = {"sample_weight": w_ev} if w_ev is not None else {}
        self.regressor_.fit(X_ev, target, **reg_kw)

        if self.magnitude_target_ == "log" and self.smearing != "none":
            self.smearing_factor_ = self._smearing_factor(X_ev, target, w_ev)
        return self

    def _smearing_factor(self, X_ev: Any, log_target: np.ndarray, w_ev: Optional[np.ndarray]) -> float:
        """Duan's smearing factor ``mean(exp(residual))`` from in-sample or K-fold held-out log residuals."""
        if self.smearing == "insample":
            regressor = self.regressor_
            assert regressor is not None, "in-sample smearing runs only after the magnitude regressor is fit"
            resid = log_target - np.asarray(regressor.predict(X_ev), dtype=np.float64).reshape(-1)
        else:
            n = log_target.size
            k = max(2, min(int(self.smearing_cv), n))
            held_out = np.empty(n, dtype=np.float64)
            for tr, _va in KFold(k, shuffle=True, random_state=self.random_state).split(np.arange(n)):
                tr_mask = np.zeros(n, dtype=bool)
                tr_mask[tr] = True
                fold = clone(self.regressor if self.regressor is not None else _default_regressor())
                kw = {"sample_weight": w_ev[tr]} if w_ev is not None else {}
                fold.fit(_take_rows(X_ev, tr_mask), log_target[tr_mask], **kw)
                held_out[~tr_mask] = np.asarray(fold.predict(_take_rows(X_ev, ~tr_mask)), dtype=np.float64)
            resid = log_target - held_out
        resid = resid[np.isfinite(resid)]
        return float(np.mean(np.exp(resid))) if resid.size else 1.0

    # --------------------------------------------------------------------------------------------- predict
    def predict_event_proba(self, X: Any) -> np.ndarray:
        """``P(y != zero_value | x)`` per row."""
        n = _n_rows(X)
        if self.classifier_ is None:
            # No classifier was needed: every training row was an event (rate 1) or none was (rate 0).
            return np.full(n, 1.0 if self.event_rate_ > 0 else 0.0)
        proba = np.asarray(self.classifier_.predict_proba(X), dtype=np.float64)
        classes = list(getattr(self.classifier_, "classes_", [0, 1]))
        return proba[:, classes.index(1)] if 1 in classes else np.zeros(n)

    def predict_magnitude(self, X: Any) -> np.ndarray:
        """``E[y | x, event]`` per row -- the expected size given that the event happens."""
        z = float(self.zero_value)
        n = _n_rows(X)
        if self.regressor_ is None:
            return np.full(n, self.constant_magnitude_ if self.constant_magnitude_ is not None else z)
        raw = np.asarray(self.regressor_.predict(X), dtype=np.float64).reshape(-1)
        if self.magnitude_target_ == "log":
            return np.asarray(z + np.exp(raw) * self.smearing_factor_, dtype=np.float64)
        return raw

    def predict(self, X: Any) -> np.ndarray:
        """``zero_value + P(event) * (E[y | event] - zero_value)``: the conditional mean of y."""
        z = float(self.zero_value)
        if self.event_rate_ == 0.0:
            return np.full(_n_rows(X), z)
        return np.asarray(z + self.predict_event_proba(X) * (self.predict_magnitude(X) - z), dtype=np.float64)
