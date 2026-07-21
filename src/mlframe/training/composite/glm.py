"""GLM-family composite for NON-GAUSSIAN regression targets (count / positive / Tweedie).

The Gaussian ``CompositeTargetEstimator`` learns a transformed continuous ``y`` and
inverts it. For count (Poisson), strictly-positive (Gamma) and zero-inflated
positive (Tweedie) targets the natural modelling space is the LOG LINK, not an
additive residual on the raw scale. ``CompositeGLMEstimator`` anchors a GBDT on a
cheap base predictor's log-mean (``init_score`` / ``base_margin`` / ``baseline``)
so the inner only learns the RESIDUAL on the link scale:

    log(E[y | x]) = log(base_mean(x)) + inner_raw_margin(x)
    E[y | x]      = base_mean(x) * exp(inner_raw_margin(x))

When one feature (or a calibrated external rate) already explains most of the mean,
feeding its log as the booster's offset lets the inner spend its capacity on the
multiplicative correction instead of re-deriving the dominant log-linear effect --
the count/positive twin of the regression residual-over-lag idea. The inner is fit
with the MATCHING deviance objective (LightGBM ``objective="poisson"/"gamma"/
"tweedie"``), so the gradient it sees is the right one for the family.

Design choices mirroring the other composite wrappers:
- sklearn-compatible (fit / predict / get_params); the base predictor + the inner
  are passed by config, never captured as closures, so clone / pickle stay clean.
- The base predictor is ANY regressor whose ``predict`` returns a positive mean on
  the original scale (a log-linear model, a Poisson GLM, a rate column). Its output
  is floored at a tiny epsilon before the log so a zero base mean cannot send the
  offset to ``-inf``.
- An inner without a per-row offset path (init_score / base_margin / baseline) is
  rejected with a clear error -- the residual contract is undefined without one.
- ``predict`` returns the mean on the ORIGINAL scale (inverse link), never the link
  margin, so downstream metrics (Poisson / Gamma / Tweedie deviance, RMSE) see the
  values they expect.

Out of scope: classification residuals (see ``CompositeClassificationEstimator``)
and continuous-y additive transforms (see ``CompositeTargetEstimator``).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from ._booster_margin import inner_raw_margin

logger = logging.getLogger(__name__)


# Supported GLM families -> the LightGBM objective string that emits the matching
# deviance gradient. All three use the LOG link, so the composite offset is always
# ``log(base_mean)`` regardless of family.
_FAMILY_OBJECTIVE = {
    "poisson": "poisson",
    "gamma": "gamma",
    "tweedie": "tweedie",
}

# Floor applied to the base mean before taking its log, so a zero / negative base
# prediction cannot drive the offset to -inf (which would poison the booster fit).
_BASE_MEAN_FLOOR = 1e-6

# Clip applied to the inner raw margin at predict so ``exp(margin)`` cannot overflow
# float64 (``exp(700)`` is already ~1e304); keeps the returned mean finite on
# adversarial / extrapolating rows.
_MARGIN_CLIP = 80.0


def _is_polars_df(x: Any) -> bool:
    """Return True iff ``x`` is a polars DataFrame; False (never raises) if polars is absent or ``x`` is any other type."""
    try:
        import polars as pl
        return isinstance(x, pl.DataFrame)
    except Exception:
        return False


def _inner_raw_margin(model: Any, X: Any) -> np.ndarray:
    """Raw (link-scale) margin from a fitted GBDT regressor, family-dispatched.

    Returns the inner's prediction in the booster's INTERNAL link space (the value
    that is ADDED to the offset before the inverse link), i.e. the LightGBM
    ``raw_score`` / XGBoost ``output_margin`` / CatBoost ``RawFormulaVal`` output.
    For a log-link objective this is the log-scale residual prediction.
    """
    return inner_raw_margin(
        model, X,
        lgbm_attr="LGBMRegressor", xgb_attr="XGBRegressor", catboost_attr="CatBoostRegressor",
        wrapper_name="CompositeGLMEstimator", keep_2d=False,
    )


def _fit_inner_with_offset(
    model: Any, X: Any, y: np.ndarray, offset: np.ndarray, sample_weight=None,
) -> None:
    """Fit a GBDT regressor with a per-row LOG-scale offset, family-dispatched.

    The offset is ``log(base_mean)``; the booster learns ``y`` with that value
    pinned as the starting margin (init_score / base_margin / baseline), so its
    raw output is the residual on the link scale.
    """
    kw: dict[str, Any] = {}
    if sample_weight is not None:
        kw["sample_weight"] = sample_weight
    cls = type(model).__name__
    if "LGBM" in cls:
        model.fit(X, y, init_score=offset, **kw)
    elif "XGB" in cls:
        model.fit(X, y, base_margin=offset, **kw)
    elif "CatBoost" in cls:
        model.fit(X, y, baseline=offset, **kw)
    else:
        raise NotImplementedError(
            f"CompositeGLMEstimator: inner {cls!r} does not accept a per-row offset "
            "(init_score / base_margin / baseline). Use a LightGBM / XGBoost / "
            "CatBoost regressor as the inner."
        )


def _default_inner(family: str, tweedie_power: float):
    """A LightGBM regressor with the family-matched objective as the inner default.

    Raises a clear ``ImportError`` if LightGBM is absent so the caller knows to
    either install it or pass an explicit ``base_estimator`` of another family.
    """
    try:
        import lightgbm as lgb
    except Exception as exc:  # pragma: no cover - exercised only without lightgbm
        raise ImportError(
            "CompositeGLMEstimator default inner requires lightgbm. Install it " "(`pip install lightgbm`) or pass an explicit base_estimator."
        ) from exc
    objective = _FAMILY_OBJECTIVE[family]
    # n_jobs=-1 (not LightGBM's own unset default) resolves via joblib.cpu_count(only_physical_cores=False)
    # (plain os.cpu_count(), <1ms) instead of the default's only_physical_cores=True path, which shells out
    # to a subprocess on Windows (loky's WMI-based physical-core detector, measured ~2-5s) -- a one-time-
    # per-process tax on the first LightGBM fit. Same "use all cores" intent, just avoids the slow path.
    kw: dict[str, Any] = dict(n_estimators=300, objective=objective, verbose=-1, n_jobs=-1)
    if family == "tweedie":
        kw["tweedie_variance_power"] = float(tweedie_power)
    return lgb.LGBMRegressor(**kw)


def _set_inner_objective(model: Any, family: str, tweedie_power: float) -> None:
    """Force a user-supplied LightGBM inner onto the family-matched objective.

    Mirrors the "matching deviance objective" contract: if the caller passes a
    LightGBM inner with the wrong / default objective, we set the correct one so
    the gradient the booster sees is the family's deviance. XGBoost / CatBoost
    inners are left untouched (the caller controls their objective explicitly);
    they still get the log-scale offset.
    """
    try:
        import lightgbm as lgb
    except Exception:
        return
    if not isinstance(model, lgb.LGBMRegressor):
        return
    params: dict[str, Any] = {"objective": _FAMILY_OBJECTIVE[family]}
    if family == "tweedie":
        params["tweedie_variance_power"] = float(tweedie_power)
    try:
        model.set_params(**params)
    except (TypeError, ValueError, KeyError) as e:
        # set_params rejected a family-matched kwarg (e.g. a LightGBM version that renamed
        # tweedie_variance_power, or a caller subclass with a narrower signature) -- the inner
        # keeps its prior objective, which may now mismatch the requested family's deviance.
        logger.warning(
            "CompositeGLMEstimator could not set the family-matched LightGBM objective %r; "
            "the inner keeps its current objective, which may not match the '%s' deviance: %s",
            params,
            family,
            e,
        )


class CompositeGLMEstimator(BaseEstimator, RegressorMixin):
    """Log-link GLM composite that learns a residual over a base mean predictor.

    For count (Poisson), strictly-positive (Gamma) and zero-inflated-positive
    (Tweedie) targets. The inner GBDT is fit with the matching deviance objective
    and a per-row log-mean offset from a cheap base predictor; ``predict`` returns
    the mean on the original scale (inverse log link of ``log(base_mean) +
    inner_margin``).

    Parameters
    ----------
    base_estimator
        The inner gradient-boosting regressor (LightGBM / XGBoost / CatBoost) that
        learns the link-scale residual. Cloned at fit. ``None`` (default) builds a
        family-matched ``LGBMRegressor``. A LightGBM inner with a mismatched
        objective is coerced onto the family objective; XGBoost / CatBoost inners
        keep their caller-set objective (only the offset is supplied).
    base_mean_estimator
        A cheap regressor whose ``predict`` returns a POSITIVE mean on the original
        scale (a log-linear model, a Poisson GLM, etc.). Cloned at fit. Default:
        ``sklearn.linear_model.GammaRegressor`` for ``gamma`` (strictly positive),
        ``PoissonRegressor`` for ``poisson`` / ``tweedie`` (both tolerate zeros).
    base_mean_column
        Alternatively, the name of a column in X holding a PRECOMPUTED positive base
        mean (e.g. an external exposure / rate). When set, ``base_mean_estimator`` is
        ignored and the column is stripped from X before fitting the inner.
    family
        One of ``"poisson"``, ``"gamma"``, ``"tweedie"``. Selects the inner deviance
        objective. All use the log link.
    tweedie_power
        Tweedie variance power ``p`` in ``(1, 2)`` (only used for ``family="tweedie"``).
        ``p->1`` approaches Poisson, ``p->2`` approaches Gamma; ``1.5`` is a common
        default for zero-inflated positive targets.
    """

    def __init__(
        self,
        base_estimator: Any = None,
        base_mean_estimator: Any = None,
        base_mean_column: str | None = None,
        family: str = "poisson",
        tweedie_power: float = 1.5,
    ) -> None:
        self.base_estimator = base_estimator
        self.base_mean_estimator = base_mean_estimator
        self.base_mean_column = base_mean_column
        self.family = family
        self.tweedie_power = tweedie_power

    # -- base mean extraction ---------------------------------------------------
    def _default_base_mean_estimator(self):
        """Family-matched default base-mean regressor: GammaRegressor for strictly-positive 'gamma', PoissonRegressor otherwise (tolerates zeros)."""
        from sklearn.linear_model import GammaRegressor, PoissonRegressor
        # GammaRegressor rejects zero / non-positive y, so only 'gamma' (strictly
        # positive target) uses it; 'poisson' and 'tweedie' (zero-inflated) use the
        # Poisson GLM base, which is a valid positive-mean log-linear fit for both.
        if self.family == "gamma":
            return GammaRegressor(max_iter=1000)
        return PoissonRegressor(max_iter=1000)

    def _base_mean_from_estimator(self, est: Any, X: Any) -> np.ndarray:
        """Positive base mean on the original scale, floored away from zero."""
        mean = np.asarray(est.predict(X), dtype=np.float64).reshape(-1)
        return np.maximum(mean, _BASE_MEAN_FLOOR)

    def _extract_mean_column(self, X: Any) -> np.ndarray:
        """Pull the precomputed base-mean column out of X (polars or pandas) as a floored positive ndarray."""
        col = self.base_mean_column
        if hasattr(X, "get_column"):  # polars
            arr = np.asarray(X.get_column(col).to_numpy(), dtype=np.float64).reshape(-1)
        else:
            arr = np.asarray(X[col].to_numpy(), dtype=np.float64).reshape(-1)
        return np.maximum(arr, _BASE_MEAN_FLOOR)

    def _drop_mean_column(self, X: Any) -> Any:
        """Strip ``base_mean_column`` from X (polars or pandas) before it is passed to the inner estimator."""
        col = self.base_mean_column
        if hasattr(X, "drop") and hasattr(X, "get_column"):  # polars
            return X.drop(col) if col in X.columns else X
        if hasattr(X, "drop"):  # pandas
            return X.drop(columns=[col]) if col in X.columns else X
        return X

    # -- sklearn API ------------------------------------------------------------
    def fit(self, X: Any, y: Any, sample_weight=None) -> "CompositeGLMEstimator":
        """Validate the family/target, derive the log-mean offset from the base predictor (or offset column), and fit the inner GBDT on the residual with the matching deviance objective."""
        if self.family not in _FAMILY_OBJECTIVE:
            raise ValueError(f"CompositeGLMEstimator: unknown family {self.family!r}; " f"choose one of {sorted(_FAMILY_OBJECTIVE)}.")
        if self.family == "tweedie" and not (1.0 < float(self.tweedie_power) < 2.0):
            raise ValueError("CompositeGLMEstimator: tweedie_power must be strictly between 1 and 2; " f"got {self.tweedie_power!r}.")
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if np.any(y_arr < 0):
            raise ValueError(f"CompositeGLMEstimator: family {self.family!r} targets must be " "non-negative (count / positive); found negative y.")
        if self.family == "gamma" and np.any(y_arr <= 0):
            raise ValueError(
                "CompositeGLMEstimator: family 'gamma' targets must be strictly "
                "positive; found a zero / non-positive y (use 'tweedie' for "
                "zero-inflated positive targets)."
            )

        if self.base_mean_column is not None:
            base_mean = self._extract_mean_column(X)
            X_inner = self._drop_mean_column(X)
            self.base_mean_estimator_ = None
        else:
            self.base_mean_estimator_ = clone(self.base_mean_estimator if self.base_mean_estimator is not None else self._default_base_mean_estimator())
            if sample_weight is not None:
                self.base_mean_estimator_.fit(X, y_arr, sample_weight=sample_weight)
            else:
                self.base_mean_estimator_.fit(X, y_arr)
            base_mean = self._base_mean_from_estimator(self.base_mean_estimator_, X)
            X_inner = X

        offset = np.log(base_mean)

        if self.base_estimator is None:
            self.estimator_ = _default_inner(self.family, self.tweedie_power)
        else:
            self.estimator_ = clone(self.base_estimator)
            _set_inner_objective(self.estimator_, self.family, self.tweedie_power)
        _fit_inner_with_offset(self.estimator_, X_inner, y_arr, offset, sample_weight)
        self.n_features_in_ = X_inner.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Mean on the ORIGINAL scale: ``inverse_link(log(base_mean) + inner_margin)``.

        For the log link this is ``base_mean * exp(inner_margin)``. Always
        non-negative and finite (the link total is clipped before ``exp``).
        """
        if not hasattr(self, "estimator_"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError("CompositeGLMEstimator.predict called before fit.")
        if self.base_mean_column is not None:
            base_mean = self._extract_mean_column(X)
            X_inner = self._drop_mean_column(X)
        else:
            base_mean = self._base_mean_from_estimator(self.base_mean_estimator_, X)
            X_inner = X
        margin = np.clip(_inner_raw_margin(self.estimator_, X_inner), -_MARGIN_CLIP, _MARGIN_CLIP)
        out = base_mean * np.exp(margin)
        return np.asarray(np.maximum(out, 0.0))


# Variance-scaled split-conformal prediction intervals. Bound from
# ``composite/conformal_glm.py`` (which imports nothing from glm, so no cycle).
# calibrate_conformal_glm(X_cal, y_cal, alpha) fits the standardized radius from a
# held-out set; predict_interval_glm(X, alpha) returns a heteroscedastic band whose
# width scales with sqrt(V(mu_hat)) and is clipped non-negative.
from .conformal_glm import (
    calibrate_conformal_glm as _calibrate_conformal_glm,
    predict_interval_glm as _predict_interval_glm,
)

CompositeGLMEstimator.calibrate_conformal_glm = _calibrate_conformal_glm
CompositeGLMEstimator.predict_interval_glm = _predict_interval_glm
