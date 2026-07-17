"""GLM-family composite (CompositeGLMEstimator): log-link Poisson / Gamma / Tweedie.

The composite anchors a family-objective GBDT on a cheap base predictor's log-mean
(init_score / base_margin / baseline) so it learns only the multiplicative RESIDUAL.
Biz_value: on a Poisson-generated count target with a dominant LOG-LINEAR base, the
residual-over-base composite beats both the base alone and a plain Poisson GBDT on
Poisson deviance and RMSE. Contract: non-negative predictions, finite deviance,
clear error on an inner with no offset path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import mean_poisson_deviance, mean_squared_error

from mlframe.training.composite import CompositeGLMEstimator

lgb = pytest.importorskip("lightgbm")


def _poisson_log_linear_data(seed: int = 0, n: int = 12000):
    """Counts with a dominant log-linear base + a mild nonlinear residual.

    ``log(mu) = 0.6 + 0.9*z`` is the dominant log-linear driver a linear-log base
    captures; ``+0.5*sign(a*b)`` is a nonlinear (XOR-ish) residual the base is blind
    to but the GBDT can learn. y ~ Poisson(mu).
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, n)
    a = rng.normal(0.0, 1.0, n)
    b = rng.normal(0.0, 1.0, n)
    log_mu = 0.6 + 0.9 * z + 0.5 * np.sign(a * b)
    mu = np.exp(np.clip(log_mu, -5.0, 5.0))
    y = rng.poisson(mu).astype(np.float64)
    X = pd.DataFrame({"z": z, "a": a, "b": b})
    return X, y


class _LogLinearPoissonBase(PoissonRegressor):
    """A log-linear Poisson base (sees only ``z`` would be ideal, but full X is fine
    -- the linear-log fit cannot represent the sign(a*b) residual either way)."""


class TestGLMBizValue:
    def test_residual_over_base_beats_base_and_plain_gbdt(self) -> None:
        X, y = _poisson_log_linear_data()
        tr, te = slice(0, 8000), slice(8000, None)
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]

        # 1) base alone (log-linear Poisson GLM).
        base = PoissonRegressor(max_iter=1000).fit(Xtr, ytr)
        pred_base = np.maximum(base.predict(Xte), 1e-6)

        # 2) plain Poisson GBDT (must re-derive the whole mean from scratch).
        plain = lgb.LGBMRegressor(n_estimators=300, objective="poisson", verbose=-1).fit(Xtr, ytr)
        pred_plain = np.maximum(plain.predict(Xte), 1e-6)

        # 3) composite: residual-over-base.
        comp = CompositeGLMEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=300, verbose=-1),
            family="poisson",
        ).fit(Xtr, ytr)
        pred_comp = np.maximum(comp.predict(Xte), 1e-6)

        dev_base = mean_poisson_deviance(yte, pred_base)
        dev_plain = mean_poisson_deviance(yte, pred_plain)
        dev_comp = mean_poisson_deviance(yte, pred_comp)
        rmse_base = mean_squared_error(yte, pred_base) ** 0.5
        rmse_plain = mean_squared_error(yte, pred_plain) ** 0.5
        rmse_comp = mean_squared_error(yte, pred_comp) ** 0.5

        # Composite must clearly beat the base alone (it learns the residual the
        # log-linear base cannot see) on BOTH deviance and RMSE.
        assert dev_comp <= dev_base * 0.92, f"composite dev {dev_comp:.4f} should beat base {dev_base:.4f}"
        assert rmse_comp <= rmse_base * 0.97, f"composite rmse {rmse_comp:.4f} should beat base {rmse_base:.4f}"
        # And it must at least match a plain Poisson GBDT on deviance (the offset
        # anchors the dominant effect, so it is never worse than re-deriving it).
        assert dev_comp <= dev_plain * 1.01, f"composite dev {dev_comp:.4f} should match/beat plain GBDT {dev_plain:.4f}"


class TestGLMContract:
    def test_predictions_non_negative_and_deviance_finite(self) -> None:
        X, y = _poisson_log_linear_data(n=3000)
        comp = CompositeGLMEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=60, verbose=-1),
            family="poisson",
        ).fit(X, y)
        pred = comp.predict(X)
        assert pred.shape == (len(X),)
        assert np.all(pred >= 0.0), "GLM composite predictions must be non-negative"
        assert np.all(np.isfinite(pred)), "predictions must be finite"
        dev = mean_poisson_deviance(y, np.maximum(pred, 1e-6))
        assert np.isfinite(dev), "Poisson deviance must be finite"

    def test_gamma_family_on_positive_target(self) -> None:
        rng = np.random.default_rng(3)
        n = 3000
        z = rng.normal(0.0, 1.0, n)
        mu = np.exp(0.5 + 0.8 * z)
        y = rng.gamma(shape=2.0, scale=mu / 2.0).astype(np.float64)
        y = np.maximum(y, 1e-3)  # strictly positive for gamma
        X = pd.DataFrame({"z": z, "w": rng.normal(size=n)})
        comp = CompositeGLMEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=80, verbose=-1),
            family="gamma",
        ).fit(X, y)
        pred = comp.predict(X)
        assert np.all(pred > 0.0)
        assert np.all(np.isfinite(pred))

    def test_tweedie_family_zero_inflated(self) -> None:
        rng = np.random.default_rng(7)
        n = 3000
        z = rng.normal(0.0, 1.0, n)
        mu = np.exp(0.3 + 0.7 * z)
        y = rng.poisson(mu).astype(np.float64) * rng.gamma(2.0, 0.5, n)  # zero-inflated positive
        X = pd.DataFrame({"z": z, "w": rng.normal(size=n)})
        comp = CompositeGLMEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=80, verbose=-1),
            family="tweedie",
            tweedie_power=1.3,
        ).fit(X, y)
        pred = comp.predict(X)
        assert np.all(pred >= 0.0)
        assert np.all(np.isfinite(pred))

    def test_default_inner_is_family_matched(self) -> None:
        # base_estimator=None must build a family-matched LightGBM inner.
        X, y = _poisson_log_linear_data(n=2000)
        comp = CompositeGLMEstimator(family="poisson").fit(X, y)
        assert comp.estimator_.get_params()["objective"] == "poisson"
        assert np.all(comp.predict(X) >= 0.0)

    def test_user_inner_objective_coerced_to_family(self) -> None:
        # A LightGBM inner with a Gaussian objective is coerced onto the family one.
        X, y = _poisson_log_linear_data(n=2000)
        comp = CompositeGLMEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=40, objective="regression", verbose=-1),
            family="poisson",
        ).fit(X, y)
        assert comp.estimator_.get_params()["objective"] == "poisson"

    def test_precomputed_base_mean_column_is_stripped(self) -> None:
        X, y = _poisson_log_linear_data(n=4000)
        base = PoissonRegressor(max_iter=1000).fit(X, y)
        X = X.copy()
        X["base_mu"] = np.maximum(base.predict(X), 1e-6)
        comp = CompositeGLMEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=80, verbose=-1),
            family="poisson",
            base_mean_column="base_mu",
        ).fit(X, y)
        assert comp.n_features_in_ == X.shape[1] - 1
        assert np.all(comp.predict(X) >= 0.0)

    def test_unknown_family_raises(self) -> None:
        X, y = _poisson_log_linear_data(n=300)
        with pytest.raises(ValueError, match="unknown family"):
            CompositeGLMEstimator(
                base_estimator=lgb.LGBMRegressor(n_estimators=10, verbose=-1),
                family="binomial",
            ).fit(X, y)

    def test_negative_target_raises(self) -> None:
        X, _ = _poisson_log_linear_data(n=300)
        y = np.full(len(X), -1.0)
        with pytest.raises(ValueError, match="non-negative"):
            CompositeGLMEstimator(
                base_estimator=lgb.LGBMRegressor(n_estimators=10, verbose=-1),
            ).fit(X, y)

    def test_gamma_zero_target_raises(self) -> None:
        X, _ = _poisson_log_linear_data(n=300)
        y = np.zeros(len(X))
        with pytest.raises(ValueError, match="strictly positive"):
            CompositeGLMEstimator(
                base_estimator=lgb.LGBMRegressor(n_estimators=10, verbose=-1),
                family="gamma",
            ).fit(X, y)

    def test_bad_tweedie_power_raises(self) -> None:
        X, y = _poisson_log_linear_data(n=300)
        with pytest.raises(ValueError, match="tweedie_power"):
            CompositeGLMEstimator(
                base_estimator=lgb.LGBMRegressor(n_estimators=10, verbose=-1),
                family="tweedie",
                tweedie_power=2.5,
            ).fit(X, y)

    def test_inner_without_offset_path_rejected(self) -> None:
        # A plain sklearn regressor has no raw-margin/offset path -> clear error.
        X, y = _poisson_log_linear_data(n=600)
        with pytest.raises(NotImplementedError):
            CompositeGLMEstimator(
                base_estimator=LinearRegression(),
                family="poisson",
            ).fit(X, y)

    def test_predict_before_fit_raises(self) -> None:
        from sklearn.exceptions import NotFittedError

        comp = CompositeGLMEstimator(family="poisson")
        with pytest.raises(NotFittedError):
            comp.predict(pd.DataFrame({"z": [0.1, 0.2]}))
