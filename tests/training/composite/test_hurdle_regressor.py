"""Unit tests for ``HurdleRegressor``: the decomposition contract, edge cases, frame formats, and sklearn fit."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from mlframe.training.composite import HurdleRegressor

N = 1500


def _data(n=N, seed=0, zero_frac=0.6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    happens = rng.random(n) < 1.0 / (1.0 + np.exp(-(1.5 * X[:, 0] + np.log((1 - zero_frac) / zero_frac))))
    y = np.where(happens, np.exp(2.0 + 0.7 * X[:, 1] + rng.normal(0, 0.4, n)), 0.0)
    return X, y


def _fast():
    return dict(
        classifier=HistGradientBoostingClassifier(max_iter=40, random_state=0),
        regressor=HistGradientBoostingRegressor(max_iter=40, random_state=0),
    )


# --------------------------------------------------------------------------------------------- the contract


def test_predict_is_event_probability_times_magnitude():
    """``predict`` must equal ``zero + P * (E[y|event] - zero)`` exactly -- the decomposition IS the contract."""
    X, y = _data()
    m = HurdleRegressor(**_fast()).fit(X, y)
    p = m.predict_event_proba(X)
    mag = m.predict_magnitude(X)
    np.testing.assert_allclose(m.predict(X), p * mag)


def test_nonzero_point_mass_is_respected():
    """With ``zero_value=5`` the event is ``y != 5`` and the decomposition shifts to that baseline."""
    X, y = _data()
    y5 = y + 5.0
    m = HurdleRegressor(zero_value=5.0, **_fast()).fit(X, y5)
    np.testing.assert_allclose(m.predict(X), 5.0 + m.predict_event_proba(X) * (m.predict_magnitude(X) - 5.0))
    assert abs(m.event_rate_ - float(np.mean(y5 != 5.0))) < 1e-12


def test_event_probabilities_are_probabilities():
    X, y = _data()
    p = HurdleRegressor(**_fast()).fit(X, y).predict_event_proba(X)
    assert np.all((p >= 0) & (p <= 1))


def test_magnitude_is_learned_on_event_rows_and_is_positive_under_log():
    X, y = _data()
    m = HurdleRegressor(**_fast()).fit(X, y)
    assert m.magnitude_target_ == "log"
    assert np.all(m.predict_magnitude(X) > 0)


def test_insample_smearing_factor_exceeds_one():
    """Duan's factor is ``mean(exp(residual))`` >= 1 by Jensen; exactly 1 would mean no correction happened."""
    X, y = _data()
    assert HurdleRegressor(**_fast()).fit(X, y).smearing_factor_ > 1.0


def test_smearing_none_uses_a_unit_factor():
    X, y = _data()
    assert HurdleRegressor(smearing="none", **_fast()).fit(X, y).smearing_factor_ == 1.0


def test_oof_smearing_is_larger_than_insample():
    """Held-out residuals are not shrunk by the booster's own fit, so the OOF factor must not be smaller."""
    X, y = _data()
    ins = HurdleRegressor(smearing="insample", **_fast()).fit(X, y).smearing_factor_
    oof = HurdleRegressor(smearing="oof", random_state=0, **_fast()).fit(X, y).smearing_factor_
    assert oof >= ins


def test_raw_magnitude_target_applies_no_smearing():
    X, y = _data()
    m = HurdleRegressor(magnitude_target="raw", **_fast()).fit(X, y)
    assert m.magnitude_target_ == "raw" and m.smearing_factor_ == 1.0


# --------------------------------------------------------------------------------------------- edge cases


def test_negative_event_magnitudes_fall_back_to_raw_and_say_so(caplog):
    """``log`` needs every event above the point mass; otherwise fall back rather than fail."""
    X, y = _data()
    y = y.copy()
    y[np.flatnonzero(y != 0)[:5]] *= -1.0
    with caplog.at_level(logging.INFO):
        m = HurdleRegressor(**_fast()).fit(X, y)
    assert m.magnitude_target_ == "raw"
    assert "fitting the magnitude on the raw scale" in caplog.text
    assert np.all(np.isfinite(m.predict(X)))


def test_all_zero_target_predicts_the_point_mass_and_warns(caplog):
    X, _ = _data()
    with caplog.at_level(logging.WARNING):
        m = HurdleRegressor(**_fast()).fit(X, np.zeros(N))
    assert m.classifier_ is None and m.regressor_ is None
    np.testing.assert_array_equal(m.predict(X), np.zeros(N))
    assert "no training row differs from zero_value" in caplog.text


def test_no_zero_rows_needs_no_classifier():
    """Every row an event: P(event) is 1 and the model reduces to the magnitude regressor."""
    X, y = _data()
    y = np.abs(y) + 1.0
    m = HurdleRegressor(**_fast()).fit(X, y)
    assert m.classifier_ is None
    np.testing.assert_array_equal(m.predict_event_proba(X), np.ones(N))
    np.testing.assert_allclose(m.predict(X), m.predict_magnitude(X))


def test_a_single_event_row_uses_its_value_as_the_magnitude():
    X, _ = _data()
    y = np.zeros(N)
    y[7] = 42.0
    m = HurdleRegressor(**_fast()).fit(X, y)
    assert m.regressor_ is None
    np.testing.assert_array_equal(m.predict_magnitude(X), np.full(N, 42.0))


def test_mismatched_lengths_raise():
    X, y = _data()
    with pytest.raises(ValueError, match="rows but y has"):
        HurdleRegressor(**_fast()).fit(X, y[:-1])


def test_non_finite_target_raises():
    X, y = _data()
    y = y.copy()
    y[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        HurdleRegressor(**_fast()).fit(X, y)


@pytest.mark.parametrize("bad", [{"magnitude_target": "sqrt"}, {"smearing": "median"}])
def test_invalid_options_raise(bad):
    X, y = _data()
    with pytest.raises(ValueError):
        HurdleRegressor(**{**_fast(), **bad}).fit(X, y)


# --------------------------------------------------------------------------------------------- formats & sklearn


def test_pandas_polars_and_numpy_inputs_agree():
    """Row subsetting happens in the caller's own format; all three must give identical predictions."""
    X, y = _data()
    cols = [f"f{i}" for i in range(X.shape[1])]
    p_np = HurdleRegressor(**_fast()).fit(X, y).predict(X)
    p_pd = HurdleRegressor(**_fast()).fit(pd.DataFrame(X, columns=cols), y).predict(pd.DataFrame(X, columns=cols))
    Xpl = pl.DataFrame(X, schema=cols)
    p_pl = HurdleRegressor(**_fast()).fit(Xpl, y).predict(Xpl)
    np.testing.assert_allclose(p_pd, p_np)
    np.testing.assert_allclose(p_pl, p_np)


def test_sample_weight_reaches_both_halves():
    """Up-weighting the event rows must move the event probability up."""
    X, y = _data()
    base = HurdleRegressor(**_fast()).fit(X, y).predict_event_proba(X).mean()
    w = np.where(y != 0, 5.0, 1.0)
    weighted = HurdleRegressor(**_fast()).fit(X, y, sample_weight=w).predict_event_proba(X).mean()
    assert weighted > base


def test_works_with_non_booster_components():
    X, y = _data()
    m = HurdleRegressor(classifier=LogisticRegression(max_iter=500), regressor=LinearRegression()).fit(X, y)
    assert np.all(np.isfinite(m.predict(X)))


def test_clone_and_get_params_round_trip():
    m = HurdleRegressor(zero_value=1.0, magnitude_target="raw", smearing="oof", smearing_cv=3, random_state=4)
    c = clone(m)
    assert c.get_params() == m.get_params()


def test_n_features_in_is_recorded():
    X, y = _data()
    assert HurdleRegressor(**_fast()).fit(X, y).n_features_in_ == X.shape[1]
