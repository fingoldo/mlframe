"""Unit + biz_value tests for CompositeSurvivalEstimator (AFT residual-over-base)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.survival import (
    CompositeSurvivalEstimator,
    concordance_index,
)


def _make_aft(n=1500, censor_frac=0.4, seed=0):
    """Synthetic log-normal AFT with a DOMINANT log-linear base + right censoring.

    log(time) = 1.5*z (base) + 0.6*x1 - 0.4*x2 + noise. The base column carries
    the dominant 1.5*z term ALREADY on the log scale; the residual the inner must
    learn is the weaker 0.6*x1 - 0.4*x2 structure.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    base_logtime = 1.5 * z
    log_time = base_logtime + 0.6 * x1 - 0.4 * x2 + rng.normal(scale=0.25, size=n)
    true_time = np.exp(log_time)
    # Right censoring: independent censor times; observed time = min, event flag.
    c = np.exp(rng.normal(loc=np.log(np.median(true_time)) + 0.5, scale=1.0, size=n))
    censored = c < true_time
    # Force overall censoring fraction roughly to target by thresholding.
    obs_time = np.where(censored, c, true_time)
    event = (~censored).astype(int)
    X = pd.DataFrame({"base_logtime": base_logtime, "x1": x1, "x2": x2})
    return X, obs_time, event, true_time


def _inner():
    """Inner."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(max_iter=80, random_state=0)


def test_cindex_finite_and_above_half():
    """Cindex finite and above half."""
    X, t, e, _ = _make_aft(seed=1)
    est = CompositeSurvivalEstimator(
        base_estimator=_inner(),
        base_column="base_logtime",
        censoring="observed_only",
    )
    est.fit(X, t, event=e)
    pred = est.predict(X)
    ci = concordance_index(t, pred, e)
    assert np.isfinite(ci)
    assert ci > 0.5


def test_censoring_handled_no_crash_and_caveat():
    """Censoring handled no crash and caveat."""
    X, t, e, _ = _make_aft(seed=2)
    est = CompositeSurvivalEstimator(
        base_estimator=_inner(),
        base_column="base_logtime",
        censoring="observed_only",
    )
    est.fit(X, t, event=e)
    assert est.n_censored_ > 0
    assert est.censoring_caveat_  # non-empty caveat surfaced
    assert est.censoring_mode_ == "observed_only"


def test_predicted_times_non_negative():
    """Predicted times non negative."""
    X, t, e, _ = _make_aft(seed=3)
    est = CompositeSurvivalEstimator(
        base_estimator=_inner(),
        base_column="base_logtime",
        censoring="observed_only",
    )
    est.fit(X, t, event=e)
    pred = est.predict(X)
    assert np.all(pred >= 0.0)
    assert np.all(np.isfinite(pred))


def test_event_all_ones_reduces_to_plain_aft():
    """Event all ones reduces to plain aft."""
    X, t, e, _ = _make_aft(seed=4)
    e_all = np.ones_like(e)
    est = CompositeSurvivalEstimator(
        base_estimator=_inner(),
        base_column="base_logtime",
        censoring="observed_only",
    )
    est.fit(X, t, event=e_all)
    assert est.n_censored_ == 0
    assert est.censoring_caveat_ == ""  # no caveat when nothing censored
    pred = est.predict(X)
    assert np.all(np.isfinite(pred)) and np.all(pred >= 0.0)


def test_event_validation_rejects_bad_indicator():
    """Event validation rejects bad indicator."""
    X, t, e, _ = _make_aft(seed=5)
    est = CompositeSurvivalEstimator(base_estimator=_inner(), base_column="base_logtime")
    bad = e.copy().astype(float)
    bad[0] = 2.0
    with pytest.raises(ValueError):
        est.fit(X, t, event=bad)


def test_requires_event():
    """Requires event."""
    X, t, _, _ = _make_aft(seed=6)
    est = CompositeSurvivalEstimator(base_estimator=_inner(), base_column="base_logtime")
    with pytest.raises(ValueError):
        est.fit(X, t, event=None)


def test_biz_value_composite_beats_base_only_on_cindex():
    """Residual-over-base composite beats base-only on the C-index over observed events.

    Base-only predictor = exp(base_logtime) (the dominant log-linear prior). The
    composite additionally learns the 0.6*x1 - 0.4*x2 residual, so on the
    held-out C-index it must rank survival ordering strictly better. Measured
    delta ~0.02-0.05; floor at +0.01 to absorb seed noise.
    """
    Xtr, ttr, etr, _ = _make_aft(n=2000, seed=10)
    Xte, tte, ete, _ = _make_aft(n=2000, seed=11)

    est = CompositeSurvivalEstimator(
        base_estimator=_inner(),
        base_column="base_logtime",
        censoring="observed_only",
    )
    est.fit(Xtr, ttr, event=etr)
    comp_pred = est.predict(Xte)
    ci_comp = concordance_index(tte, comp_pred, ete)

    base_pred = np.exp(Xte["base_logtime"].to_numpy())
    ci_base = concordance_index(tte, base_pred, ete)

    assert ci_comp > ci_base + 0.01, f"composite {ci_comp:.4f} vs base {ci_base:.4f}"


class _StubGBSA:
    """Minimal GBSA-like stub: a risk score = linear combo of the 2 features.

    Exposes only ``.predict`` (the method ``_predict_aware_resid_log`` calls), so
    the aware-path centring logic can be tested WITHOUT scikit-survival installed.
    """

    def fit(self, X, y):
        """Fit."""
        return self

    def predict(self, X):
        """Predict."""
        X = np.asarray(X, dtype=np.float64)
        return X[:, 0] * 0.7 - X[:, 1] * 0.3


def test_aware_predict_is_batch_independent(monkeypatch):
    """Aware-path predictions must be PER-ROW deterministic: a row's prediction
    must not change with the rest of its batch.

    Pre-fix ``_predict_aware_resid_log`` centred risk by the PREDICT-batch mean,
    so ``predict([a])`` differed from ``predict([a, b])[0]`` (silent batch-
    composition bug). The frozen train-risk centre makes them identical.
    """
    X, t, e, _ = _make_aft(n=400, seed=7)
    est = CompositeSurvivalEstimator(
        base_estimator=_inner(),
        base_column="base_logtime",
        censoring="observed_only",
    )
    # Bypass the real fit: inject a stub GBSA + frozen train-risk centre and flip
    # the resolved mode to 'aware' so predict() routes through the aware branch.
    est.fit(X, t, event=e)
    stub = _StubGBSA()
    est.inner_ = stub
    est.censoring_mode_ = "aware"
    X_feat = est._drop_base_column(X)
    est._aware_risk_center_ = float(np.mean(stub.predict(est._to_2d_float(X_feat))))

    single = est.predict(X.iloc[[0]])
    pair = est.predict(X.iloc[[0, 1]])
    triple = est.predict(X.iloc[[0, 1, 2]])
    assert single[0] == pytest.approx(pair[0], rel=1e-12, abs=1e-12)
    assert single[0] == pytest.approx(triple[0], rel=1e-12, abs=1e-12)
