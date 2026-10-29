"""The binary probe of the FE accuracy gate fits its logistic model with a Newton iteration; it must agree with the scikit-learn fit and defer to it when it does not converge."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mlframe.feature_selection.filters import _fe_accuracy_gate_newton as newton
from mlframe.feature_selection.filters._fe_accuracy_gate import keep_engineered_over_source, measure_feature_uplift
from mlframe.metrics.core import fast_roc_auc


def _design(seed: int, n: int = 1500):
    """A collinear four-column design (s, s^2, s^3, sin s) and a noisy binary label that depends on s."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    X = np.column_stack([s, s**2, s**3, np.sin(s)]).astype(np.float32)
    y = ((1.2 * s + 0.4 * rng.normal(size=n)) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_newton_held_out_auc_matches_scikit_learn(seed):
    """Same held-out ROC-AUC to 1e-3 on a collinear standardised design."""
    X, y = _design(seed)
    cut = int(0.9 * len(y))
    scaler = StandardScaler()
    Xt, Xv = scaler.fit_transform(X[:cut]), scaler.transform(X[cut:])
    fit = newton.fit_logistic_ridge_newton(Xt, y[:cut], 1.0)
    assert fit is not None
    w, b = fit
    ref = LogisticRegression(max_iter=200, C=1.0).fit(Xt, y[:cut])
    auc_newton = fast_roc_auc(y[cut:], 1.0 / (1.0 + np.exp(-(Xv.astype(np.float64) @ w + b))))
    auc_ref = fast_roc_auc(y[cut:], ref.predict_proba(Xv)[:, 1])
    assert abs(auc_newton - auc_ref) < 1e-3


def test_newton_returns_none_when_it_does_not_converge(monkeypatch):
    """With one iteration allowed the fit is not at the optimum, so it declines and the caller keeps the scikit-learn fit."""
    X, y = _design(0)
    monkeypatch.setattr(newton, "_MAX_ITER", 1)
    assert newton.fit_logistic_ridge_newton(StandardScaler().fit_transform(X), y, 1.0) is None


@pytest.mark.parametrize("quadratic", [0.0, 0.8])
def test_gate_decision_is_the_same_with_and_without_the_newton_fit(monkeypatch, quadratic):
    """The keep/drop decision of the accuracy gate does not depend on which solver fitted the probe."""
    rng = np.random.default_rng(7)
    n = 3000
    s = rng.normal(size=n)
    y = ((s + quadratic * s**2 + rng.normal(size=n)) > 0).astype(int)
    eng = np.column_stack([s**2, s**3, np.sin(s)])
    monkeypatch.setenv("MLFRAME_GATE_NEWTON", "1")
    fast = keep_engineered_over_source(s, eng, y, seed=0)
    up_fast = measure_feature_uplift(s, eng, y, classification=True, seed=0)
    monkeypatch.setenv("MLFRAME_GATE_NEWTON", "0")
    exact = keep_engineered_over_source(s, eng, y, seed=0)
    up_exact = measure_feature_uplift(s, eng, y, classification=True, seed=0)
    assert fast == exact
    assert abs(up_fast - up_exact) < 1e-3
