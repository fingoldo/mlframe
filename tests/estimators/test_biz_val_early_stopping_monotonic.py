"""biz_value: the monotonic strict-decline stop in EarlyStoppingWrapper.

Quantitatively pins the win of ``monotonic_decline_patience`` (default-on):

  1. overfit-prone target -> the monotonic stop ends training STRICTLY earlier than a
     patience-only run with the same large patience, WITHOUT hurting held-out RMSE.
  2. cleanly-improving target -> the detector does NOT fire prematurely (it lets the model
     train to a near-optimal stage; it must not truncate a still-improving curve).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

from mlframe.estimators.early_stopping import EarlyStoppingWrapper


def _rmse(yt, yp):
    return float(np.sqrt(mean_squared_error(yt, yp)))


def _overfit_data(seed=3, n=300, d=6):
    """Tiny signal buried in noise: a GradientBoosting curve peaks early then overfits."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = 0.6 * X[:, 0] + rng.randn(n) * 1.5  # mostly noise -> late stages overfit train
    return X, y


def _clean_data(seed=5, n=300, d=6):
    """Strong learnable signal: more boosting stages keep helping val (no early overfit)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2] + rng.randn(n) * 0.1
    return X, y


def _fit_sgd(X, y, *, monotonic, patience, eta0, max_iter=120):
    """partial_fit backend: each iteration is a REAL fit, so a stop saves genuine compute."""
    from sklearn.linear_model import SGDRegressor

    es = EarlyStoppingWrapper(
        SGDRegressor(max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=eta0),
        max_iter=max_iter, validation_fraction=0.2,
        patience=patience, monotonic_decline_patience=monotonic,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        es.fit(X, y)
    return es


def test_biz_val_monotonic_stops_earlier_without_holdout_loss():
    # Aggressive learning rate on a mostly-noise target -> the val curve peaks then monotonically
    # diverges, exactly the confident-overfitting shape the monotonic detector is built to catch.
    X, y = _overfit_data()
    n_val = max(1, int(len(X) * 0.2))
    Xv, yv = X[-n_val:], y[-n_val:]

    # Large patience so the patience counter essentially never trips -> isolates the monotonic effect.
    es_mono = _fit_sgd(X, y, monotonic=3, patience=999, eta0=0.3)
    es_none = _fit_sgd(X, y, monotonic=None, patience=999, eta0=0.3)

    # Work-saving: the monotonic stop ends training STRICTLY earlier than the patience-only run.
    assert es_mono.n_iterations_ < es_none.n_iterations_, (
        f"monotonic stop saved no iterations: {es_mono.n_iterations_} vs {es_none.n_iterations_}"
    )
    # No holdout loss: the restored global-best is at least as good as the no-stop run's best.
    rmse_mono = _rmse(yv, es_mono.predict(Xv))
    rmse_none = _rmse(yv, es_none.predict(Xv))
    assert rmse_mono <= rmse_none + 1e-9, f"monotonic stop hurt holdout RMSE: {rmse_mono:.4f} vs {rmse_none:.4f}"


def test_biz_val_monotonic_does_not_fire_on_clean_target():
    """On a cleanly-improving target the detector must not prematurely truncate the curve."""
    X, y = _clean_data()
    n_val = max(1, int(len(X) * 0.2))
    Xv, yv = X[-n_val:], y[-n_val:]

    # Gentle learning rate on a strong signal -> the val curve improves smoothly; no 3-strict-decline run.
    es_mono = _fit_sgd(X, y, monotonic=3, patience=999, eta0=0.005, max_iter=80)
    es_none = _fit_sgd(X, y, monotonic=None, patience=999, eta0=0.005, max_iter=80)

    # No-harm: holdout RMSE with the default-on detector is essentially the same as without it.
    rmse_mono = _rmse(yv, es_mono.predict(Xv))
    rmse_none = _rmse(yv, es_none.predict(Xv))
    assert rmse_mono <= rmse_none + 0.05, f"monotonic detector fired prematurely: {rmse_mono:.4f} vs {rmse_none:.4f}"
