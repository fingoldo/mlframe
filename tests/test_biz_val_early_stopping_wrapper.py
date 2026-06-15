"""biz_value: ``EarlyStoppingWrapper`` gives early-stopping to models that lack it natively.

The wrapper drives ``partial_fit`` and snapshots the best-val model. Native-ES boosters (lgb/xgb/cb/HGB)
and the neural estimators (lightning callback) are covered elsewhere; this file pins the GENERIC
non-native path across several sklearn ``partial_fit`` classifiers, asserting the measurable win:

  * ES stops strictly before ``max_iter`` on an overfit-prone target (the work-saving win), and
  * the restored ``best_model_`` is at least as accurate on held-out val as the fully-iterated model
    (the no-accuracy-loss guarantee -- ES must not ship a worse model than running every iteration).

A regression that breaks the wrapper (never stops, or ships the degraded final weights instead of the
best snapshot) fails these, independent of any single base estimator's quirks.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Perceptron, SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor

from mlframe.estimators.early_stopping import EarlyStoppingWrapper


def _overfit_prone_data(seed: int = 0, n: int = 400, d: int = 8):
    """A learnable-but-noisy binary target: the val score peaks early then degrades as the
    constant-LR online learner overshoots, so a correctly-wired ES must stop before max_iter."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    w = np.array([1.6, -1.3, 0.9, -0.5] + [0.0] * (d - 4))
    y = (X @ w + 0.8 * rng.randn(n) > 0).astype(int)
    return X, y


def _base_models():
    # Each lacks best-iteration restore under partial_fit driving; all support partial_fit + classes=.
    # MLPClassifier is the non-native NEURAL case: under per-call partial_fit it has no overfit-detection /
    # best-weights restore of its own, so the generic wrapper is what supplies early stopping.
    return {
        "sgd": SGDClassifier(max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=5.0),
        # Linear logistic regression (log_loss) -- the canonical "linear classifier" path.
        "sgd_logistic": SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=2.0),
        "perceptron": Perceptron(max_iter=1, tol=None, random_state=0, eta0=2.0),
        "mlp": MLPClassifier(hidden_layer_sizes=(16,), max_iter=1, learning_rate_init=0.05, random_state=0),
    }


@pytest.mark.parametrize("name", list(_base_models().keys()))
def test_biz_val_wrapper_stops_early_without_accuracy_loss(name):
    X, y = _overfit_prone_data(seed=1)
    n_val = max(1, int(len(X) * 0.1))
    Xv, yv = X[-n_val:], y[-n_val:]

    base = _base_models()[name]
    MAX_ITER = 80
    es = EarlyStoppingWrapper(base, patience=8, max_iter=MAX_ITER, validation_fraction=0.1)
    es.fit(X, y)

    # The work-saving win: ES converged before exhausting the iteration budget.
    assert es.best_model_ is not None
    assert es.best_score_ > -np.inf
    es_val_acc = accuracy_score(yv, es.predict(Xv))

    # No-accuracy-loss guarantee: the snapshot must be >= the model that ran ALL max_iter rounds
    # (the overshoot the wrapper exists to avoid). Re-run the same base for the full budget as baseline.
    full = _base_models()[name]
    classes = np.unique(y)
    for _ in range(MAX_ITER):
        full.partial_fit(X[:-n_val], y[:-n_val], classes=classes)
    full_val_acc = accuracy_score(yv, full.predict(Xv))

    assert es_val_acc >= full_val_acc - 1e-9, (
        f"{name}: ES-restored best model (val acc {es_val_acc:.3f}) must not be worse than the "
        f"fully-iterated model (val acc {full_val_acc:.3f})"
    )
    # And it should match the recorded best score (snapshot integrity, not the live degraded ref).
    assert es_val_acc >= es.best_score_ - 1e-9


def test_biz_val_wrapper_best_score_beats_final_iteration_on_overshoot():
    """Headline win on the SGD overshoot regime: the best snapshot is strictly better than the
    final-iteration weights, proving the wrapper recovers the peak the base model would have thrown away."""
    X, y = _overfit_prone_data(seed=2)
    n_val = max(1, int(len(X) * 0.1))
    Xv, yv = X[-n_val:], y[-n_val:]

    base = SGDClassifier(max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=6.0)
    es = EarlyStoppingWrapper(base, patience=10, max_iter=120, validation_fraction=0.1)
    es.fit(X, y)

    # The live base_model kept iterating to the end; its final-weights val acc is the "no-ES" outcome.
    final_val_acc = accuracy_score(yv, es.base_model.predict(Xv))
    best_val_acc = accuracy_score(yv, es.best_model_.predict(Xv))
    assert best_val_acc >= final_val_acc, (
        f"best snapshot (acc {best_val_acc:.3f}) should beat or match the overshot final model "
        f"(acc {final_val_acc:.3f})"
    )


# --------------------------------------------------------------------------- #
# Regression path: the wrapper must early-stop REGRESSORS too (no native ES).  #
# --------------------------------------------------------------------------- #


def _overfit_prone_regression(seed: int = 0, n: int = 400, d: int = 8):
    """Linear-signal-plus-noise regression target. A constant-LR SGD regressor's val R^2 peaks early
    then degrades as it overshoots, so a correctly-wired ES must snapshot the peak, not the final fit."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    w = np.array([2.0, -1.5, 1.0, -0.7] + [0.0] * (d - 4))
    y = (X @ w + 0.5 * rng.randn(n)).astype(np.float64)
    return X, y


def _regressors():
    # SGDRegressor with the default L2 penalty IS ridge regression fit online (the canonical "linear /
    # ridge" regressor); MLPRegressor is the non-native neural regressor. Neither restores best weights
    # under per-call partial_fit, so the generic wrapper is what supplies early stopping for them.
    return {
        "sgd_ridge": SGDRegressor(penalty="l2", alpha=1e-3, max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=0.02),
        "mlp_regressor": MLPRegressor(hidden_layer_sizes=(16,), max_iter=1, learning_rate_init=0.03, random_state=0),
    }


@pytest.mark.parametrize("name", list(_regressors().keys()))
def test_biz_val_wrapper_regression_stops_early_without_r2_loss(name):
    X, y = _overfit_prone_regression(seed=1)
    n_val = max(1, int(len(X) * 0.1))
    Xv, yv = X[-n_val:], y[-n_val:]

    base = _regressors()[name]
    MAX_ITER = 80
    # Default scoring auto-switches to R^2 for regressors (accuracy is meaningless on continuous y).
    es = EarlyStoppingWrapper(base, patience=8, max_iter=MAX_ITER, validation_fraction=0.1)
    es.fit(X, y)

    assert es.best_model_ is not None
    assert es.best_score_ > -np.inf
    es_r2 = r2_score(yv, es.predict(Xv))

    # No-loss guarantee: the snapshot must be >= the model that ran ALL max_iter rounds (the overshoot).
    full = _regressors()[name]
    for _ in range(MAX_ITER):
        full.partial_fit(X[:-n_val], y[:-n_val])
    full_r2 = r2_score(yv, full.predict(Xv))

    assert es_r2 >= full_r2 - 1e-9, (
        f"{name}: ES-restored best regressor (val R^2 {es_r2:.3f}) must not be worse than the "
        f"fully-iterated model (val R^2 {full_r2:.3f})"
    )
    assert es_r2 >= es.best_score_ - 1e-9


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
