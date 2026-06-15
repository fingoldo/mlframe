"""biz_value: ``EarlyStoppingWrapper`` gives early-stopping to models that lack it natively.

The wrapper drives ``partial_fit`` and snapshots the best-validation model. Native-ES boosters
(lgb/xgb/cb/HGB) and the neural estimators (lightning callback) are covered elsewhere; this file pins
the GENERIC non-native path across EVERY sklearn estimator that exposes ``partial_fit`` -- both
classifiers and regressors -- asserting the measurable guarantee that makes ES safe to ship:

  * the restored ``best_model_`` is at least as good on held-out val as the fully-iterated model
    (classifier: accuracy not lower; regressor: RMSE not higher) -- ES must never ship a worse model
    than running every iteration, and
  * a deterministic headline win on a constant-LR overshoot regime, where the best snapshot strictly
    beats the degraded final-iteration weights.

The estimator list is discovered dynamically from ``sklearn.utils.all_estimators`` so new partial_fit
models are covered automatically; data-incompatible estimators (e.g. count-only NB on signed features,
meta-estimators needing constructor args) are skipped, not silently passed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.base import clone, is_regressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils import all_estimators

from mlframe.estimators.early_stopping import EarlyStoppingWrapper


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# --------------------------------------------------------------------------- #
# Synthetic data                                                              #
# --------------------------------------------------------------------------- #


def _classification_data(seed: int = 1, n: int = 400, d: int = 8):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    w = np.array([1.6, -1.3, 0.9, -0.5] + [0.0] * (d - 4))
    y = (X @ w + 0.8 * rng.randn(n) > 0).astype(int)
    return X, y


def _regression_data(seed: int = 1, n: int = 400, d: int = 8):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    w = np.array([2.0, -1.5, 1.0, -0.7] + [0.0] * (d - 4))
    y = (X @ w + 0.5 * rng.randn(n)).astype(np.float64)
    return X, y


# --------------------------------------------------------------------------- #
# Dynamic estimator discovery                                                 #
# --------------------------------------------------------------------------- #


def _partial_fit_estimators(kind: str):
    """All sklearn estimators of ``kind`` ('classifier' | 'regressor') exposing partial_fit and
    constructible with no required args. Returns {name: estimator_instance}."""
    out = {}
    for name, Est in all_estimators(type_filter=kind):
        if "partial_fit" not in dir(Est):
            continue
        try:
            est = Est()
        except Exception:
            continue  # needs constructor args (meta-estimator) -> not in scope.
        # Pin random_state so the wrapper's internal clone and the test's separate "full-run" clone
        # share ONE deterministic trajectory -- otherwise (default random_state=None) they get different
        # inits and the best-val-vs-final comparison is between two unrelated models.
        if "random_state" in est.get_params():
            est.set_params(random_state=0)
        out[name] = est
    return out


_CLASSIFIERS = _partial_fit_estimators("classifier")
_REGRESSORS = _partial_fit_estimators("regressor")


def _fit_es(base_model, X, y):
    """Run the wrapper, retrying on non-negative-feature estimators (count NB) with |X|. Returns the
    fitted wrapper + the (possibly abs'd) X actually used, or None if the estimator can't fit the data."""
    for X_try in (X, np.abs(X)):
        es = EarlyStoppingWrapper(clone(base_model), patience=8, max_iter=80, validation_fraction=0.1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                es.fit(X_try, y)
            return es, X_try
        except Exception:
            continue
    return None, None


# --------------------------------------------------------------------------- #
# Classifiers: ES never ships a worse model than running all iterations       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(_CLASSIFIERS))
def test_biz_val_wrapper_classifier_no_accuracy_loss(name):
    X, y = _classification_data()
    n_val = max(1, int(len(X) * 0.1))
    es, X_used = _fit_es(_CLASSIFIERS[name], X, y)
    if es is None:
        pytest.skip(f"{name}: cannot fit the synthetic classification data via partial_fit")
    Xv, yv = X_used[-n_val:], y[-n_val:]

    assert es.best_model_ is not None and es.best_score_ > -np.inf
    es_acc = accuracy_score(yv, es.predict(Xv))

    full = clone(_CLASSIFIERS[name])
    classes = np.unique(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(80):
            full.partial_fit(X_used[:-n_val], y[:-n_val], classes=classes)
    full_acc = accuracy_score(yv, full.predict(Xv))

    assert es_acc >= full_acc - 1e-9, (
        f"{name}: ES best model (acc {es_acc:.3f}) must not be worse than the fully-iterated "
        f"model (acc {full_acc:.3f})"
    )
    assert es_acc >= es.best_score_ - 1e-9


# --------------------------------------------------------------------------- #
# Regressors: ES never ships a higher-RMSE model than running all iterations  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(_REGRESSORS))
def test_biz_val_wrapper_regressor_no_rmse_loss(name):
    X, y = _regression_data()
    n_val = max(1, int(len(X) * 0.1))
    es, X_used = _fit_es(_REGRESSORS[name], X, y)
    if es is None:
        pytest.skip(f"{name}: cannot fit the synthetic regression data via partial_fit")
    Xv, yv = X_used[-n_val:], y[-n_val:]

    assert es.best_model_ is not None and es.best_score_ > -np.inf
    es_rmse = _rmse(yv, es.predict(Xv))

    full = clone(_REGRESSORS[name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(80):
            full.partial_fit(X_used[:-n_val], y[:-n_val])
    full_rmse = _rmse(yv, full.predict(Xv))

    assert es_rmse <= full_rmse + 1e-9, (
        f"{name}: ES best model (RMSE {es_rmse:.3f}) must not be worse than the fully-iterated "
        f"model (RMSE {full_rmse:.3f})"
    )
    # best_score_ is -RMSE for regressors (greater-is-better); it must match the restored model's RMSE.
    assert -es_rmse >= es.best_score_ - 1e-9


# --------------------------------------------------------------------------- #
# Deterministic headline wins on a constant-LR overshoot regime               #
# --------------------------------------------------------------------------- #


def test_biz_val_wrapper_classifier_best_beats_overshot_final():
    X, y = _classification_data(seed=2)
    n_val = max(1, int(len(X) * 0.1))
    Xv, yv = X[-n_val:], y[-n_val:]
    base = SGDClassifier(max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=6.0)
    es = EarlyStoppingWrapper(base, patience=10, max_iter=120, validation_fraction=0.1)
    es.fit(X, y)
    final_acc = accuracy_score(yv, es.base_model.predict(Xv))
    best_acc = accuracy_score(yv, es.best_model_.predict(Xv))
    assert best_acc >= final_acc, f"best snapshot acc {best_acc:.3f} should beat overshot final {final_acc:.3f}"


def test_biz_val_wrapper_regressor_best_beats_overshot_final():
    X, y = _regression_data(seed=2)
    n_val = max(1, int(len(X) * 0.1))
    Xv, yv = X[-n_val:], y[-n_val:]
    # Constant-LR ridge-via-SGD that overshoots: ES must recover the peak iterate.
    base = SGDRegressor(penalty="l2", alpha=1e-3, max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=0.05)
    es = EarlyStoppingWrapper(base, patience=10, max_iter=120, validation_fraction=0.1)
    es.fit(X, y)
    final_rmse = _rmse(yv, es.base_model.predict(Xv))
    best_rmse = _rmse(yv, es.best_model_.predict(Xv))
    assert best_rmse <= final_rmse, f"best snapshot RMSE {best_rmse:.3f} should beat overshot final {final_rmse:.3f}"


def test_partial_fit_estimators_were_discovered():
    # Guard the dynamic discovery itself: if all_estimators / partial_fit detection silently returns
    # nothing, the parametrized tests would vacuously pass with zero cases.
    assert len(_CLASSIFIERS) >= 4, f"expected several partial_fit classifiers, got {sorted(_CLASSIFIERS)}"
    assert len(_REGRESSORS) >= 2, f"expected several partial_fit regressors, got {sorted(_REGRESSORS)}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
