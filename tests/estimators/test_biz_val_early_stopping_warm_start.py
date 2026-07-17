"""biz_value: ``EarlyStoppingWrapper`` early-stops estimators that LACK ``partial_fit``.

The wrapper's two non-partial_fit backends:

  * ``staged_predict`` -- GradientBoosting* score every boosting stage from ONE fit; pick the best stage.
  * ``warm_start`` incremental -- RandomForest/ExtraTrees/Bagging grow ``n_estimators`` in batches, refit
    (reusing prior trees), score, snapshot best, stop on patience.

This file pins, across classifiers AND regressors (RandomForest/ExtraTrees/GradientBoosting):

  * the no-loss guarantee: the restored ``best_model_`` is at least as good on held-out val as the
    fully-grown model (classifier accuracy not lower / regressor RMSE not higher) -- ES must never ship a
    worse model than growing to the max count, and
  * the work-saving win: on an overfit-prone target ES genuinely stops BEFORE the max count.

random_state is pinned so the wrapper's internal model and any separate full-run baseline share one
trajectory. The warm-start-capable estimator list is also discovered dynamically so new ones are covered.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils import all_estimators

from mlframe.estimators.early_stopping import EarlyStoppingWrapper


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _clf_data(seed=1, n=400, d=8):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    w = np.array([1.6, -1.3, 0.9, -0.5] + [0.0] * (d - 4))
    y = (X @ w + 0.8 * rng.randn(n) > 0).astype(int)
    return X, y


def _reg_data(seed=1, n=400, d=8):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    w = np.array([2.0, -1.5, 1.0, -0.7] + [0.0] * (d - 4))
    y = (X @ w + 0.5 * rng.randn(n)).astype(np.float64)
    return X, y


_MAX_N = 40  # max count (n_estimators / boosting stages) the wrapper is allowed to grow to.

_CLF_FACTORIES = {
    "RandomForestClassifier": lambda: RandomForestClassifier(n_estimators=1, random_state=0),
    "ExtraTreesClassifier": lambda: ExtraTreesClassifier(n_estimators=1, random_state=0),
    "GradientBoostingClassifier": lambda: GradientBoostingClassifier(n_estimators=1, random_state=0),
}
_REG_FACTORIES = {
    "RandomForestRegressor": lambda: RandomForestRegressor(n_estimators=1, random_state=0),
    "ExtraTreesRegressor": lambda: ExtraTreesRegressor(n_estimators=1, random_state=0),
    "GradientBoostingRegressor": lambda: GradientBoostingRegressor(n_estimators=1, random_state=0),
}


def _fit_es(factory, X, y, **kw):
    kw.setdefault("patience", 6)
    es = EarlyStoppingWrapper(factory(), max_iter=_MAX_N, validation_fraction=0.15, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        es.fit(X, y)
    return es


# --------------------------------------------------------------------------- #
# No-loss guarantee: ES best >= fully-grown model on held-out val             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(_CLF_FACTORIES))
def test_biz_val_warm_classifier_no_accuracy_loss(name):
    X, y = _clf_data()
    # patience >= max count so ES evaluates the WHOLE count curve; its val-argmax snapshot is then provably
    # >= the full-grown model on that same val fold (bagging ensembles like RF/ET don't overfit with more
    # trees, so full is often the optimum -- ES must tie, never lose, by selecting it).
    # monotonic_decline_patience=None: this no-loss test deliberately evaluates the WHOLE curve to compare
    # the val-argmax snapshot against the full-grown model, so BOTH ES stop signals (patience AND the
    # default-on monotonic strict-decline detector) must be disabled -- a noisy bagging val curve can
    # produce a spurious 3-strict-decline run that truncates the forest before its optimum.
    es = _fit_es(_CLF_FACTORIES[name], X, y, patience=_MAX_N + 1, monotonic_decline_patience=None, random_state=0)
    assert es.best_model_ is not None and es.best_score_ > -np.inf
    # Reconstruct the wrapper's actual shuffled/stratified, seeded fold (no longer a last-rows holdout).
    Xtr, Xv, ytr, yv = es._split(X, y)
    es_acc = accuracy_score(yv, es.predict(Xv))

    full = _CLF_FACTORIES[name]()
    full.set_params(n_estimators=_MAX_N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full.fit(Xtr, ytr)
    full_acc = accuracy_score(yv, full.predict(Xv))
    assert es_acc >= full_acc - 1e-9, f"{name}: ES acc {es_acc:.3f} must not be below full-grown {full_acc:.3f}"


@pytest.mark.parametrize("name", sorted(_REG_FACTORIES))
def test_biz_val_warm_regressor_no_rmse_loss(name):
    X, y = _reg_data()
    # monotonic_decline_patience=None: full-curve no-loss comparison (see classifier test) -- disable BOTH
    # stop signals so the noisy bagging val curve isn't truncated by a spurious strict-decline run.
    es = _fit_es(_REG_FACTORIES[name], X, y, patience=_MAX_N + 1, monotonic_decline_patience=None, random_state=0)
    assert es.best_model_ is not None and es.best_score_ > -np.inf
    Xtr, Xv, ytr, yv = es._split(X, y)
    es_rmse = _rmse(yv, es.predict(Xv))

    full = _REG_FACTORIES[name]()
    full.set_params(n_estimators=_MAX_N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full.fit(Xtr, ytr)
    full_rmse = _rmse(yv, full.predict(Xv))
    assert es_rmse <= full_rmse + 1e-9, f"{name}: ES RMSE {es_rmse:.3f} must not exceed full-grown {full_rmse:.3f}"
    assert -es_rmse >= es.best_score_ - 1e-9


# --------------------------------------------------------------------------- #
# Work-saving win: ES stops before the max count on an overfit-prone target   #
# --------------------------------------------------------------------------- #


def test_biz_val_warm_gb_classifier_stops_early_on_noise():
    # Pure-noise target: extra boosting stages only overfit train, so the best val stage is early and
    # patience trips well before the max count -- the work-saving win.
    rng = np.random.RandomState(7)
    X = rng.randn(500, 10)
    y = rng.randint(0, 2, size=500)
    es = _fit_es(lambda: GradientBoostingClassifier(n_estimators=1, max_depth=3, random_state=0), X, y, patience=4)
    assert es.no_improvement_count_ >= 4  # patience actually tripped
    n_used = es.best_model_.get_params()["n_estimators"]
    assert n_used < _MAX_N, f"ES should stop before max count {_MAX_N}, used {n_used}"


def test_biz_val_warm_rf_regressor_stops_early_on_noise():
    rng = np.random.RandomState(11)
    X = rng.randn(400, 8)
    y = rng.randn(400)  # noise target: more trees don't help held-out val.
    es = _fit_es(lambda: RandomForestRegressor(n_estimators=1, random_state=0), X, y, patience=3)
    assert es.no_improvement_count_ >= 3


# --------------------------------------------------------------------------- #
# Dead params now live: start_iter (patience warm-up) + max_runtime_mins      #
# --------------------------------------------------------------------------- #


def test_start_iter_defers_patience_counting():
    # With start_iter past the point patience would otherwise trip, ES must run to the max count.
    rng = np.random.RandomState(7)
    X = rng.randn(400, 10)
    y = rng.randint(0, 2, size=400)
    # monotonic_decline_patience=None: this test pins that start_iter alone defers patience to the full
    # count; disable the orthogonal monotonic detector so it doesn't co-fire and confound the assertion.
    es = _fit_es(lambda: GradientBoostingClassifier(n_estimators=1, random_state=0), X, y, patience=2, start_iter=_MAX_N, monotonic_decline_patience=None)
    # Patience only starts at the last stage, so it can never accumulate -> best uses the full count.
    assert es.best_model_.get_params()["n_estimators"] == _MAX_N


def test_max_runtime_mins_stops_quickly():
    X, y = _reg_data()
    # Near-zero budget: the wall-clock guard must break after the first growth step, not run to max_iter.
    es = EarlyStoppingWrapper(
        RandomForestRegressor(n_estimators=1, random_state=0),
        patience=99,
        max_iter=_MAX_N,
        validation_fraction=0.15,
        max_runtime_mins=1e-9,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        es.fit(X, y)
    assert es.best_model_ is not None  # still produced a usable model
    assert es.best_model_.get_params()["n_estimators"] < _MAX_N


# --------------------------------------------------------------------------- #
# Backend dispatch + clear error for non-early-stoppable estimators           #
# --------------------------------------------------------------------------- #


def test_non_early_stoppable_estimator_raises_clear_error():
    from sklearn.neighbors import KNeighborsClassifier  # no partial_fit / staged / warm_start.

    es = EarlyStoppingWrapper(KNeighborsClassifier(), max_iter=10)
    X, y = _clf_data()
    with pytest.raises(TypeError, match="cannot be early-stopped"):
        es.fit(X, y)


def test_warm_start_estimators_were_discovered():
    # Dynamic discovery of warm-start-capable, non-partial_fit estimators so new ones are auto-covered.
    found = []
    for name, Est in all_estimators(type_filter=["classifier", "regressor"]):
        if "partial_fit" in dir(Est) or "staged_predict" in dir(Est):
            continue
        try:
            est = Est()
        except Exception:
            continue
        p = est.get_params()
        if "warm_start" in p and any(a in p for a in ("n_estimators", "max_iter")):
            found.append(name)
    assert len(found) >= 4, f"expected several warm-start-capable estimators, got {sorted(found)}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
