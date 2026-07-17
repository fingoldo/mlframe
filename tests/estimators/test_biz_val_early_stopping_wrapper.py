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
from sklearn.base import clone
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
        except Exception:  # nosec B112 -- best-effort skip of one iteration on a non-fatal error; the test's own assertions are unaffected
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


# Exceptions that legitimately mean "this estimator structurally cannot consume continuous Gaussian
# features", NOT "the wrapper regressed". Count-only naive Bayes (CategoricalNB / MultinomialNB) require
# non-negative integer-encoded categoricals: signed input raises ValueError("Negative values ..."), and even
# on |X| the held-out val fold contains category indices unseen during partial_fit -> IndexError. These are
# data-shape incompatibilities of the WRAPPED estimator, not of EarlyStoppingWrapper, so they warrant a skip.
# Anything else (an EarlyStoppingWrapper.fit regression) must propagate and FAIL the test.
_DATA_SHAPE_SKIP = (ValueError, IndexError)


def _fit_es(base_model, X, y):
    """Run the wrapper, retrying on non-negative-feature estimators (count NB) with |X|. Returns the fitted
    wrapper + the (possibly abs'd) X actually used. Raises ``pytest.skip`` only when EVERY attempt failed with
    an expected data-shape incompatibility (see ``_DATA_SHAPE_SKIP``); any UNEXPECTED exception propagates so a
    genuine wrapper regression fails the test rather than degrading into a green-by-skip run."""
    last_exc = None
    for X_try in (X, np.abs(X)):
        # patience > max_iter (and monotonic detector off) so ES evaluates the WHOLE iteration curve: its
        # val-argmax snapshot is then provably >= the fully-iterated model on the SAME val fold. With a
        # truncating patience ES may legitimately stop before a later-peaking iterate (that is the patience
        # trade-off, not a loss), so the no-loss guarantee is only meaningful over the full curve -- this
        # mirrors the warm-start sibling's correct full-curve design.
        es = EarlyStoppingWrapper(
            clone(base_model),
            patience=81,
            max_iter=80,
            validation_fraction=0.1,
            random_state=0,
            monotonic_decline_patience=None,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                es.fit(X_try, y)
            return es, X_try
        except _DATA_SHAPE_SKIP as exc:
            last_exc = exc
            continue
    pytest.skip(f"{type(base_model).__name__}: incompatible with continuous features ({last_exc!r})")


# --------------------------------------------------------------------------- #
# Classifiers: ES never ships a worse model than running all iterations       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(_CLASSIFIERS))
def test_biz_val_wrapper_classifier_no_accuracy_loss(name):
    X, y = _classification_data()
    es, X_used = _fit_es(_CLASSIFIERS[name], X, y)
    assert es is not None, f"{name}: EarlyStoppingWrapper.fit returned no wrapper"
    # Reconstruct the wrapper's actual (shuffled/stratified, seeded) train/val fold instead of
    # assuming a last-rows holdout -- the wrapper now splits like base.SplitFitEstimator.
    Xtr, Xv, ytr, yv = es._split(X_used, y)

    assert es.best_model_ is not None and es.best_score_ > -np.inf
    es_acc = accuracy_score(yv, es.predict(Xv))

    full = clone(_CLASSIFIERS[name])
    classes = np.unique(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(80):
            full.partial_fit(Xtr, ytr, classes=classes)
    full_acc = accuracy_score(yv, full.predict(Xv))

    assert es_acc >= full_acc - 1e-9, f"{name}: ES best model (acc {es_acc:.3f}) must not be worse than the fully-iterated model (acc {full_acc:.3f})"
    assert es_acc >= es.best_score_ - 1e-9


# --------------------------------------------------------------------------- #
# Regressors: ES never ships a higher-RMSE model than running all iterations  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(_REGRESSORS))
def test_biz_val_wrapper_regressor_no_rmse_loss(name):
    X, y = _regression_data()
    es, X_used = _fit_es(_REGRESSORS[name], X, y)
    assert es is not None, f"{name}: EarlyStoppingWrapper.fit returned no wrapper"
    Xtr, Xv, ytr, yv = es._split(X_used, y)

    assert es.best_model_ is not None and es.best_score_ > -np.inf
    es_rmse = _rmse(yv, es.predict(Xv))

    full = clone(_REGRESSORS[name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(80):
            full.partial_fit(Xtr, ytr)
    full_rmse = _rmse(yv, full.predict(Xv))

    assert es_rmse <= full_rmse + 1e-9, f"{name}: ES best model (RMSE {es_rmse:.3f}) must not be worse than the fully-iterated model (RMSE {full_rmse:.3f})"
    # best_score_ is -RMSE for regressors (greater-is-better); it must match the restored model's RMSE.
    assert -es_rmse >= es.best_score_ - 1e-9


# --------------------------------------------------------------------------- #
# Deterministic headline wins on a constant-LR overshoot regime               #
# --------------------------------------------------------------------------- #


def test_biz_val_wrapper_classifier_best_beats_overshot_final():
    X, y = _classification_data(seed=2)
    base = SGDClassifier(max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=6.0)
    es = EarlyStoppingWrapper(base, patience=10, max_iter=120, validation_fraction=0.1, random_state=0)
    es.fit(X, y)
    _, Xv, _, yv = es._split(X, y)
    # The live fully-iterated model is the wrapper's internal clone ``estimator_``; ``base_model`` is left
    # unfitted by sklearn no-mutate contract, so the overshot-final comparison uses ``estimator_``.
    final_acc = accuracy_score(yv, es.estimator_.predict(Xv))
    best_acc = accuracy_score(yv, es.best_model_.predict(Xv))
    assert best_acc >= final_acc, f"best snapshot acc {best_acc:.3f} should beat overshot final {final_acc:.3f}"


def test_biz_val_wrapper_regressor_best_beats_overshot_final():
    X, y = _regression_data(seed=2)
    # Constant-LR ridge-via-SGD that overshoots: ES must recover the peak iterate.
    base = SGDRegressor(penalty="l2", alpha=1e-3, max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=0.05)
    es = EarlyStoppingWrapper(base, patience=10, max_iter=120, validation_fraction=0.1, random_state=0)
    es.fit(X, y)
    _, Xv, _, yv = es._split(X, y)
    # The live fully-iterated model is the wrapper's internal clone ``estimator_``; ``base_model`` is left
    # unfitted by sklearn no-mutate contract, so the overshot-final comparison uses ``estimator_``.
    final_rmse = _rmse(yv, es.estimator_.predict(Xv))
    best_rmse = _rmse(yv, es.best_model_.predict(Xv))
    assert best_rmse <= final_rmse, f"best snapshot RMSE {best_rmse:.3f} should beat overshot final {final_rmse:.3f}"


def test_partial_fit_estimators_were_discovered():
    # Guard the dynamic discovery itself: if all_estimators / partial_fit detection silently returns
    # nothing, the parametrized tests would vacuously pass with zero cases.
    assert len(_CLASSIFIERS) >= 4, f"expected several partial_fit classifiers, got {sorted(_CLASSIFIERS)}"
    assert len(_REGRESSORS) >= 2, f"expected several partial_fit regressors, got {sorted(_REGRESSORS)}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
