"""sklearn-API-compliance regression tests for ``EarlyStoppingWrapper`` (SK1).

``fit`` must NOT mutate the caller-supplied ``base_model``: it clones it into a private working attribute
(``estimator_``) so that under clone / GridSearchCV the shared ``base_model`` object is never trained in place.
"""

import numpy as np
import pytest

from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.utils.validation import check_is_fitted

from mlframe.estimators.early_stopping import EarlyStoppingWrapper


def _is_fitted(est) -> bool:
    try:
        check_is_fitted(est)
        return True
    except NotFittedError:
        return False


def _data(regression: bool):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4))
    if regression:
        y = X[:, 0] * 2.0 + rng.standard_normal(80) * 0.1
    else:
        y = (X[:, 0] > 0).astype(int)
    return X, y


@pytest.mark.parametrize(
    "base_factory, regression",
    [
        (lambda: SGDClassifier(max_iter=1, tol=None, random_state=0), False),
        (lambda: SGDRegressor(max_iter=1, tol=None, random_state=0), True),
    ],
    ids=["classifier", "regressor"],
)
def test_fit_does_not_mutate_base_model_and_clones_are_independent(base_factory, regression):
    base = base_factory()
    X, y = _data(regression)

    wrapper = EarlyStoppingWrapper(base, patience=3, max_iter=20, validation_fraction=0.2, random_state=0)

    # Two clones (as GridSearchCV would produce) fit on different folds must not share fitted state.
    w1 = clone(wrapper)
    w2 = clone(wrapper)
    # clone must round-trip the verbatim base_model (still unfitted).
    assert not _is_fitted(w1.base_model)
    assert not _is_fitted(w2.base_model)

    w1.fit(X[:40], y[:40])
    w2.fit(X[40:], y[40:])

    # The original caller-supplied base_model is NEVER fitted by fit().
    assert not _is_fitted(base), "fit() mutated the caller-supplied base_model in place"
    assert not _is_fitted(w1.base_model)
    assert not _is_fitted(w2.base_model)

    # The two clones operated on independent working estimators.
    assert w1.estimator_ is not w2.estimator_
    assert w1.estimator_ is not base
    assert _is_fitted(w1.estimator_)

    # n_features_in_ stamped from the fit data width.
    assert w1.n_features_in_ == X.shape[1]


@pytest.mark.parametrize(
    "base_factory, regression",
    [
        (lambda: SGDClassifier(max_iter=1, tol=None, random_state=0), False),
        (lambda: SGDRegressor(max_iter=1, tol=None, random_state=0), True),
    ],
    ids=["classifier", "regressor"],
)
def test_get_params_and_clone_roundtrip(base_factory, regression):
    base = base_factory()
    wrapper = EarlyStoppingWrapper(base, patience=4, max_iter=15)
    params = wrapper.get_params(deep=False)
    assert params["base_model"] is base
    assert params["patience"] == 4
    cloned = clone(wrapper)
    assert cloned.get_params(deep=False)["patience"] == 4

    # predict before fit raises the canonical NotFittedError (check_is_fitted on best_model_).
    with pytest.raises(NotFittedError):
        cloned.predict(np.zeros((2, 4)))
