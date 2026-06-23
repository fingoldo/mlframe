"""sklearn-API-compliance regression tests for ``BorutaShap`` (SK2).

``fit`` must NOT reassign the verbatim ``model`` constructor param when it defaults to None: the resolved
RandomForest surrogate goes into the learned ``model_`` attribute instead, so ``get_params`` on a fitted
instance still returns ``model=None`` and ``clone`` round-trips the original args (no stale fitted estimator).
"""

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone

from mlframe.feature_selection.boruta_shap import BorutaShap


def _data(seed=0):
    rng = np.random.default_rng(seed)
    n = 200
    X = pd.DataFrame(
        {
            "signal": rng.standard_normal(n),
            "noise1": rng.standard_normal(n),
            "noise2": rng.standard_normal(n),
        }
    )
    y = pd.Series((X["signal"] > 0).astype(int))
    return X, y


def _fast_selector(**kw):
    # Tiny config so each fit is < ~5s.
    params = dict(importance_measure="gini", n_trials=5, percentile=90, verbose=False)
    params.update(kw)
    return BorutaShap(**params)


def test_fit_does_not_mutate_model_param_when_default():
    X, y = _data()
    sel = _fast_selector(model=None, random_state=0)
    sel.fit(X, y)

    # The verbatim param stays None; the resolved surrogate lives on model_.
    assert sel.get_params(deep=False)["model"] is None, "fit() reassigned the verbatim model param"
    assert sel.model is None
    assert sel.model_ is not None


def test_clone_of_fitted_roundtrips_params():
    X, y = _data()
    sel = _fast_selector(model=None, random_state=0)
    sel.fit(X, y)

    original = BorutaShap(importance_measure="gini", n_trials=5, percentile=90, verbose=False, model=None, random_state=0)
    cloned = clone(sel)
    assert cloned.get_params(deep=False) == original.get_params(deep=False)


def test_train_or_test_param_not_mutated_by_auto():
    X, y = _data()
    # 'auto' may pin held-out permutation, which must NOT rewrite the verbatim train_or_test param.
    sel = _fast_selector(model=None, random_state=0, importance_measure="auto", train_or_test="train")
    sel.fit(X, y)
    assert sel.get_params(deep=False)["train_or_test"] == "train"


def test_set_params_random_state_then_fit_is_reproducible():
    X, y = _data()
    a = _fast_selector(model=None)
    a.set_params(random_state=7)
    a.fit(X, y)

    b = _fast_selector(model=None)
    b.set_params(random_state=7)
    b.fit(X, y)

    assert sorted(a.selected_features_) == sorted(b.selected_features_)


def test_transform_validates_feature_space():
    X, y = _data()
    sel = _fast_selector(model=None, random_state=0)
    sel.fit(X, y)

    # Wrong width -> ValueError.
    with pytest.raises(ValueError):
        sel.transform(X.iloc[:, :2])
