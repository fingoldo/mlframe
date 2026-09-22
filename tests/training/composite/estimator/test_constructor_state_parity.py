"""``from_fitted_inner`` builds the same estimator ``fit`` does, for every registry transform.

The suite and every OOF refit build their wrappers with ``from_fitted_inner``, so any state it computes differently from
``fit`` is state every deployed composite has wrong. It lacked the base range (the soft shrink was inert), and it clipped T
to a ``+/-10 std(y)`` guess where ``fit`` uses the train-T envelope: 42 of 51 transforms got a band tens of times too wide
(+/-48 against [-0.17, 0.18] for asinh_residual) or not even centred (ratio's T lives in [0.6, 4.8]).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY

# Attributes only one constructor sets, with the reason.
_EXEMPT = {"inner_pre_pipeline_": "from_fitted_inner records the entry's pipeline explicitly; fit trains the inner itself"}


def _pair(name: str):
    """``(fit wrapper, from_fitted_inner wrapper, X)`` on the same data, inner and spec."""
    t = TRANSFORMS_REGISTRY[name]
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"b": rng.uniform(2.0, 10.0, n), "b2": rng.uniform(1.0, 5.0, n), "x": rng.normal(size=n), "g": np.arange(n) % 6})
    y = 2.0 * X["b"] + 0.5 * X["b2"] + 3.0 + rng.normal(0.0, 0.2, n)
    kw = {"base_column": "b"}
    if t.n_bases > 1:
        kw["base_columns"] = ("b", "b2")
    if t.requires_groups:
        kw["group_column"] = "g"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name=name, **kw).fit(X, y)
        built = CompositeTargetEstimator.from_fitted_inner(
            fitted_inner=fitted.estimator_, transform_name=name, base_column="b", transform_fitted_params=fitted.fitted_params_,
            y_train=y.to_numpy(), base_columns=kw.get("base_columns"),
            base_train=X[["b", "b2"]].to_numpy() if t.n_bases > 1 else X["b"].to_numpy(),
            group_column=kw.get("group_column"), groups_train=X["g"].to_numpy() if t.requires_groups else None,
        )
    return fitted, built, X


@pytest.mark.parametrize("name", sorted(TRANSFORMS_REGISTRY))
def test_from_fitted_inner_matches_fit(name: str):
    """Same public fitted attributes (bar the exemptions), same parameter keys, same T-clip band, same predictions."""
    fitted, built, X = _pair(name)
    pub = lambda est: {k for k in vars(est) if k.endswith("_") and not k.startswith("_")}
    assert pub(fitted) ^ pub(built) <= set(_EXEMPT), sorted(pub(fitted) ^ pub(built))
    assert set(fitted.fitted_params_) == set(built.fitted_params_), sorted(set(fitted.fitted_params_) ^ set(built.fitted_params_))
    band = lambda est: (est.fitted_params_["t_clip_low"], est.fitted_params_["t_clip_high"])
    np.testing.assert_allclose(band(built), band(fitted), rtol=1e-9, err_msg="T-clip envelope differs from fit()")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.testing.assert_allclose(built.predict(X.iloc[:40]), fitted.predict(X.iloc[:40]), rtol=1e-9, atol=1e-9, equal_nan=True)
