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


def _regime(n: int, lo: float, hi: float, alpha: float, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """A frame whose target is ``alpha * base`` plus a feature signal, with the base drawn from ``[lo, hi]``."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"b": rng.uniform(lo, hi, n), "x": rng.normal(size=n)})
    return X, alpha * X["b"].to_numpy() + 0.5 * X["x"].to_numpy() + rng.normal(0.0, 0.3, n)


# Everything ``fit`` derives from the training rows, which an ``update()`` refit must derive the same way from its buffer.
_DATA_DERIVED = ("alpha", "beta", "y_clip_low", "y_clip_high", "y_train_median", "t_clip_low", "t_clip_high")


@pytest.mark.parametrize("name", ["linear_residual", "linear_residual_robust"])
def test_an_update_refit_leaves_the_state_a_fresh_fit_on_the_buffer_would(name: str):
    """After a drift refit, every data-derived key equals what ``fit`` computes on the same rows (EST-14's parity leg).

    The refit used to move alpha/beta and the y-clip but leave the base range at the dead regime, and to clip T with a
    formula ``fit`` does not use. One ``update`` call carries the whole new regime, so the refit sees exactly those rows.
    """
    X_old, y_old = _regime(500, 0.0, 10.0, 0.9, seed=0)
    X_new, y_new = _regime(400, 50.0, 60.0, 2.0, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        streamed = CompositeTargetEstimator(
            base_estimator=LinearRegression(), transform_name=name, base_column="b",
            online_refit_enabled=True, online_refit_min_buffer_n=200, online_refit_buffer_n=400,
        ).fit(X_old, y_old)
        info = streamed.update(y_recent=y_new, base_recent=X_new["b"].to_numpy())
        fresh = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name=name, base_column="b").fit(X_new, y_new)
    assert info["refit"], "the fixture must trigger a refit for the comparison to mean anything"
    got, want = streamed.fitted_params_, fresh.fitted_params_
    tol = 1e-9
    for key in _DATA_DERIVED:
        np.testing.assert_allclose(got[key], want[key], rtol=tol, atol=tol, err_msg=f"{name}: {key} differs from a fresh fit")
    assert set(got["base_fit_range"]) == set(want["base_fit_range"])
    for part in want["base_fit_range"]:  # lo / hi / iqr of the base the spec is calibrated on
        np.testing.assert_allclose(np.asarray(got["base_fit_range"][part], dtype=float), np.asarray(want["base_fit_range"][part], dtype=float),
                                   rtol=1e-9, err_msg=f"base range {part} still describes the regime the buffer replaced")


@pytest.mark.parametrize("name", sorted(TRANSFORMS_REGISTRY))
def test_an_unpickled_wrapper_is_the_wrapper_that_was_pickled(name: str):
    """Pickling keeps every public fitted attribute, every fitted parameter and every prediction (the unpickle leg)."""
    import pickle

    fitted, _built, X = _pair(name)
    restored = pickle.loads(pickle.dumps(fitted))  # nosec B301 - an object this test created, not untrusted input
    pub = lambda est: {k for k in vars(est) if k.endswith("_") and not k.startswith("_")}
    assert pub(restored) == pub(fitted), sorted(pub(restored) ^ pub(fitted))
    assert set(restored.fitted_params_) == set(fitted.fitted_params_)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.testing.assert_allclose(restored.predict(X.iloc[:40]), fitted.predict(X.iloc[:40]), rtol=0, atol=0, equal_nan=True)
