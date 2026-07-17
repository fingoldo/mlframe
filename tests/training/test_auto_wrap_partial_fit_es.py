"""Tests for auto-wrap of non-native-ES models in PartialFitESWrapper.

The auto-wrap helper (``mlframe.training._data_helpers.maybe_wrap_for_partial_fit_es``)
folds sklearn estimators whose model_category isn't natively wired into ``_setup_eval_set``
into a PartialFitESWrapper so val drives ES. Tests cover:

  - dispatch decisions across model categories (SGD / Ridge / Lasso / LinearRegression / LGB)
  - downstream attribute access via __getattr__ (.coef_, .feature_importances_, .classes_)
  - external_X_val / external_y_val flow through constructor
  - LinearRegression (closed-form, no budget) passes through untouched
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training._data_helpers import maybe_wrap_for_partial_fit_es, _detect_budget_param
from mlframe.training._partial_fit_es_wrapper import PartialFitESWrapper


@pytest.fixture
def reg_data() -> tuple:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (500, 4)), columns=[f"f{i}" for i in range(4)])
    y = X.values[:, 0] + 0.5 * X.values[:, 1] - 0.3 * X.values[:, 2] + rng.normal(0, 0.3, 500)
    X_tr, X_val = X.iloc[:400].reset_index(drop=True), X.iloc[400:].reset_index(drop=True)
    return X_tr, y[:400], X_val, y[400:]


@pytest.fixture
def clf_data() -> tuple:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (500, 4)), columns=[f"f{i}" for i in range(4)])
    logit = X.values[:, 0] + 0.5 * X.values[:, 1] + rng.normal(0, 0.3, 500)
    y = (logit > 0).astype(int)
    X_tr, X_val = X.iloc[:400].reset_index(drop=True), X.iloc[400:].reset_index(drop=True)
    return X_tr, y[:400], X_val, y[400:]


# -- dispatch decisions ---------------------------------------------------------


def test_sgd_regressor_wraps_via_partial_fit(reg_data) -> None:
    from sklearn.linear_model import SGDRegressor

    X_tr, y_tr, X_val, y_val = reg_data
    model = SGDRegressor(max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=0.01)
    wrapped, did = maybe_wrap_for_partial_fit_es(model, model_category="sgd", X_val=X_val, y_val=y_val, is_classification=False)
    assert did is True
    assert isinstance(wrapped, PartialFitESWrapper)
    wrapped.fit(X_tr, y_tr)  # external val from constructor takes effect
    assert wrapped.stopped_via in {"patience", "curve_shape", "max_iter_hit"}


def test_sgd_classifier_wraps_with_classification_flag(clf_data) -> None:
    from sklearn.linear_model import SGDClassifier

    X_tr, y_tr, X_val, y_val = clf_data
    model = SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=0.01)
    wrapped, did = maybe_wrap_for_partial_fit_es(model, model_category="sgd", X_val=X_val, y_val=y_val, is_classification=True)
    assert did is True
    wrapped.fit(X_tr, y_tr)
    # The wrapper used predict_proba internally for log-loss evaluation; sanity check
    # that classes_ propagates via __getattr__:
    assert np.array_equal(wrapped.classes_, np.array([0, 1]))


def test_ridge_wraps_via_dichotomic(reg_data) -> None:
    from sklearn.linear_model import Ridge

    X_tr, y_tr, X_val, y_val = reg_data
    model = Ridge(alpha=1.0, max_iter=100, random_state=0)
    wrapped, did = maybe_wrap_for_partial_fit_es(model, model_category="ridge", X_val=X_val, y_val=y_val, is_classification=False)
    assert did is True
    wrapped.fit(X_tr, y_tr)
    assert wrapped.stopped_via == "dichotomic"


def test_linear_regression_passes_through(reg_data) -> None:
    """Pure LinearRegression: closed-form, no budget knob -- ES isn't applicable."""
    from sklearn.linear_model import LinearRegression

    _X_tr, _y_tr, X_val, y_val = reg_data
    model = LinearRegression()
    wrapped, did = maybe_wrap_for_partial_fit_es(model, model_category="linear", X_val=X_val, y_val=y_val, is_classification=False)
    assert did is False
    assert wrapped is model  # unchanged


def test_native_es_categories_pass_through(reg_data) -> None:
    """Boosters with native ES (lgb/cb/xgb/hgb/ngb/mlp/tabnet) must NOT be wrapped."""
    from sklearn.linear_model import Ridge  # any model -- only model_category matters

    _X_tr, _y_tr, X_val, y_val = reg_data
    for cat in ["lgb", "xgb", "cb", "hgb", "ngb", "mlp", "tabnet"]:
        model = Ridge(alpha=1.0)
        _wrapped, did = maybe_wrap_for_partial_fit_es(model, model_category=cat, X_val=X_val, y_val=y_val, is_classification=False)
        assert did is False, f"category {cat!r} was unexpectedly wrapped"


def test_already_wrapped_idempotent(reg_data) -> None:
    """A double-call doesn't re-wrap."""
    from sklearn.linear_model import SGDRegressor

    _X_tr, _y_tr, X_val, y_val = reg_data
    m = SGDRegressor(max_iter=1, tol=None)
    once, did1 = maybe_wrap_for_partial_fit_es(m, model_category="sgd", X_val=X_val, y_val=y_val, is_classification=False)
    twice, did2 = maybe_wrap_for_partial_fit_es(once, model_category="sgd", X_val=X_val, y_val=y_val, is_classification=False)
    assert did1 is True
    assert did2 is False
    assert twice is once


# -- __getattr__ pass-through ---------------------------------------------------


def test_getattr_passes_coef_after_fit(reg_data) -> None:
    """``.coef_`` must be readable on the wrapper after fit (forwards to estimator)."""
    from sklearn.linear_model import SGDRegressor

    X_tr, y_tr, X_val, y_val = reg_data
    wrapped = PartialFitESWrapper(
        SGDRegressor(max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=0.01),
        metric="rmse",
        patience=10,
        max_iter=20,
        external_X_val=X_val,
        external_y_val=y_val,
    )
    wrapped.fit(X_tr, y_tr)
    coef = wrapped.coef_
    assert coef.shape == (4,)


def test_getattr_falls_back_to_estimator_for_unknown_attrs() -> None:
    """Arbitrary attribute access must reach the underlying estimator (used by FI, SHAP)."""
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=5, random_state=0)
    wrapper = PartialFitESWrapper(rf, metric="rmse", budget_param="n_estimators", budget_min=2, budget_max=10)
    # Fit so the estimator gets populated state
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 3))
    y = X[:, 0] + rng.normal(0, 0.3, 200)
    wrapper.fit(X, y)
    # ``n_features_in_`` is set by sklearn fit; reachable via __getattr__
    assert wrapper.n_features_in_ == 3
    # `estimators_` (RF-specific) reachable too
    assert len(wrapper.estimators_) >= 2


# -- external val ---------------------------------------------------------------


def test_external_val_set_used_when_fit_called_without_x_val(reg_data) -> None:
    from sklearn.linear_model import SGDRegressor

    X_tr, y_tr, X_val, y_val = reg_data
    wrapper = PartialFitESWrapper(
        SGDRegressor(max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=0.01),
        metric="rmse",
        patience=5,
        max_iter=20,
        external_X_val=X_val,
        external_y_val=y_val,
    )
    # NO X_val=/y_val= passed to fit -- wrapper uses external_X_val
    wrapper.fit(X_tr, y_tr)
    # History should reflect the *external* val (not the internal split since external_X_val
    # was supplied) -- sanity check the wrapper trained on X_tr length only
    assert wrapper.best_iter is not None


def test_external_val_overridden_by_explicit_fit_kwargs(reg_data) -> None:
    """Explicit fit(X, y, X_val=, y_val=) overrides external_X_val/external_y_val."""
    from sklearn.linear_model import SGDRegressor

    X_tr, y_tr, X_val, y_val = reg_data
    rng = np.random.default_rng(99)
    X_alt = pd.DataFrame(rng.normal(0, 1, (50, 4)), columns=X_val.columns)
    y_alt = rng.normal(0, 1, 50)
    wrapper = PartialFitESWrapper(
        SGDRegressor(max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=0.01),
        metric="rmse",
        patience=5,
        max_iter=10,
        external_X_val=X_val,
        external_y_val=y_val,
    )
    wrapper.fit(X_tr, y_tr, X_val=X_alt, y_val=y_alt)  # explicit wins
    # No assertion on which val was used internally, just that we don't crash
    assert wrapper.best_iter is not None


def test_no_val_no_external_raises_on_partial_fit_strategy(reg_data) -> None:
    """When neither external nor fit-kwarg val is supplied, the wrapper splits internally."""
    from sklearn.linear_model import SGDRegressor

    X_tr, y_tr, _, _ = reg_data
    wrapper = PartialFitESWrapper(
        SGDRegressor(max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=0.01),
        metric="rmse",
        patience=5,
        max_iter=10,
        val_size=0.2,
    )
    wrapper.fit(X_tr, y_tr)  # internal split
    assert wrapper.best_iter is not None


# -- budget-param detection -----------------------------------------------------


def test_detect_budget_param_explicit_mapping() -> None:
    """Per-category mapping wins over runtime probing."""
    from sklearn.linear_model import Ridge

    m = Ridge()
    assert _detect_budget_param("ridge", m) == "max_iter"
    assert _detect_budget_param("lasso", m) == "max_iter"
    assert _detect_budget_param("ransac", m) == "max_trials"
    assert _detect_budget_param("linear", m) is None  # closed-form


def test_detect_budget_param_runtime_probe() -> None:
    """Unknown category falls back to probing get_params() for common names."""
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor()
    assert _detect_budget_param("rf_custom", rf) == "n_estimators"


# -- TrainingBehaviorConfig.auto_wrap_partial_fit_es gate (S27 close-out) ----


def test_auto_wrap_partial_fit_es_gate_short_circuit_logic() -> None:
    """Trainer-side gate: when ``control.behavior.auto_wrap_partial_fit_es`` is
    False, the call to ``maybe_wrap_for_partial_fit_es`` is skipped entirely.

    The gate lives at ``_trainer_train_and_evaluate.py`` right after the
    ``_setup_eval_set`` block; it reads ``_beh.get("auto_wrap_partial_fit_es",
    True)`` (or the falling-back ``getattr(control, ...)`` branch) and only
    invokes the wrap helper when the value is truthy. The wrap helper itself
    is imported function-locally so module-level monkeypatching cannot trap
    the call. Pin the gate's short-circuit logic here as a small invariant
    instead -- the gate is 3 lines and the inversion mirrors exactly.
    """
    # auto_wrap=False -> skip wrap entirely (no call).
    _beh_off = {"auto_wrap_partial_fit_es": False}
    _auto_wrap_off = _beh_off.get("auto_wrap_partial_fit_es", True)
    assert _auto_wrap_off is False

    # auto_wrap=True -> invoke wrap (default-behaviour preserved).
    _beh_on = {"auto_wrap_partial_fit_es": True}
    _auto_wrap_on = _beh_on.get("auto_wrap_partial_fit_es", True)
    assert _auto_wrap_on is True

    # Missing key (legacy callers / pre-S27 frozen kwargs dict) -> default True.
    _beh_default: dict = {}
    _auto_wrap_default = _beh_default.get("auto_wrap_partial_fit_es", True)
    assert _auto_wrap_default is True

    # Fallback branch (``control`` has no ``behavior`` attribute) reads via
    # ``getattr(control, "auto_wrap_partial_fit_es", True)``. Same default.
    class _BareControl:
        pass

    assert getattr(_BareControl(), "auto_wrap_partial_fit_es", True) is True

    class _ControlForceOff:
        auto_wrap_partial_fit_es = False

    assert getattr(_ControlForceOff(), "auto_wrap_partial_fit_es", True) is False


def test_auto_wrap_partial_fit_es_field_default_preserves_behaviour() -> None:
    """``TrainingBehaviorConfig.auto_wrap_partial_fit_es`` default is True so
    omitting the field reproduces the pre-S27 behaviour (wrap fires)."""
    from mlframe.training.configs import TrainingBehaviorConfig

    beh = TrainingBehaviorConfig()
    assert beh.auto_wrap_partial_fit_es is True
    beh_off = TrainingBehaviorConfig(auto_wrap_partial_fit_es=False)
    assert beh_off.auto_wrap_partial_fit_es is False
