"""Regression tests for TC14/TC39 (0-row val guard in _setup_eval_set) and
TC13 (internal ES split seed threaded from the outer seed, not a hardcoded constant)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training._data_helpers import _setup_eval_set, maybe_wrap_for_partial_fit_es
from mlframe.training._partial_fit_es_wrapper import PartialFitESWrapper, _split_train_val


# -- TC14/TC39 -----------------------------------------------------------------------------


@pytest.mark.parametrize("category", ["lgb", "cb", "xgb", "mlp"])
def test_setup_eval_set_raises_on_zero_row_val(category):
    """A 0-row val must raise an actionable error from ANY source (it silently
    disables early stopping). Pre-fix code skipped straight to eval_set population."""
    fit_params: dict = {}
    empty_df = pd.DataFrame({"a": pd.Series([], dtype="float64")})
    empty_target = pd.Series([], dtype="float64")
    with pytest.raises(ValueError, match="0-row validation set"):
        _setup_eval_set(
            model_type_name="LGBMClassifier",
            fit_params=fit_params,
            val_df=empty_df,
            val_target=empty_target,
            model_category=category,
        )


def test_setup_eval_set_accepts_nonempty_val():
    """Sanity: a non-empty val still populates eval_set (no false positive)."""
    fit_params: dict = {}
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.Series([0, 1, 0])
    _setup_eval_set(
        model_type_name="LGBMClassifier",
        fit_params=fit_params,
        val_df=df,
        val_target=y,
        model_category="lgb",
    )
    assert "eval_set" in fit_params


# -- TC13 ----------------------------------------------------------------------------------


def test_split_seed_independent_across_outer_seeds():
    """The internal ES split must depend on the seed threaded into the wrapper, so two
    different outer seeds yield different splits. Pre-fix: random_state hardcoded to 42,
    so the split was identical regardless of the outer seed."""
    X = np.arange(200).reshape(100, 2).astype(float)
    y = (np.arange(100) % 2).astype(int)
    Xtr_a, _, _, _ = _split_train_val(X, y, 0.3, random_state=1)
    Xtr_b, _, _, _ = _split_train_val(X, y, 0.3, random_state=2)
    assert not np.array_equal(Xtr_a, Xtr_b), "different seeds must produce different splits"


def test_split_seed_reproducible_for_same_seed():
    """Same random_state passed to _split_train_val twice yields bit-identical splits."""
    X = np.arange(200).reshape(100, 2).astype(float)
    y = (np.arange(100) % 2).astype(int)
    a = _split_train_val(X, y, 0.3, random_state=7)
    b = _split_train_val(X, y, 0.3, random_state=7)
    assert np.array_equal(a[0], b[0])


def test_wrapper_default_random_state_is_none():
    """Pre-fix the default was a hardcoded 42; the default must now be None so the split
    varies with the outer seed unless an explicit seed is passed."""
    from sklearn.linear_model import SGDRegressor

    w = PartialFitESWrapper(SGDRegressor())
    assert w.random_state is None
    w2 = PartialFitESWrapper(SGDRegressor(), random_state=5)
    assert w2.random_state == 5


def test_maybe_wrap_threads_random_state():
    """maybe_wrap_for_partial_fit_es must forward its random_state to the wrapper so the
    ES split derives from the outer suite seed, not a constant."""
    from sklearn.linear_model import SGDRegressor

    X = np.random.RandomState(0).randn(50, 3)
    y = np.random.RandomState(1).randn(50)
    wrapped, did = maybe_wrap_for_partial_fit_es(
        SGDRegressor(),
        model_category="sgd",
        X_val=X,
        y_val=y,
        is_classification=False,
        random_state=123,
    )
    assert did
    assert isinstance(wrapped, PartialFitESWrapper)
    assert wrapped.random_state == 123
