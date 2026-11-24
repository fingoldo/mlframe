"""The unary-credit pass inside MRMR: the exact transform of a selected column reaches ``transform()`` as a replayable ``unary_credit`` recipe, a control gets nothing, and the pass can be rolled back."""

from __future__ import annotations

import pickle
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._fe_unary_credit import credit_unary_forms, unary_credit_enabled

N = 4000


def _frame(case: str, seed: int = 0):
    """The probe fixtures with six noise columns."""
    rng = np.random.default_rng(seed)
    x2 = rng.standard_normal(N)
    noise = 0.3 * rng.standard_normal(N)
    if case == "sin_wide":
        x1 = rng.uniform(-6, 6, N)
        y = np.sin(x1) + x2 + noise
    elif case == "log_lognormal":
        x1 = rng.lognormal(0, 1, N)
        y = np.log(x1) + x2 + noise
    elif case == "linear":
        x1 = rng.lognormal(0, 1, N)
        y = x1 + x2 + noise
    else:
        raise ValueError(case)
    X = pd.DataFrame({"x1": x1, "x2": x2, **{f"n{k}": rng.standard_normal(N) for k in range(6)}})
    return X, pd.Series(y, name="y")


def test_the_pass_names_the_form_and_the_recipe_replays_it():
    """``sin(x1)`` is credited for the sine fixture and its recipe rebuilds exactly ``sin(x1)`` from a frame that holds only the source column."""
    from mlframe.feature_selection.filters.engineered_recipes.shared import apply_recipe

    X, y = _frame("sin_wide")
    extra = credit_unary_forms(SimpleNamespace(_engineered_features_=[]), X, y.to_numpy(), ["x1", "x2"], seed=0)
    assert [name for _, name in extra] == ["sin(x1)"]
    recipe = extra[0][0]
    assert recipe.kind == "unary_credit" and recipe.src_names == ("x1",)
    restored = pickle.loads(pickle.dumps(recipe))  # nosec B301 - the bytes come from pickle.dumps of this test's own object
    np.testing.assert_allclose(apply_recipe(restored, X.iloc[:50][["x1"]]), np.sin(X["x1"].to_numpy()[:50]))


def test_a_name_already_taken_by_an_engineered_column_is_not_added_twice():
    """If another family already produced ``sin(x1)`` the pass adds nothing."""
    X, y = _frame("sin_wide")
    assert credit_unary_forms(SimpleNamespace(_engineered_features_=["sin(x1)"]), X, y.to_numpy(), ["x1", "x2"], seed=0) == []


def test_the_pass_is_a_no_op_for_a_multiclass_target_and_for_non_numeric_input():
    """Three class codes, an ndarray frame or a missing target: nothing to do (a binary target takes the logit criterion, see below)."""
    X, y = _frame("sin_wide")
    mrmr = SimpleNamespace(_engineered_features_=[])
    three = np.digitize(y.to_numpy(), np.quantile(y.to_numpy(), [1 / 3, 2 / 3]))
    assert credit_unary_forms(mrmr, X, three, ["x1", "x2"]) == []
    assert credit_unary_forms(mrmr, X.to_numpy(), y.to_numpy(), ["x1", "x2"]) == []
    assert credit_unary_forms(mrmr, X, None, ["x1", "x2"]) == []


def test_a_binary_target_takes_the_logit_criterion_and_credits_the_exact_form():
    """Goal 5, thresholded half: a class code of a sigmoid of ``2 log(x1) + x2`` credits ``log(x1)``; the plain linear logit credits nothing."""
    rng = np.random.default_rng(0)
    n = 8000
    x1 = rng.lognormal(0, 1, n)
    x2 = rng.standard_normal(n)
    frame = pd.DataFrame({"x1": x1, "x2": x2})
    mrmr = SimpleNamespace(_engineered_features_=[])
    y_log = (rng.random(n) < 1 / (1 + np.exp(-(2 * np.log(x1) + x2)))).astype(int)
    assert [name for _, name in credit_unary_forms(mrmr, frame, y_log, ["x1", "x2"], seed=0)] == ["log(x1)"]
    y_lin = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * x1 + x2 - 1)))).astype(int)
    assert credit_unary_forms(mrmr, frame, y_lin, ["x1", "x2"], seed=0) == []


def test_the_rollback_switch(monkeypatch):
    """``MLFRAME_FE_UNARY_CREDIT=0`` turns the pass off and any other value leaves it on."""
    X, y = _frame("sin_wide")
    mrmr = SimpleNamespace(_engineered_features_=[])
    monkeypatch.setenv("MLFRAME_FE_UNARY_CREDIT", "0")
    assert not unary_credit_enabled() and credit_unary_forms(mrmr, X, y.to_numpy(), ["x1", "x2"]) == []
    monkeypatch.setenv("MLFRAME_FE_UNARY_CREDIT", "1")
    assert unary_credit_enabled()


@pytest.mark.slow
@pytest.mark.parametrize("case,expected", [("sin_wide", "sin(x1)"), ("log_lognormal", "log(x1)")])
def test_mrmr_outputs_the_exact_transform(case, expected):
    """Goals 1-3 end to end: the output holds the exact form of the transformed column, and ``transform`` returns its values."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _frame(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(verbose=0, random_state=0, n_jobs=1, fe_max_steps=1).fit(X, y)
    names = list(m.get_feature_names_out())
    assert expected in names, names
    assert len(names) <= 3, f"goal 4: two signal features must not give {len(names)} columns: {names}"
    out = pd.DataFrame(m.transform(X), columns=names)
    truth = np.sin(X["x1"]) if case == "sin_wide" else np.log(X["x1"])
    np.testing.assert_allclose(out[expected].to_numpy(), truth.to_numpy(), rtol=1e-6, atol=1e-8)


@pytest.mark.slow
def test_mrmr_adds_no_unary_credit_on_a_linear_target():
    """The control: a lognormal feature with a linear target gets no ``unary_credit`` column."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _frame("linear")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(verbose=0, random_state=0, n_jobs=1, fe_max_steps=1).fit(X, y)
    assert not [r for r in (m._engineered_recipes_ or []) if r.kind == "unary_credit"]


def _multi_frame(with_copies: bool, n: int = 8000, seed: int = 0):
    """Four signal features (sin, log, square and a linear one) among four noise columns, optionally with four correlated (0.7 to 0.9, not identical) copies of the signal columns."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-6, 6, n)
    x2 = rng.lognormal(0, 1, n)
    x3 = rng.standard_normal(n)
    x4 = rng.standard_normal(n)
    y = np.sin(x1) + np.log(x2) + x3**2 + 0.5 * x4 + 0.3 * rng.standard_normal(n)
    cols = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, **{f"n{k}": rng.standard_normal(n) for k in range(4)}}
    if with_copies:

        def z(v):
            """Standardised copy."""
            return (v - v.mean()) / v.std()

        cols["c1"] = 0.8 * z(x1) + 0.6 * rng.standard_normal(n)
        cols["c2"] = 0.8 * z(np.log(x2)) + 0.6 * rng.standard_normal(n)
        cols["c3"] = 0.7 * z(x3) + 0.7 * rng.standard_normal(n)
        cols["c4"] = 0.9 * z(x4) + 0.44 * rng.standard_normal(n)
    return pd.DataFrame(cols), pd.Series(y, name="y")


@pytest.mark.slow
@pytest.mark.parametrize("with_copies", [False, True])
def test_several_signal_features_and_correlated_copies(with_copies):
    """Several signal features, each with its own link: every link is named, the output is at the noise floor in MAE (0.24; the raw signal columns give 1.2) and none of the correlated copies (0.7 to 0.9 with a signal
    column, not identical) is in it."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split

    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _multi_frame(with_copies)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(verbose=0, random_state=0, n_jobs=1, fe_max_steps=1).fit(Xtr, ytr)
    names = list(m.get_feature_names_out())
    assert any("sin(x1)" in n for n in names), names
    assert any("log(x2)" in n for n in names), names
    assert "sq0(x3)" in names, names
    assert not [n for n in names if "c1" in n or "c2" in n or "c3" in n or "c4" in n], f"a correlated copy reached the output: {names}"
    assert len(names) <= 6, names
    train = pd.DataFrame(m.transform(Xtr)).reset_index(drop=True).fillna(0)
    test = pd.DataFrame(m.transform(Xte)).reset_index(drop=True).fillna(0)
    mae = float(np.abs(yte.to_numpy() - RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(train, ytr.to_numpy()).predict(test)).mean())
    assert mae < 0.29, f"ridge MAE {mae:.3f} on the output {names} is above the noise floor 0.24"
