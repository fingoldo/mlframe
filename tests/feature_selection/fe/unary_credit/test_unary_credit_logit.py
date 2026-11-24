"""``credit_columns_binary``: the logit-deviance credit of a unary transform for a binary target (research tt_b1: 126 of 126 judge-best transforms recognised, 0 harmful accepts, 3 of 78 control false positives)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _unary_credit_logit
from mlframe.feature_selection.filters._unary_credit_logit import credit_columns_binary

N = 6000


def _sigmoid(z):
    """Logistic function."""
    return 1.0 / (1.0 + np.exp(-z))


def _binary(case: str, seed: int = 0):
    """``(X with x1 and x2, y in {0, 1})`` of a case."""
    rng = np.random.default_rng(seed)
    x2 = rng.standard_normal(N)
    if case == "log":
        x1 = rng.lognormal(0, 1, N)
        eta = 2.0 * np.log(x1) + x2
    elif case == "square":
        x1 = rng.standard_normal(N)
        eta = 1.5 * (x1**2 - 1.0) + x2
    elif case == "sin":
        x1 = rng.uniform(-6, 6, N)
        eta = 2.5 * np.sin(x1) + x2
    elif case == "linear_lognormal":
        x1 = rng.lognormal(0, 1, N)
        eta = 0.8 * x1 + x2 - 1.0
    elif case == "linear":
        x1 = rng.standard_normal(N)
        eta = 1.2 * x1 + x2
    else:
        raise ValueError(case)
    return np.column_stack([x1, x2]), (rng.random(N) < _sigmoid(eta)).astype(np.float64)


@pytest.mark.parametrize("case,expected", [("log", "log"), ("square", "sq0"), ("sin", "sin")])
def test_the_exact_form_of_a_logit_is_credited(case, expected):
    """The class code of a sigmoid of ``f(x1) + x2``: the exact ``f`` is credited on x1 only (the simplest of the tied forms), with a held-out gain in nats per row."""
    X, y = _binary(case)
    verdicts = credit_columns_binary(X, y, seed=0)
    assert verdicts[0].form is not None and verdicts[0].form.name == expected, verdicts[0]
    assert verdicts[0].replicated and verdicts[0].effect > 0.0
    assert verdicts[1].form is None


@pytest.mark.parametrize("case", ["linear", "linear_lognormal"])
def test_a_linear_logit_credits_nothing(case):
    """Controls: a plain linear logit on a normal or a lognormal feature (the logit of a gamma-scale or Poisson law is exactly linear in x, too)."""
    X, y = _binary(case)
    assert [v.form for v in credit_columns_binary(X, y, seed=1)] == [None, None]


def test_unusable_targets_credit_nothing():
    """A multi-valued target, a rare class (under 2 percent) and too few rows are refused without an exception."""
    X, y = _binary("log")
    assert all(v.form is None for v in credit_columns_binary(X, y + (np.arange(N) % 5 == 0)))
    rare = np.zeros(N)
    rare[:50] = 1.0
    assert all(v.form is None for v in credit_columns_binary(X, rare))
    assert all(v.form is None for v in credit_columns_binary(X[:300], y[:300]))


@pytest.mark.parametrize("case,expected", [("log", "log"), ("sin", "sin")])
def test_the_host_and_the_device_null_name_the_same_form(case, expected, monkeypatch):
    """The bootstrap null runs on the GPU for large problems and on the host otherwise (``MLFRAME_FE_CREDIT_GPU=0`` forces the host): both name the generating form and credit nothing on the noise column."""
    X, y = _binary(case)
    monkeypatch.setenv("MLFRAME_FE_CREDIT_GPU", "0")
    assert _unary_credit_logit._device_module() is None
    host = credit_columns_binary(X, y, seed=0)
    monkeypatch.delenv("MLFRAME_FE_CREDIT_GPU")
    device_available = _unary_credit_logit._device_module() is not None
    monkeypatch.setattr(_unary_credit_logit, "GPU_MIN_CELLS", 1)  # every problem is large enough for the device
    device = credit_columns_binary(X, y, seed=0)
    for verdicts in (host, device):
        assert verdicts[0].form is not None and verdicts[0].form.name == expected, verdicts[0]
        assert verdicts[1].form is None
    if not device_available:
        pytest.skip("no CUDA device: the host null ran in both calls")


def test_a_second_pass_runs_only_when_the_first_one_credited_something(monkeypatch):
    """With nothing credited the design of every column of the second pass equals the first one's, so the pass is not run: one discovery test per column on a control, and more when a form is credited."""
    calls = []
    original = _unary_credit_logit._test_column

    def counting(*args, **kwargs):
        """The wrapped function, counted."""
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_unary_credit_logit, "_test_column", counting)
    X, y = _binary("linear")
    assert [v.form for v in credit_columns_binary(X, y, seed=1)] == [None, None]
    assert len(calls) == 2
    calls.clear()
    X, y = _binary("log")
    assert credit_columns_binary(X, y, seed=0)[0].form is not None
    assert len(calls) > 2


def test_an_unchanged_second_credit_call_of_a_fit_is_not_tested_again(monkeypatch):
    """The credit runs twice per fit (early and after the late restores); when the columns, the engineered recipes and the target are the same the second call repeats the first one's verdicts and returns without testing."""
    from types import SimpleNamespace

    from mlframe.feature_selection.filters._fe_unary_credit import credit_unary_forms

    X, y = _binary("log")
    frame = pd.DataFrame(X, columns=["x1", "x2"])
    mrmr = SimpleNamespace(_engineered_features_=[], _engineered_recipes_=[], feature_names_in_=["x1", "x2"], unary_forms_={}, unary_forms_tied_={}, _unary_credit_soft_=set())
    calls = []
    original = _unary_credit_logit.credit_columns_binary

    def counting(*args, **kwargs):
        """The wrapped function, counted."""
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_unary_credit_logit, "credit_columns_binary", counting)
    first = credit_unary_forms(mrmr, frame, y, ["x1", "x2"], seed=0)
    assert [name for _recipe, name in first] == ["log(x1)"] and len(calls) == 1
    assert credit_unary_forms(mrmr, frame, y, ["x1", "x2"], seed=0) == [] and len(calls) == 1
    mrmr._engineered_recipes_ = [first[0][0]]  # a recipe added in between changes the sources of the interaction stage: the credit is judged again
    mrmr._engineered_features_ = ["log(x1)"]
    credit_unary_forms(mrmr, frame, y, ["x1", "x2"], seed=0)
    assert len(calls) == 2
