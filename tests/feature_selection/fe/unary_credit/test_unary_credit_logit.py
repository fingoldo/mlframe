"""``credit_columns_binary``: the logit-deviance credit of a unary transform for a binary target (research tt_b1: 126 of 126 judge-best transforms recognised, 0 harmful accepts, 3 of 78 control false positives)."""

from __future__ import annotations

import numpy as np
import pytest

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
