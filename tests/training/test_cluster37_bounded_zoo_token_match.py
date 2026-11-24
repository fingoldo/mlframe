"""Cluster 37 regression: the bounded-zoo check matches model names by token, not by substring (audit 2026-11-20, finding 37-F37).

The old check matched "ols" anywhere in a name, so ``lgb_colsample`` was classed as an unbounded linear model. "linear" stays a substring match
(``LinearRegression`` lowercases to ``linearregression``); "ols" and "lstsq" are whole tokens.
"""

from mlframe.training.core._ar_skip import _zoo_is_bounded_only


def test_colsample_is_not_an_ols_model():
    """A tree model whose name contains 'ols' inside 'colsample' stays bounded (37-F37)."""
    assert _zoo_is_bounded_only(["lgb_colsample"]) is True


def test_real_ols_token_is_unbounded():
    """A model named as an OLS token is unbounded (37-F37)."""
    assert _zoo_is_bounded_only(["ols"]) is False


def test_linear_regression_is_unbounded():
    """Control: the LinearRegression name still counts as unbounded (37-F37)."""
    assert _zoo_is_bounded_only(["LinearRegression"]) is False


def test_ridge_stays_bounded():
    """Control: Ridge is regularised and stays bounded (37-F37)."""
    assert _zoo_is_bounded_only(["Ridge"]) is True
