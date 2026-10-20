"""A binned-aggregate column the CMI redundancy gate rejects is kept when it lifts a linear fit, and not otherwise.

The column ``binagg_std(aux|qbin(g))`` is a function of the cell of ``g``. The CMI gate conditions on the same partition, so with
``aux ~ N(0, sigma(g))`` and a target that is a function of ``sigma(g)`` it rejects the column as redundant, although a ridge model on the raw
columns scores held-out R^2 of 0.00 and about 0.85 once the column is added. These tests pin the corrected behaviour and the cases where the
column must still be dropped.
"""

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

from mlframe.feature_selection.filters._binned_numeric_agg_fe import binned_numeric_agg_with_recipes

_SPREAD_COLUMN = "binagg_std(aux|qbin(g))"


def _sigma_frame(n: int, seed: int):
    """``g`` uniform, ``aux ~ N(0, 0.5 + 2|g - 0.5|)`` and a continuous target equal to that spread plus small noise."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(0, 1, n)
    sigma = 0.5 + 2.0 * np.abs(g - 0.5)
    X = pd.DataFrame({"g": g, "aux": rng.normal(0, sigma, n), "noise": rng.standard_normal(n)})
    return X, sigma + 0.1 * rng.standard_normal(n)


@pytest.mark.parametrize("seed", [0, 1])
def test_spread_column_survives_the_default_gates(seed):
    """The spread column survives the default gates."""
    X, y = _sigma_frame(12_000, seed)
    _, kept, _ = binned_numeric_agg_with_recipes(X, y)
    assert _SPREAD_COLUMN in kept


def test_aggregates_of_noise_columns_are_not_kept_when_the_group_is_informative():
    """Every aggregate is a function of the group's cell, so with an informative group a noise column's per-cell values look useful to a null that ignores cells."""
    rng = np.random.default_rng(5)
    n = 20_000
    g = rng.uniform(0, 1, n)
    sigma = 0.5 + 2.0 * np.abs(g - 0.5)
    X = pd.DataFrame({f"n{i}": rng.standard_normal(n) for i in range(10)})
    X["g"] = g
    X["aux"] = rng.normal(0, sigma, n)
    y = (sigma + 0.1 * rng.standard_normal(n) > sigma.mean()).astype(np.int64)
    _, kept, _ = binned_numeric_agg_with_recipes(X, y)
    assert _SPREAD_COLUMN in kept
    assert [c for c in kept if re.search(r"\bn\d+\b", c)] == [], "no aggregate that involves a pure-noise column may be kept"


def test_kept_columns_lift_a_linear_model_out_of_fold():
    """Cross-validated ridge R^2 on the raw columns is near zero and is large with the kept out-of-fold columns."""
    X, y = _sigma_frame(12_000, seed=2)
    X_aug, kept, _ = binned_numeric_agg_with_recipes(X, y)
    assert kept, "the sigma(cell) fixture must keep a column, or the comparison below is vacuous"
    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    raw_r2 = cross_val_score(Ridge(), X.to_numpy(), y, cv=cv, scoring="r2").mean()
    aug_r2 = cross_val_score(Ridge(), X_aug.to_numpy(), y, cv=cv, scoring="r2").mean()
    assert raw_r2 < 0.05
    assert aug_r2 > 0.5


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_target_unrelated_to_the_features_keeps_nothing(seed):
    """No column is kept on pure noise, however many candidates the wide frame offers."""
    rng = np.random.default_rng(seed)
    n = 8_000
    X = pd.DataFrame({f"n{i}": rng.standard_normal(n) for i in range(10)})
    y = rng.integers(0, 2, n)
    _, kept, _ = binned_numeric_agg_with_recipes(X, y)
    assert kept == []


def test_a_linear_target_the_raw_columns_explain_keeps_nothing():
    """When a linear model on the raw columns already explains the target, the aggregates add nothing and are dropped."""
    rng = np.random.default_rng(3)
    n = 10_000
    X = pd.DataFrame({"g": rng.uniform(0, 1, n), "aux": rng.standard_normal(n), "noise": rng.standard_normal(n)})
    y = X["g"].to_numpy() + 0.5 * X["aux"].to_numpy() + 0.1 * rng.standard_normal(n)
    _, kept, _ = binned_numeric_agg_with_recipes(X, y)
    assert kept == []
