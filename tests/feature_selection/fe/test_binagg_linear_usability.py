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
from scipy.stats import beta
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

from mlframe.feature_selection.filters._binned_numeric_agg_fe import binned_numeric_agg_with_recipes

_SPREAD_COLUMN = "binagg_std(aux|qbin(g))"
# Policy values for the seed panel below, not tuned to the current result: the check must find the spread column in at least 70 percent of panel
# runs with 90 percent two-sided confidence, and the rate of runs that gain a noise aggregate must not be shown to exceed the nominal family-wise level.
_POWER_FLOOR = 0.70
_NOMINAL_FAMILY_WISE = 0.05
_PANEL_SEEDS = tuple(range(10))


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


def _clopper_pearson_lower(successes: int, trials: int, confidence: float = 0.90) -> float:
    """Lower end of the two-sided Clopper-Pearson interval for ``successes`` out of ``trials``."""
    return 0.0 if successes == 0 else float(beta.ppf((1.0 - confidence) / 2.0, successes, trials - successes + 1))


def test_power_and_calibration_over_a_seed_panel():
    """The check finds the spread column in enough runs for the lower confidence bound to clear the power floor, and the runs in which it ADDS a noise aggregate are not above the nominal rate.

    Only columns the check adds on top of the CMI gate count: the CMI gate keeps a noise aggregate on its own in some runs, which is
    outside what this test measures.
    """
    found, noisy_runs = 0, []
    for seed in _PANEL_SEEDS:
        rng = np.random.default_rng(100 + seed)
        n = 12_000
        g = rng.uniform(0, 1, n)
        sigma = 0.5 + 2.0 * np.abs(g - 0.5)
        X = pd.DataFrame({f"n{i}": rng.standard_normal(n) for i in range(6)})
        X["g"] = g
        X["aux"] = rng.normal(0, sigma, n)
        y = (sigma + 0.1 * rng.standard_normal(n) > sigma.mean()).astype(np.int64)
        _, kept_on, _ = binned_numeric_agg_with_recipes(X, y)
        _, kept_off, _ = binned_numeric_agg_with_recipes(X, y, linear_usability=False)
        found += int(_SPREAD_COLUMN in kept_on)
        added_noise = [c for c in kept_on if c not in kept_off and re.search(r"\bn\d+\b", c)]
        if added_noise:
            noisy_runs.append((seed, added_noise))
    power_lower = _clopper_pearson_lower(found, len(_PANEL_SEEDS))
    assert power_lower >= _POWER_FLOOR, f"spread column found in {found} of {len(_PANEL_SEEDS)} runs, 90 percent lower bound {power_lower:.2f} < {_POWER_FLOOR}"
    noise_lower = _clopper_pearson_lower(len(noisy_runs), len(_PANEL_SEEDS))
    assert noise_lower <= _NOMINAL_FAMILY_WISE, f"noise aggregates added in {len(noisy_runs)} of {len(_PANEL_SEEDS)} runs, 90 percent lower bound {noise_lower:.2f} above {_NOMINAL_FAMILY_WISE}: {noisy_runs}"


def test_the_cmi_gate_alone_does_not_keep_noise_aggregates_of_an_informative_group_above_the_nominal_rate():
    """Without the linear check, the share of runs that keep an aggregate of a pure-noise column is not shown to exceed the nominal family-wise level.

    The permutation null of the conditional MI must shuffle the target within the strata of the conditioning bins: shuffled over all rows it
    loses the target's dependence on the group, its mean sits about four times lower than for a candidate that is a function of the group's
    partition, and 7 of 30 runs kept a noise aggregate.
    """
    noisy_runs = 0
    seeds = range(30)
    for seed in seeds:
        rng = np.random.default_rng(100 + seed)
        n = 12_000
        g = rng.uniform(0, 1, n)
        sigma = 0.5 + 2.0 * np.abs(g - 0.5)
        X = pd.DataFrame({f"n{i}": rng.standard_normal(n) for i in range(6)})
        X["g"] = g
        X["aux"] = rng.normal(0, sigma, n)
        y = (sigma + 0.1 * rng.standard_normal(n) > sigma.mean()).astype(np.int64)
        _, kept, _ = binned_numeric_agg_with_recipes(X, y, linear_usability=False)
        noisy_runs += int(any(re.search(r"\bn\d+\b", c) for c in kept))
    lower = _clopper_pearson_lower(noisy_runs, len(seeds))
    assert lower <= _NOMINAL_FAMILY_WISE, f"noise aggregates kept in {noisy_runs} of {len(seeds)} runs, 90 percent lower bound {lower:.2f} above {_NOMINAL_FAMILY_WISE}"
