"""The composite-column MI of ``categorical_group_concat`` equals scikit-learn's ``discrete_features=True`` estimators, for a label target and for a continuous one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_engineering.categorical_group_concat import _composite_mi_with_target


def _levels(n, k, seed):
    """A categorical column of ``k`` levels as strings, with an uneven frequency table (one level of a single row)."""
    rng = np.random.default_rng(seed)
    codes = rng.choice(k, size=n, p=np.r_[np.full(k - 1, 0.9 / (k - 1)), 0.1])
    values = pd.Series(np.array([f"c{i}" for i in range(k)])[codes])
    values.iloc[0] = "solo"
    return values, codes


@pytest.mark.parametrize("seed", [0, 3])
def test_label_target_equals_mutual_info_classif(seed):
    """Two label vectors: the plug-in MI of their contingency table."""
    from sklearn.feature_selection import mutual_info_classif

    values, codes = _levels(2000, 6, seed)
    y = (codes + np.random.default_rng(seed + 1).integers(0, 2, size=2000)) % 3
    c, _ = pd.factorize(values)
    ref = float(mutual_info_classif(c.reshape(-1, 1), y, discrete_features=True, random_state=seed)[0])
    assert _composite_mi_with_target(values, y, seed) == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("seed", [0, 3])
def test_continuous_target_equals_mutual_info_regression(seed):
    """A discrete column against a continuous target: scikit-learn's k-NN estimator with the same noise stream, to 1e-10."""
    from sklearn.feature_selection import mutual_info_regression

    values, codes = _levels(2000, 5, seed)
    y = codes + 0.7 * np.random.default_rng(seed + 2).standard_normal(2000)
    c, _ = pd.factorize(values)
    ref = float(mutual_info_regression(c.reshape(-1, 1), y, discrete_features=True, random_state=seed)[0])
    assert _composite_mi_with_target(values, y, seed) == pytest.approx(ref, abs=1e-10)
