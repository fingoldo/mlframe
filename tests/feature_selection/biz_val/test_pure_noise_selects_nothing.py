"""Pure noise selects nothing: no feature survives on a target independent of every input, whatever the input distribution.

The inputs span the shapes a real frame carries (Gaussian, uniform, heavy-tailed, skewed, binary, categorical, near-constant,
id-like), because a selector that controls false positives only on Gaussian columns fails on the others. The target is independent
of every column by construction, so any selected feature is a false positive of the selector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR

N = 2000
P = 8


def _column(kind: str, rng: np.random.Generator, n: int) -> np.ndarray:
    """One input column drawn from the named distribution."""
    if kind == "normal":
        return rng.normal(size=n)
    if kind == "uniform":
        return rng.uniform(-1.0, 1.0, size=n)
    if kind == "student_t2":
        return rng.standard_t(df=2, size=n)
    if kind == "lognormal":
        return rng.lognormal(mean=0.0, sigma=1.5, size=n)
    if kind == "exponential":
        return rng.exponential(scale=1.0, size=n)
    if kind == "binary":
        return rng.integers(0, 2, size=n).astype(np.float64)
    if kind == "categorical":
        return rng.integers(0, 6, size=n).astype(np.float64)
    if kind == "near_constant":
        return np.where(rng.random(n) < 0.01, rng.normal(size=n), 0.0)
    if kind == "id_like":
        return rng.permutation(n).astype(np.float64)
    raise ValueError(kind)


DISTRIBUTIONS = ("normal", "uniform", "student_t2", "lognormal", "exponential", "binary", "categorical", "near_constant", "id_like")


def _frame(kind: str, seed: int) -> pd.DataFrame:
    """A frame of ``P`` columns all drawn from ``kind``; the ``id_like`` and ``categorical`` kinds are mixed with normal columns."""
    rng = np.random.default_rng(seed)
    cols = {f"c{j}": _column(kind, rng, N) for j in range(P)}
    if kind in ("id_like", "categorical"):
        cols["c_norm"] = rng.normal(size=N)
    return pd.DataFrame(cols)


@pytest.mark.parametrize("kind", DISTRIBUTIONS)
@pytest.mark.parametrize("target", ("binary", "continuous"))
def test_mrmr_selects_nothing_on_pure_noise(kind, target):
    """MRMR with default settings keeps no feature when the target is independent of every column."""
    seed = 11 + DISTRIBUTIONS.index(kind)
    X = _frame(kind, seed)
    rng = np.random.default_rng(seed + 1000)
    if target == "binary":
        y = (rng.random(N) > 0.5).astype(np.int64)
    else:
        y = rng.normal(size=N)
    sel = MRMR(verbose=0, random_state=0, n_jobs=1).fit(X, y)
    assert getattr(sel, "no_significant_features_", False) is True, f"no_significant_features_ not set on pure noise (inputs={kind}, target={target})"
