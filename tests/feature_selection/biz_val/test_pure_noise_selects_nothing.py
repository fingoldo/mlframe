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
from tests.feature_selection.binning_cases import binning_kwargs

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


# Production binning strategies the guarantee must hold under. Research-only strategies (knuth, blocks, ...) are excluded by design.
BINNING_CASES = ("fixed", "rice_capped", "uniform", "mdlp", "mdlp_validated")


def _false_significant(kind: str, target: str, binning_case: str, seed: int, **mrmr_kwargs) -> bool:
    """Whether MRMR keeps a feature as significant on a target independent of every column of one frame (the fit reports ``no_significant_features_`` otherwise)."""
    X = _frame(kind, seed)
    rng = np.random.default_rng(seed + 1000)
    y = (rng.random(N) > 0.5).astype(np.int64) if target == "binary" else rng.normal(size=N)
    sel = MRMR(verbose=0, random_state=0, n_jobs=1, **binning_kwargs(binning_case), **mrmr_kwargs).fit(X, y)
    return getattr(sel, "no_significant_features_", False) is not True


#: The guarantee is a family-wise level of ``FAMILY_ALPHA = 0.01`` per fit, not "never": one frame per distribution is one draw at that level, so a group of nine distributions may hold one false-significant fit
#: (the chance of two or more at the level 0.01 is 0.3 percent) and the calibration test below measures the rate on many frames.
MAX_FALSE_SIGNIFICANT_PER_GROUP = 1


@pytest.mark.parametrize("binning_case", BINNING_CASES)
@pytest.mark.parametrize("target", ("binary", "continuous"))
def test_mrmr_selects_nothing_on_pure_noise(target, binning_case):
    """MRMR keeps no feature when the target is independent of every column, under every production binning strategy: at most one of the nine input distributions is a false-significant fit (the level is 0.01 per fit)."""
    false_significant = [kind for kind in DISTRIBUTIONS if _false_significant(kind, target, binning_case, 11 + DISTRIBUTIONS.index(kind))]
    assert len(false_significant) <= MAX_FALSE_SIGNIFICANT_PER_GROUP, f"false-significant fits on pure noise (target={target}, binning={binning_case}): {false_significant}"


@pytest.mark.parametrize("kind,target,binning_case", [("uniform", "binary", "mdlp"), ("lognormal", "continuous", "uniform"), ("normal", "binary", "fixed")])
def test_the_false_significant_rate_on_pure_noise_is_at_the_family_level(kind, target, binning_case):
    """Thirty frames per cell (the screen alone, no feature engineering): at most two false-significant fits at the family-wise level 0.01 (the chance of three or more is 0.3 percent). Measured 0 of 30 in each cell."""
    false_significant = sum(_false_significant(kind, target, binning_case, 100 + k, fe_max_steps=0) for k in range(30))
    assert false_significant <= 2, false_significant
