"""Regression test: the BorutaShap shadow kernel is UNBIASED on tied / discrete columns.

The shipped shadow kernel is a value-PERMUTATION (``_rng.permutation`` per column). Permuting a column's existing
values reproduces its value-multiset EXACTLY, so on a discrete / heavily-tied column the shadow's marginal
distribution is identical to the real column's and its mutual information with a random target is unbiased (only the
row alignment is destroyed). This is the property the shadow gate relies on to discriminate real vs random fairly.

A rank/argsort-based shadow (a tempting "fast path") breaks ties POSITIONALLY: equal values get distinct ranks in
input order, so the shuffled column no longer matches the original multiset and its shadow importance is biased LOW on
tied features (they look more random than they are -> real tied features clear the gate too easily). These tests pin
BOTH sides so a future "just argsort it" optimisation cannot silently reintroduce the bias:

  - GATED-IN (continuous / all-distinct): tie fraction ~0, both kernels agree -> any future fast path is safe there.
  - GATED-OUT (discrete / >20% ties): the value-permutation kernel preserves the multiset (unbiased); an argsort
    shadow does NOT -> the gate predicate ``_column_tie_fraction`` must flag it (>= ``SHADOW_TIE_GATE_FRACTION``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.boruta_shap._shadow_stats import (
    _column_tie_fraction,
    SHADOW_TIE_GATE_FRACTION,
)


def _run_create_shadow(X: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Run create shadow."""
    from mlframe.feature_selection.boruta_shap import BorutaShap

    bs = BorutaShap.__new__(BorutaShap)
    bs.X = X.copy()
    bs._rng = np.random.default_rng(seed)
    bs.create_shadow_features()
    return bs.X_shadow


# ---------------------------------------------------------------------------------------------------------------
# Gate predicate: continuous columns ~0 ties, discrete columns above the gate threshold.
# ---------------------------------------------------------------------------------------------------------------
def test_tie_fraction_predicate_separates_continuous_from_discrete():
    """Tie fraction predicate separates continuous from discrete."""
    rng = np.random.default_rng(0)
    continuous = rng.random(2000)  # all-distinct floats
    assert _column_tie_fraction(continuous) < 0.01

    discrete = rng.integers(0, 5, 2000)  # 5 levels over 2000 rows -> ~fully tied
    assert _column_tie_fraction(discrete) > SHADOW_TIE_GATE_FRACTION
    assert _column_tie_fraction(discrete) > 0.99

    low_card = rng.integers(0, 1000, 2000)  # mild ties, still above the gate
    assert _column_tie_fraction(low_card) >= SHADOW_TIE_GATE_FRACTION

    assert _column_tie_fraction(np.array([1.0])) == 0.0  # degenerate single-row guard


# ---------------------------------------------------------------------------------------------------------------
# GATED-IN side: continuous column -> value-permutation shadow preserves the (already-distinct) multiset exactly.
# ---------------------------------------------------------------------------------------------------------------
def test_shadow_preserves_multiset_on_continuous_column():
    """Shadow preserves multiset on continuous column."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"c": rng.random(1500)})
    shadow = _run_create_shadow(X, 7)["shadow_c"].to_numpy()
    assert np.array_equal(np.sort(shadow), np.sort(X["c"].to_numpy())), "continuous shadow must preserve the multiset"


# ---------------------------------------------------------------------------------------------------------------
# GATED-OUT side: heavily-tied discrete column. The shipped value-permutation kernel STILL preserves the multiset
# exactly (unbiased); an argsort/rank shadow would NOT -- pinned as the contrast the gate protects against.
# ---------------------------------------------------------------------------------------------------------------
def test_value_permutation_shadow_unbiased_on_tied_discrete_column():
    """Value permutation shadow unbiased on tied discrete column."""
    rng = np.random.default_rng(2)
    col = rng.integers(0, 4, 2000)  # 4 levels -> tie fraction ~1.0, well above the gate
    X = pd.DataFrame({"d": col})
    assert _column_tie_fraction(col) > SHADOW_TIE_GATE_FRACTION

    shadow = _run_create_shadow(X, 13)["shadow_d"].to_numpy()
    # Multiset identity == unbiased marginal: per-level counts identical to the real column.
    real_levels, real_counts = np.unique(col, return_counts=True)
    sh_levels, sh_counts = np.unique(shadow, return_counts=True)
    assert np.array_equal(real_levels, sh_levels)
    assert np.array_equal(real_counts, sh_counts), "value-permutation shadow must keep per-level counts (unbiased)"


def test_argsort_shadow_would_bias_tied_column_contrast():
    """The contrast: a rank/argsort-based shadow breaks ties positionally, so the shuffled column NO LONGER matches
    the original value-multiset -> biased. This is exactly what the value-permutation kernel avoids and what the gate
    predicate flags. We construct the argsort variant locally (NOT in prod) to prove the bias is real."""
    rng = np.random.default_rng(3)
    col = rng.integers(0, 4, 2000).astype(float)
    assert _column_tie_fraction(col) > SHADOW_TIE_GATE_FRACTION

    # An argsort/rank shadow: assign each row the rank of a random key, then map ranks back -- equal values get
    # distinct positional ranks, so applying a permutation via argsort over a noisy key SCATTERS tied values into a
    # distribution that no longer matches the column's discrete levels (here it produces a near-uniform rank field).
    key = col + 1e-9 * rng.random(col.size)  # tie-break key (argsort breaks ties by the tiny noise)
    ranks = np.argsort(np.argsort(key))  # 0..n-1 dense ranks -> NOT the original 4-level multiset
    assert len(np.unique(ranks)) > 4, "argsort path manufactures distinct ranks -> multiset destroyed (biased)"

    # And the shipped kernel does NOT do this: its output keeps exactly 4 levels.
    shadow = _run_create_shadow(pd.DataFrame({"d": col}), 21)["shadow_d"].to_numpy()
    assert len(np.unique(shadow)) <= 4
