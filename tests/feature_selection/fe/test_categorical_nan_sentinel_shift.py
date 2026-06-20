"""Wave 9.1 loop-iter-31 regression: categorical NaN encoding must NOT
emit raw ``-1`` codes from ``pd.factorize`` / ``.cat.codes``.

Pre-fix at ``discretization.py:1149-1189``: the categorical block
bypassed ``missing_strategy`` entirely. ``_multi_col_factorize_native``
and ``pd.factorize`` emit ``-1`` for NaN, which then flowed downstream
into:
- ``merge_vars`` (post-iter-21 ``np.zeros(expected_nclasses,
  dtype=np.int64)`` then ``freqs[newclass] += 1``) - negative class
  indices wrap to ``freqs[-1]``, silently merging NaN observations
  with the LAST real category.
- Direct-index joint histograms via the same negative-index wrap.
- ``nbins = data.max(axis=0) + 1`` at line 1189 reported one less bin
  than reality.

Net effect: NaN rows in pd.Categorical / object / string / bool
columns silently merged with the largest real category, biasing every
MI / SU / MRMR score on NaN-bearing categoricals.

Sibling of iter 9 (numeric NaN bin collision) and iter 11
(propagate strategy silent merge).

Fix at ``discretization.py:1156+``: shift the factorize output by +1
when any ``-1`` sentinel is present, so NaN -> 0 and real categories
become ``1..K``. Under ``missing_strategy='raise'``, refuse instead.
Auto-promote dtype guard below catches any new-max overflow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_categorical_nan_does_not_emit_negative_codes():
    """Codes for NaN rows must NOT be -1 (which downstream wraps to
    last real category).
    """
    from mlframe.feature_selection.filters.discretization import categorize_dataset
    cats = pd.Categorical(
        ["A", "B", "C", "D"] * 25 + [None] * 50,
        categories=["A", "B", "C", "D", "E"],
    )
    df = pd.DataFrame({"cat": cats})
    data, cols, nbins = categorize_dataset(
        df=df, n_bins=4, dtype=np.int16,
        missing_strategy="separate_bin",
    )
    codes = data[:, cols.index("cat")]
    assert (codes >= 0).all(), (
        f"categorical codes must be non-negative; got min={codes.min()}"
    )


def test_categorical_nan_separate_bin_disjoint_from_real():
    """Default ``separate_bin`` strategy: NaN must occupy its own bin
    disjoint from real category codes.
    """
    from mlframe.feature_selection.filters.discretization import categorize_dataset
    cats = pd.Categorical(
        ["A", "B", "C", "D"] * 25 + [None] * 50,
        categories=["A", "B", "C", "D"],
    )
    df = pd.DataFrame({"cat": cats})
    data, cols, nbins = categorize_dataset(
        df=df, n_bins=4, dtype=np.int16,
        missing_strategy="separate_bin",
    )
    codes = data[:, cols.index("cat")]
    real_codes = {int(c) for c in codes[~pd.isna(cats)]}
    nan_codes = {int(c) for c in codes[pd.isna(cats)]}
    assert not (real_codes & nan_codes), (
        f"NaN code collides with real categories: real={sorted(real_codes)}, "
        f"nan={nan_codes}"
    )


def test_categorical_nan_nbins_reports_correctly():
    """``nbins`` must include the dedicated NaN bin in the count."""
    from mlframe.feature_selection.filters.discretization import categorize_dataset
    # 4 real categories + 50 NaN -> 5 distinct codes -> nbins=5
    cats = pd.Categorical(
        ["A", "B", "C", "D"] * 25 + [None] * 50,
        categories=["A", "B", "C", "D"],
    )
    df = pd.DataFrame({"cat": cats})
    data, cols, nbins = categorize_dataset(
        df=df, n_bins=4, dtype=np.int16,
        missing_strategy="separate_bin",
    )
    assert int(nbins[cols.index("cat")]) == 5


def test_categorical_no_nan_unchanged():
    """Negative control: data without NaN must produce the same codes
    as before the fix (0..K-1).
    """
    from mlframe.feature_selection.filters.discretization import categorize_dataset
    cats = pd.Categorical(["A", "B", "C", "D"] * 25, categories=["A", "B", "C", "D"])
    df = pd.DataFrame({"cat": cats})
    data, cols, nbins = categorize_dataset(
        df=df, n_bins=4, dtype=np.int16,
        missing_strategy="separate_bin",
    )
    codes_set = {int(c) for c in data[:, cols.index("cat")]}
    assert codes_set == {0, 1, 2, 3}
    assert int(nbins[cols.index("cat")]) == 4


def test_categorical_raise_strategy_fires_on_nan():
    """``missing_strategy='raise'`` must reject NaN-containing
    categoricals.
    """
    from mlframe.feature_selection.filters.discretization import categorize_dataset
    cats = pd.Categorical(["A", "B"] * 5 + [None] * 2, categories=["A", "B"])
    df = pd.DataFrame({"cat": cats})
    with pytest.raises(ValueError, match="NaN"):
        categorize_dataset(
            df=df, n_bins=4, dtype=np.int16,
            missing_strategy="raise",
        )
