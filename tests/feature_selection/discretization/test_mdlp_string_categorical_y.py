"""Wave 9.1 loop-iter-50 regression: ``mdlp_bin_edges`` MUST handle
non-numeric y dtypes (string, object, pandas Categorical).

Pre-fix at ``supervised_binning.py:62``::

    y = np.asarray(y).ravel().astype(np.int64)

This raw int64 cast crashed with::

    ValueError: invalid literal for int() with base 10: 'yes'

on any classifier user passing string labels (the standard sklearn
convention) or pandas Categorical (LabelEncoder / .astype('category')
output). MRMR's default ``nbins_strategy="mdlp"`` makes this the
production path, so default config crashed on string y.

Error pointed into MDLP internals, not into y dtype, so caller
debugging was hard.

Severity: high. Any classifier user with string labels hits this on
default config.

Fix at supervised_binning.py:61 (~14 LOC): detect object/string/
categorical dtype, factorize via pandas (or numpy unique fallback)
to integer class IDs before the int64 cast. MDLP only needs class
identity at split-purity computation, so factorize is sufficient and
order-preserving for already-integer inputs.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _frame():
    rng = np.random.default_rng(7)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 4)),
                      columns=[f"f{i}" for i in range(4)])
    return X


def test_mrmr_fit_with_string_y_does_not_crash():
    """The iter-50 contract: string y (sklearn classifier convention)
    must fit cleanly on default config.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    X = _frame()
    y = pd.Series(np.where(X["f0"] > 0, "yes", "no"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(
            quantization_nbins=8, full_npermutations=2,
            baseline_npermutations=2, fe_max_steps=0, verbose=0,
        ).fit(X, y)
    assert sel.support_ is not None


def test_mrmr_fit_with_categorical_y_does_not_crash():
    """``pd.Categorical`` y (LabelEncoder / .astype('category')
    output) must also work.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    X = _frame()
    y_str = pd.Series(np.where(X["f0"] > 0, "yes", "no"))
    y_cat = pd.Series(pd.Categorical(y_str))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(
            quantization_nbins=8, full_npermutations=2,
            baseline_npermutations=2, fe_max_steps=0, verbose=0,
        ).fit(X, y_cat)
    assert sel.support_ is not None


def test_mdlp_bin_edges_string_y_directly():
    """Direct call into ``mdlp_bin_edges`` with string y must factorize
    internally rather than crashing on ``.astype(int64)``.
    """
    from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges
    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal(n)
    y = np.where(x > 0, "high", "low")
    edges = mdlp_bin_edges(x, y)
    assert isinstance(edges, np.ndarray)
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf


def test_mdlp_bin_edges_integer_y_unchanged():
    """Negative control: integer y still works identically (factorize
    is a no-op for already-integer inputs)."""
    from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges
    rng = np.random.default_rng(1)
    n = 200
    x = rng.standard_normal(n)
    y = (x > 0).astype(np.int64)
    edges = mdlp_bin_edges(x, y)
    assert isinstance(edges, np.ndarray)


def test_mdlp_bin_edges_pandas_categorical_y():
    """pandas Categorical (with .codes) accepted via factorize."""
    from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges
    rng = np.random.default_rng(2)
    n = 200
    x = rng.standard_normal(n)
    y_str = np.where(x > 0, "A", "B")
    y_cat = pd.Categorical(y_str)
    # Pass underlying array; the factorize branch handles it.
    edges = mdlp_bin_edges(x, np.asarray(y_cat))
    assert isinstance(edges, np.ndarray)
