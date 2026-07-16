"""Regression test: multi-output (union/intersect) fits must populate the same diagnostic
attribute surface as single-target fits.

``_fit_multioutput`` returns before the legacy single-fit body runs, so ``degenerate_columns_``,
``provenance_``, ``fe_provenance_``, and ``fe_rejection_ledger_`` (all documented public
fitted attributes) used to simply never be set on a multi-output-fitted estimator -- any
downstream code reading them (e.g. a report generator that unconditionally does
``mrmr.degenerate_columns_``) raised ``AttributeError``, but only for 2D-y fits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR


def _fast(**kw):
    """Build a fast-fitting MRMR instance for these tests, overridable via kwargs."""
    base = dict(full_npermutations=5, baseline_npermutations=3, n_jobs=1, verbose=0, fe_fast_search=False, interactions_max_order=1, random_seed=9)
    base.update(kw)
    return MRMR(**base)


def _xy_2d_y(seed: int = 5, n: int = 160):
    """Build a small synthetic 2D-y fixture (two binary targets driven by columns 0 and 2)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    Y2 = np.column_stack([(X[:, 0] > 0).astype(int), (X[:, 2] > 0).astype(int)])
    return X, Y2


@pytest.mark.parametrize("strategy", ["union", "intersect"])
def test_multioutput_fit_populates_degenerate_columns(strategy):
    """A multi-output fit must populate degenerate_columns_ like a single-target fit."""
    X, Y2 = _xy_2d_y()
    MRMR._FIT_CACHE.clear()
    m = _fast(multioutput_strategy=strategy).fit(X, Y2)
    assert hasattr(m, "degenerate_columns_")
    assert isinstance(m.degenerate_columns_, dict)


@pytest.mark.parametrize("strategy", ["union", "intersect"])
def test_multioutput_fit_populates_provenance(strategy):
    """A multi-output fit must populate provenance_ like a single-target fit."""
    X, Y2 = _xy_2d_y()
    MRMR._FIT_CACHE.clear()
    m = _fast(multioutput_strategy=strategy).fit(X, Y2)
    assert hasattr(m, "provenance_")
    assert m.provenance_ is not None
    assert m.provenance_["step"] == "mrmr_multioutput"
    assert m.provenance_["n_rows"] == X.shape[0]


@pytest.mark.parametrize("strategy", ["union", "intersect"])
def test_multioutput_fit_populates_fe_provenance(strategy):
    """A multi-output fit must populate fe_provenance_ (raw-only, no engineered features)."""
    X, Y2 = _xy_2d_y()
    MRMR._FIT_CACHE.clear()
    m = _fast(multioutput_strategy=strategy).fit(X, Y2)
    assert hasattr(m, "fe_provenance_")
    assert isinstance(m.fe_provenance_, pd.DataFrame)
    # multi-output aggregation never unions engineered features (raw-only per the docstring),
    # so every row's origin must be "raw".
    if len(m.fe_provenance_):
        assert set(m.fe_provenance_["origin"].unique()) == {"raw"}


@pytest.mark.parametrize("strategy", ["union", "intersect"])
def test_multioutput_fit_populates_fe_rejection_ledger(strategy):
    """A multi-output fit must populate fe_rejection_ledger_ like a single-target fit."""
    X, Y2 = _xy_2d_y()
    MRMR._FIT_CACHE.clear()
    m = _fast(multioutput_strategy=strategy).fit(X, Y2)
    assert hasattr(m, "fe_rejection_ledger_")
    assert isinstance(m.fe_rejection_ledger_, pd.DataFrame)
