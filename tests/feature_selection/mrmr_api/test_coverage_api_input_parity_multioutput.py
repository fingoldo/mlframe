"""Input-format parity + multioutput-path API coverage for MRMR.

Two gaps addressed:

1. Input parity: a fit on pandas DataFrame, polars DataFrame, and a bare ndarray of the
   SAME underlying values + same seed must select the SAME features (by position). The
   selector core promises format-independence; nothing pinned it at the public-API level.

2. Multioutput path: ``multioutput_strategy`` in {'union','intersect'} drives the
   per-column sub-fit aggregation (sets ``multioutput_strategy_`` + ``multioutput_supports_``),
   while 'joint'/None treat a 2D y as a single joint target (legacy single-target path).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR

polars = pytest.importorskip("polars")


_COLS = [f"f{i}" for i in range(5)]


def _xy(seed: int = 5, n: int = 160):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    # signal on f0 and f2 so a non-empty, stable selection exists
    y = (X[:, 0] + 0.4 * X[:, 2] > 0).astype(np.int32)
    return X, y


def _fast(**kw):
    base = dict(full_npermutations=5, baseline_npermutations=3, n_jobs=1, verbose=0,
                fe_fast_search=False, interactions_max_order=1, random_seed=9)
    base.update(kw)
    return MRMR(**base)


def _fit_support(Xin, y):
    MRMR._FIT_CACHE.clear()
    m = _fast().fit(Xin, y)
    return np.sort(np.asarray(m.support_))


def test_pandas_polars_ndarray_select_same_features():
    X, y = _xy()
    sup_pd = _fit_support(pd.DataFrame(X, columns=_COLS), y)
    sup_np = _fit_support(X, y)
    sup_pl = _fit_support(polars.DataFrame(X, schema=_COLS), y)
    np.testing.assert_array_equal(sup_pd, sup_np)
    np.testing.assert_array_equal(sup_pd, sup_pl)
    assert sup_pd.size >= 1


def test_pandas_polars_names_out_agree():
    """get_feature_names_out (raw portion) matches between pandas and polars input."""
    X, y = _xy()
    MRMR._FIT_CACHE.clear()
    m_pd = _fast().fit(pd.DataFrame(X, columns=_COLS), y)
    MRMR._FIT_CACHE.clear()
    m_pl = _fast().fit(polars.DataFrame(X, schema=_COLS), y)
    # Compare the RAW selected names (engineered names may differ if FE differs; the raw
    # selection is what parity guarantees).
    raw_pd = [n for n in m_pd.get_feature_names_out() if "(" not in str(n)]
    raw_pl = [n for n in m_pl.get_feature_names_out() if "(" not in str(n)]
    assert raw_pd == raw_pl


def test_multioutput_union_aggregates_per_column_supports():
    X, _ = _xy()
    Y2 = np.column_stack([(X[:, 0] > 0).astype(int), (X[:, 2] > 0).astype(int)])
    ydf = pd.DataFrame(Y2, columns=["y0", "y1"])
    MRMR._FIT_CACHE.clear()
    m = _fast(multioutput_strategy="union").fit(pd.DataFrame(X, columns=_COLS), ydf)
    assert m.multioutput_strategy_ == "union"
    assert set(m.multioutput_supports_.keys()) == {"y0", "y1"}
    # union support is the union of the per-column raw selections (by name -> index).
    union_names = set()
    for names in m.multioutput_supports_.values():
        union_names.update(names)
    sel_names = {str(m.feature_names_in_[i]) for i in np.asarray(m.support_, dtype=int)}
    assert union_names == sel_names
    assert m.n_features_in_ == X.shape[1]


def test_multioutput_intersect_is_subset_of_union():
    X, _ = _xy()
    Y2 = np.column_stack([(X[:, 0] > 0).astype(int), (X[:, 2] > 0).astype(int)])
    ydf = pd.DataFrame(Y2, columns=["y0", "y1"])

    MRMR._FIT_CACHE.clear()
    m_u = _fast(multioutput_strategy="union").fit(pd.DataFrame(X, columns=_COLS), ydf)
    MRMR._FIT_CACHE.clear()
    m_i = _fast(multioutput_strategy="intersect").fit(pd.DataFrame(X, columns=_COLS), ydf)

    assert m_i.multioutput_strategy_ == "intersect"
    sup_u = set(np.asarray(m_u.support_, dtype=int).tolist())
    sup_i = set(np.asarray(m_i.support_, dtype=int).tolist())
    assert sup_i.issubset(sup_u)


def test_multioutput_invalid_strategy_raises():
    X, _ = _xy()
    Y2 = np.column_stack([(X[:, 0] > 0).astype(int), (X[:, 2] > 0).astype(int)])
    ydf = pd.DataFrame(Y2, columns=["y0", "y1"])
    MRMR._FIT_CACHE.clear()
    m = _fast(multioutput_strategy="not_a_strategy")
    with pytest.raises(ValueError):
        m.fit(pd.DataFrame(X, columns=_COLS), ydf)


def test_multioutput_joint_treats_2d_y_as_single_target():
    """strategy='joint' does NOT run the per-column union path; no multioutput_strategy_ attr
    is set (the 2D y is collapsed to a single joint target via the legacy single-fit path)."""
    X, _ = _xy()
    Y2 = np.column_stack([(X[:, 0] > 0).astype(int), (X[:, 2] > 0).astype(int)])
    ydf = pd.DataFrame(Y2, columns=["y0", "y1"])
    MRMR._FIT_CACHE.clear()
    m = _fast(multioutput_strategy="joint").fit(pd.DataFrame(X, columns=_COLS), ydf)
    assert not hasattr(m, "multioutput_strategy_")
    assert np.asarray(m.support_).size >= 1
