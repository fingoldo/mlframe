"""The same data fitted as pandas and as polars selects the same features with the same gains.

Polars input is bridged to an Arrow-backed pandas view when FE runs; the existing polars tests check that polars input is accepted and not
mutated, never that the two formats reach the same selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR

pl = pytest.importorskip("polars")


def _frames(seed=0, n=600):
    """A classification frame with two signal columns, one interaction pair and noise, as pandas and polars."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(6)})
    y = ((X["x0"] + 0.8 * X["x1"] + 0.6 * X["x2"] * X["x3"]) > 0).astype(np.int64)
    return X, pl.from_pandas(X), y


def _fit(X, y, **kw):
    """A deterministic light fit."""
    MRMR._FIT_CACHE.clear()
    base = dict(random_seed=7, n_jobs=1, verbose=0, full_npermutations=3, baseline_npermutations=2)
    base.update(kw)
    return MRMR(**base).fit(X, y)


def test_polars_and_pandas_fits_select_identical_support():
    """No FE: identical support, output names and gains."""
    X_pd, X_pl, y = _frames()
    m_pd = _fit(X_pd, y, fe_max_steps=0)
    m_pl = _fit(X_pl, y, fe_max_steps=0)
    assert np.array_equal(np.asarray(m_pd.support_), np.asarray(m_pl.support_))
    assert list(map(str, m_pd.get_feature_names_out())) == list(map(str, m_pl.get_feature_names_out()))
    assert np.allclose(np.asarray(m_pd.mrmr_gains_), np.asarray(m_pl.mrmr_gains_), rtol=1e-12, atol=0)


def test_polars_and_pandas_fits_identical_with_fe_enabled():
    """With FE on (the Arrow-bridge path): identical output names, including engineered recipes, in order."""
    X_pd, X_pl, y = _frames(seed=1)
    m_pd = _fit(X_pd, y, fe_max_steps=1)
    m_pl = _fit(X_pl, y, fe_max_steps=1)
    assert list(map(str, m_pd.get_feature_names_out())) == list(map(str, m_pl.get_feature_names_out()))
    assert [r.name for r in (m_pd._engineered_recipes_ or [])] == [r.name for r in (m_pl._engineered_recipes_ or [])]


def test_polars_and_pandas_transform_emit_identical_values():
    """The two fitted selectors transform their own input format to the same numbers."""
    X_pd, X_pl, y = _frames(seed=2)
    m_pd = _fit(X_pd, y, fe_max_steps=1)
    m_pl = _fit(X_pl, y, fe_max_steps=1)
    out_pd = np.asarray(pd.DataFrame(m_pd.transform(X_pd)).to_numpy(), dtype=np.float64)
    t_pl = m_pl.transform(X_pl)
    out_pl = np.asarray(t_pl.to_numpy() if hasattr(t_pl, "to_numpy") else t_pl, dtype=np.float64)
    assert out_pd.shape == out_pl.shape
    assert np.allclose(out_pd, out_pl, rtol=1e-12, atol=0, equal_nan=True)
