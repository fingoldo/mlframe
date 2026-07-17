"""Pin: MRMR feature engineering RUNS on a polars input (2026-06-16).

polars is a primary input format. The FE families require pandas (several raise TypeError on a
polars frame), so historically a polars input silently received NO feature engineering. MRMR.fit now
bridges a polars DataFrame to an Arrow-backed pandas view (``get_pandas_view_of_polars_df``) when FE
is enabled, so engineered features are produced. This test pins that contract: a polars input must
yield the SAME engineered structure as the equivalent pandas input, not raw-only.
"""

from __future__ import annotations

import numpy as np
import pytest

pl = pytest.importorskip("polars")
import pandas as pd  # noqa: E402

from mlframe.feature_selection.filters.mrmr import MRMR  # noqa: E402


def _canonical(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(1.0, 5.0, n)
    b = rng.uniform(1.0, 5.0, n)
    c = rng.uniform(1.0, 5.0, n)
    d = rng.uniform(0.0, 2 * np.pi, n)
    f = rng.normal(0.0, 1.0, n)
    y = a**2 / b + f / 5.0 + 3.0 * np.log(c) * np.sin(d)
    data = {"a": a, "b": b, "c": c, "d": d, "e": rng.normal(0.0, 1.0, n)}
    return data, y


def _eng_names(m, raw_cols):
    return [nm for nm in m.get_feature_names_out() if nm not in set(raw_cols)]


def test_fe_runs_on_polars_input_and_matches_pandas():
    """A polars DataFrame must get feature engineering (not silently skipped), and the engineered
    structure must match the equivalent pandas fit."""
    data, y = _canonical()
    raw_cols = list(data.keys())
    base = dict(verbose=0, random_seed=0, n_jobs=1, fe_max_steps=2, dcd_enable=False, build_friend_graph=False, cluster_aggregate_enable=False)

    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)

    np.random.seed(0)
    m_pd = MRMR(**base).fit(df_pd.copy(), pd.Series(y, name="y"))
    np.random.seed(0)
    m_pl = MRMR(**base).fit(df_pl.clone(), pl.Series("y", y))

    eng_pd = _eng_names(m_pd, raw_cols)
    eng_pl = _eng_names(m_pl, raw_cols)

    # The pandas fit must produce engineered features (fixture sanity).
    assert eng_pd, f"pandas baseline produced no engineered features: {m_pd.get_feature_names_out()}"
    # The polars input must NOT be silently raw-only -- FE must have run.
    assert eng_pl, f"polars input produced NO engineered features (FE silently skipped); selection={list(m_pl.get_feature_names_out())}"
    # And the engineered structure should match the pandas fit (same bridge -> same FE).
    assert set(eng_pl) == set(eng_pd), f"polars FE diverged from pandas FE:\n  pandas={sorted(eng_pd)}\n  polars={sorted(eng_pl)}"


def test_fe_disabled_polars_stays_native_no_error():
    """With FE off (fe_max_steps=0) a polars input must still fit (native path), raw-only."""
    data, y = _canonical(n=2000)
    df_pl = pl.DataFrame(data)
    m = MRMR(verbose=0, random_seed=0, n_jobs=1, fe_max_steps=0, dcd_enable=False, build_friend_graph=False, cluster_aggregate_enable=False).fit(
        df_pl.clone(), pl.Series("y", y)
    )
    # no crash, and selection is a subset of the raw columns (no FE).
    assert set(m.get_feature_names_out()) <= set(data.keys())
