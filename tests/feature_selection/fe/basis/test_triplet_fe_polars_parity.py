"""The hybrid_orth triplet FE stage must run on polars input, not just pandas.

Before the matrix-native FE seam, ~23 FE families guarded on ``isinstance(X, pd.DataFrame)`` and SILENTLY SKIPPED on a
polars frame -- so a polars-native suite ran with most of the FE arsenal disabled. The seam
(``fe_decide_on_subsample`` + ``_fe_frame_ops``) makes the family run format-agnostically: it subsamples any frame to a
pandas decision block, replays winners on the source frame via the (already format-agnostic) recipe path, and appends the
engineered columns in the source framework. This pins that a polars fit engineers the SAME triplet columns as pandas and
selects equivalently -- the fix must be a no-op on the selection, only removing the format restriction.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _fit(df, y):
    sel = MRMR(
        fe_hybrid_orth_triplet_enable=True,
        fe_hybrid_orth_triplet_max_degree=1,
        verbose=0,
        random_seed=0,
    )
    sel.fit(df, y)
    return sel


def _triplet_cols(sel):
    return sorted(c for c in (getattr(sel, "hybrid_orth_features_", None) or []) if c.split("__", 1)[0].count("*") == 2)


def test_triplet_fe_runs_on_polars_and_matches_pandas():
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(0)
    n = 1500
    x1, x2, x3 = (rng.standard_normal(n) for _ in range(3))
    noise = rng.standard_normal((n, 4))  # decoys so selection is non-trivial
    y = (np.sign(x1 * x2 * x3) > 0).astype(int)

    data = {"x1": x1, "x2": x2, "x3": x3, "n0": noise[:, 0], "n1": noise[:, 1], "n2": noise[:, 2], "n3": noise[:, 3]}
    X_pd = pd.DataFrame(data)
    X_pl = pl.DataFrame(data)

    sel_pd = _fit(X_pd, pd.Series(y))
    sel_pl = _fit(X_pl, pd.Series(y))

    tri_pd = _triplet_cols(sel_pd)
    tri_pl = _triplet_cols(sel_pl)

    # The whole point: polars must NOT silently skip the triplet stage -- it engineers triplet columns like pandas.
    assert tri_pd, "pandas baseline engineered no triplet columns; test fixture is wrong"
    assert tri_pl, "polars input engineered NO triplet columns -- the FE family was skipped (the bug this seam fixes)"
    # Same engineered triplet set and same final selection (the fix is selection-equivalent, format only).
    assert tri_pl == tri_pd, f"polars triplet columns diverged from pandas:\n  pd={tri_pd}\n  pl={tri_pl}"
    assert sorted(map(str, sel_pl.support_)) == sorted(map(str, sel_pd.support_)), "polars vs pandas selection diverged"


def _quad_cols(sel):
    return sorted(c for c in (getattr(sel, "hybrid_orth_features_", None) or []) if c.split("__", 1)[0].count("*") == 3)


def test_quadruplet_fe_runs_on_polars_and_matches_pandas():
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(1)
    n = 1500
    x1, x2, x3, x4 = (rng.standard_normal(n) for _ in range(4))
    noise = rng.standard_normal((n, 3))
    y = (np.sign(x1 * x2 * x3 * x4) > 0).astype(int)
    data = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "n0": noise[:, 0], "n1": noise[:, 1], "n2": noise[:, 2]}
    X_pd, X_pl = pd.DataFrame(data), pl.DataFrame(data)

    def _fit_q(df):
        s = MRMR(fe_hybrid_orth_quadruplet_enable=True, fe_hybrid_orth_quadruplet_max_degree=1, verbose=0, random_seed=1)
        s.fit(df, pd.Series(y))
        return s

    sel_pd, sel_pl = _fit_q(X_pd), _fit_q(X_pl)
    q_pd, q_pl = _quad_cols(sel_pd), _quad_cols(sel_pl)
    assert q_pd, "pandas baseline engineered no quadruplet columns; fixture wrong"
    assert q_pl, "polars input engineered NO quadruplet columns -- the FE family was skipped"
    assert q_pl == q_pd, f"polars quad columns diverged from pandas:\n  pd={q_pd}\n  pl={q_pl}"
    assert sorted(map(str, sel_pl.support_)) == sorted(map(str, sel_pd.support_))
