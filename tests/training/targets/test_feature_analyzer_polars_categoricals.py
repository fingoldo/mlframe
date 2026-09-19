"""A polars frame's categorical/string columns are counted in polars, not converted to pandas first.

Converting a 498k x 109 production frame (one text column with 232k distinct values) cost 3.4s for a distinct count
polars answers in milliseconds. The report must be the same as the pandas route's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from mlframe.training.targets import analyze_feature_distribution


def _frame(n=3000):
    rng = np.random.default_rng(0)
    base = rng.normal(size=n)
    return pl.DataFrame(
        {
            "x": base,
            "x_twin": base + rng.normal(scale=0.01, size=n),
            "flag": rng.random(n) < 0.5,
            "const": np.ones(n),
            "text": [f"t{i}" for i in rng.integers(0, 2000, size=n)],
            "cat": pl.Series(rng.choice(["a", "b", None], size=n).tolist(), dtype=pl.Categorical),
        }
    )


def test_polars_report_matches_pandas_report():
    df = _frame()
    y = df["x"].to_numpy() * 2.0
    rep_pl = analyze_feature_distribution(df, y=y, high_cardinality_max=100)
    rep_pd = analyze_feature_distribution(df.to_pandas(), y=y, high_cardinality_max=100)
    assert rep_pl.n_features == rep_pd.n_features == 6
    assert sorted(rep_pl.pathologies) == sorted(rep_pd.pathologies)
    assert sorted(rep_pl.drop_candidates) == sorted(rep_pd.drop_candidates)
    assert rep_pl.diagnostics["high_cardinality_features"] == ["text"]
    assert rep_pl.diagnostics["n_categorical"] == rep_pd.diagnostics["n_categorical"] == 2


def test_categoricals_are_not_converted(monkeypatch):
    import mlframe.training.utils as utils

    seen = []
    orig = utils.get_pandas_view_of_polars_df

    def _spy(frame, *a, **k):
        seen.append(list(frame.columns))
        return orig(frame, *a, **k)

    monkeypatch.setattr(utils, "get_pandas_view_of_polars_df", _spy)
    analyze_feature_distribution(_frame())
    assert seen and all("text" not in cols and "cat" not in cols for cols in seen)
