"""The cat_pair and cat_triple stages cross the categorical-looking RAW columns: continuous columns are never handed over as categorical members."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl import _fe_stage_cascade_mid_a as S


def _frame(n=600, seed=0):
    """Continuous columns, three small-integer columns and one engineered-looking integer column that is not a raw input."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(6)})
    for k in range(3):
        df[f"i{k}"] = rng.integers(0, 5, n).astype(np.int8)
    df["eng_count"] = rng.integers(0, 4, n).astype(np.int8)
    return df, rng.integers(0, 2, n)


def test_cat_pair_members_are_the_categorical_raw_columns(monkeypatch):
    """The pair family receives only the integer raw columns, never a continuous one or an engineered intermediate."""
    import mlframe.feature_selection.filters._cat_pair_fe as CP

    seen = {}

    def fake(X, y, *, cat_cols=None, **kw):
        """Record the members handed to the family and return an empty result."""
        seen["cols"] = list(cat_cols)
        return X, [], [], pd.DataFrame()

    monkeypatch.setattr(CP, "hybrid_cat_pair_fe", fake)
    df, y = _frame()
    raw = [c for c in df.columns if c != "eng_count"]
    S._stage_cat_pair(SimpleNamespace(), lambda *a, **k: True, df, y, raw, {}, 0)
    assert seen["cols"] == ["i0", "i1", "i2"]


def test_a_frame_without_two_categorical_raw_columns_skips_the_family(monkeypatch):
    """Nothing to cross: the family is not called at all."""
    import mlframe.feature_selection.filters._cat_pair_fe as CP

    calls = []
    monkeypatch.setattr(CP, "hybrid_cat_pair_fe", lambda *a, **k: calls.append(1))
    df, y = _frame()
    S._stage_cat_pair(SimpleNamespace(), lambda *a, **k: True, df, y, [f"x{i}" for i in range(6)], {}, 0)
    assert not calls
