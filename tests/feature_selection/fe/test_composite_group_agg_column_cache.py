"""The per-call column cache of the composite group aggregates changes no column and no recipe, and copies each numeric column once for all key sets."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _composite_group_agg_fe as cga


def _frame(n: int = 3000) -> pd.DataFrame:
    """Three group columns and four numeric columns of a row-major block (strided columns), one with missing values."""
    rng = np.random.default_rng(0)
    num = pd.DataFrame(rng.standard_normal((n, 4)).astype(np.float32), columns=[f"x{j}" for j in range(4)])
    num.loc[::37, "x2"] = np.nan
    for j, card in enumerate((4, 3, 5)):
        num[f"g{j}"] = rng.integers(0, card, n)
    return num


_KEY_SETS = [("g0", "g1"), ("g0", "g2"), ("g1", "g2")]
_NUM = ["x0", "x1", "x2", "x3"]


def _generate(X: pd.DataFrame) -> "tuple[pd.DataFrame, dict]":
    """The engineered frame and the raw recipes of the three key sets."""
    return cga.generate_composite_group_agg_features(X, _KEY_SETS, _NUM)


def test_cached_columns_give_the_same_features_and_recipes(monkeypatch):
    """With the cache switched off (budget 0) every column and every recipe scalar and lookup is the same."""
    X = _frame()
    got_df, got_rec = _generate(X)
    monkeypatch.setattr(cga, "_COLUMN_CACHE_MAX_BYTES", 0)
    want_df, want_rec = _generate(X)
    assert got_df.shape[1] == len(_KEY_SETS) * len(_NUM) * (len(cga.COMPOSITE_STAT_NAMES) + 2)
    pd.testing.assert_frame_equal(got_df, want_df)
    assert set(got_rec) == set(want_rec)
    for name, want in want_rec.items():
        got = got_rec[name]
        assert (got["global_value"], got["global_mean"], got["global_std"]) == (want["global_value"], want["global_mean"], want["global_std"]), name
        assert dict(got["group_lookup_dict"]) == dict(want["group_lookup_dict"]), name


def test_each_numeric_column_is_copied_and_hashed_once(monkeypatch):
    """Four numeric columns over three key sets: one stacked copy, four content hashes (twelve of each before)."""
    X = _frame()
    stacks, hashes = [], []
    real_stack = cga.fe_stack_rows
    monkeypatch.setattr(cga, "fe_stack_rows", lambda cols, dtype: stacks.append(len(cols)) or real_stack(cols, dtype))
    real_digest = cga._ColumnCache.digest

    def digest(self, col, x):
        """Count the columns whose hash is computed (a miss of the cache's own table)."""
        if col not in self._digests:
            hashes.append(col)
        return real_digest(self, col, x)

    monkeypatch.setattr(cga._ColumnCache, "digest", digest)
    monkeypatch.setenv("MLFRAME_EDGE_JOBS", "1")
    _generate(X)
    assert stacks == [len(_NUM)]
    assert sorted(hashes) == sorted(_NUM)


def test_cache_holds_nothing_over_its_budget(monkeypatch):
    """A column set larger than the budget is left to the tasks."""
    monkeypatch.setattr(cga, "_COLUMN_CACHE_MAX_BYTES", 16)
    cache = cga._ColumnCache(_frame(200), _NUM)
    assert cache.values("x0") is None
