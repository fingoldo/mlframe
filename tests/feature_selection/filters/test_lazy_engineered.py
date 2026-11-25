"""Engineered columns replayed from their recipe when read (``LazyEngineeredColumn``): the values are those of the eager replay, nothing is computed by moving a column from frame to frame, and the
decision seam keeps the winners of a large frame lazy."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _lazy_engineered as L
from mlframe.feature_selection.filters._fe_frame_ops import fe_append_columns, fe_column_values, fe_extract_columns
from mlframe.feature_selection.filters._mrmr_fit_impl._helpers import fe_decide_on_subsample


def _frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Three float32 columns and an int8 one."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"x0": rng.standard_normal(n).astype(np.float32), "x1": rng.standard_normal(n).astype(np.float32), "x2": rng.standard_normal(n).astype(np.float32), "k": rng.integers(0, 5, n).astype(np.int8)}
    )


def _lazy(frame: pd.DataFrame, fn, calls: "list[int] | None" = None) -> L.LazyEngineeredColumn:
    """A lazy column computing ``fn(frame)``; ``calls`` collects one entry per replay."""

    def compute() -> np.ndarray:
        """The replay."""
        if calls is not None:
            calls.append(1)
        return fn(frame)

    return L.LazyEngineeredColumn(compute, len(frame))


def test_nothing_is_computed_until_a_value_is_read():
    """Building the frame, moving the column, dropping others and asking for the dtype compute nothing; the first read computes once and the second is answered by the cache."""
    base = _frame()
    calls: "list[int]" = []
    col = _lazy(base, lambda d: d["x0"].to_numpy() ** 2, calls)
    X = pd.concat([base, pd.DataFrame({"e": col}, index=base.index, copy=False)], axis=1)
    moved = fe_append_columns(base, fe_extract_columns(X, ["e"]))
    _ = moved.drop(columns=["x1"]).dtypes, moved.select_dtypes(exclude=("category", "object", "string", "bool")).columns
    assert calls == []
    first = fe_column_values(moved, "e")
    second = np.asarray(moved["e"])
    assert len(calls) == 1 and np.array_equal(first, second)
    assert np.allclose(first, base["x0"].to_numpy() ** 2)


def test_every_way_of_reading_a_column_gives_float32_values_not_objects():
    """``Series.to_numpy``, ``np.asarray``, a mixed frame's ``to_numpy`` and a block with a dtype and ``na_value`` are float32 (a mixed block used to come out as ``object``)."""
    base = _frame()
    X = pd.concat([base, pd.DataFrame({"e": _lazy(base, lambda d: d["x1"].to_numpy() * 3.0)}, index=base.index, copy=False)], axis=1)
    expected = base["x1"].to_numpy() * 3.0
    assert X["e"].to_numpy().dtype == np.float32 and np.allclose(X["e"].to_numpy(), expected)
    assert np.asarray(X["e"]).dtype == np.float32
    assert X.to_numpy().dtype == np.float32
    block = X[["x0", "e"]].to_numpy(dtype=np.float32, na_value=np.nan)
    assert block.dtype == np.float32 and np.allclose(block[:, 1], expected)
    assert float(X["e"].mean()) == pytest.approx(float(expected.mean()), rel=1e-5)


def test_a_row_gather_holds_the_values_of_those_rows():
    """``iloc`` / ``take`` give the rows of the eager column, as a held column (nothing is recomputed per row)."""
    base = _frame()
    calls: "list[int]" = []
    X = pd.concat([base, pd.DataFrame({"e": _lazy(base, lambda d: np.sqrt(np.abs(d["x2"].to_numpy())), calls)}, index=base.index, copy=False)], axis=1)
    idx = np.array([3, 10, 11, 400, 599])
    sub = X.iloc[idx]
    assert np.allclose(sub["e"].to_numpy(), np.sqrt(np.abs(base["x2"].to_numpy()[idx])))
    assert isinstance(sub["e"].array, L.LazyEngineeredColumn) and sub["e"].array.nbytes == idx.size * 4
    assert len(calls) == 1


def test_the_cache_is_bounded_and_a_dropped_column_is_recomputed():
    """The cache keeps the newest columns up to its byte bound; a column that left it is replayed again on the next read and gives the same values."""
    base = _frame(n=1000)
    cache = L._ColumnCache(max_bytes=2 * 4000 + 100)
    old = L._CACHE
    L._CACHE = cache
    try:
        calls: "list[int]" = []
        cols = [_lazy(base, lambda d, j=j: d["x0"].to_numpy() + float(j), calls) for j in range(3)]
        for c in cols:
            c.materialize()
        assert len(calls) == 3 and cache._bytes <= cache.max_bytes
        again = cols[0].materialize()
        assert len(calls) == 4 and np.allclose(again, base["x0"].to_numpy())
    finally:
        L._CACHE = old


def test_a_replay_of_the_wrong_length_is_refused():
    """A recipe that returns another row count fails at the read with a message, not with a shifted column."""
    base = _frame()
    col = L.LazyEngineeredColumn(lambda: np.zeros(7, dtype=np.float32), len(base))
    with pytest.raises(ValueError, match="expected"):
        col.materialize()


# ---- the decision seam -------------------------------------------------------------------------------------------------------------------------------------------------------------


def _fit_with_recipes(X_sub: pd.DataFrame, y_sub: np.ndarray, **kwargs):
    """A closed-form family: appends ``sq(x0)`` and ``x1 + x2`` and returns their recipes."""
    out = X_sub.copy()
    out["sq(x0)"] = X_sub["x0"].to_numpy() ** 2
    out["x1+x2"] = X_sub["x1"].to_numpy() + X_sub["x2"].to_numpy()
    recipes = [SimpleNamespace(name="sq(x0)", kind="sq"), SimpleNamespace(name="x1+x2", kind="sum")]
    return out, recipes


def _fake_apply_recipe(recipe, X):
    """The replay of the two recipes above on any frame."""
    if recipe.kind == "sq":
        return np.asarray(X["x0"], dtype=np.float64) ** 2
    return np.asarray(X["x1"], dtype=np.float64) + np.asarray(X["x2"], dtype=np.float64)


@pytest.fixture
def seam(monkeypatch):
    """The replay of the fake recipes in place of the real ``apply_recipe``."""
    from mlframe.feature_selection.filters import engineered_recipes

    monkeypatch.setattr(engineered_recipes, "apply_recipe", _fake_apply_recipe)


def test_the_seam_keeps_the_winners_of_a_large_frame_lazy_and_equal_to_the_eager_replay(monkeypatch, seam):
    """Above the row floor the appended columns are lazy and, read, equal the eager replay to the digit; below it (or with the floor at 0) they are numpy columns."""
    X = _frame(n=4000)
    y = np.zeros(len(X))
    monkeypatch.setenv("MLFRAME_FE_LAZY_REPLAY_MIN_ROWS", "0")
    eager = fe_decide_on_subsample(_fit_with_recipes, X, y, subsample_n=500, recipes_at=1)
    X_eager = eager[0]
    assert type(X_eager["sq(x0)"].array) is not L.LazyEngineeredColumn
    monkeypatch.setenv("MLFRAME_FE_LAZY_REPLAY_MIN_ROWS", "1000")
    before = dict(L.STATS)
    lazy = fe_decide_on_subsample(_fit_with_recipes, X, y, subsample_n=500, recipes_at=1)
    X_lazy = lazy[0]
    assert isinstance(X_lazy["sq(x0)"].array, L.LazyEngineeredColumn) and isinstance(X_lazy["x1+x2"].array, L.LazyEngineeredColumn)
    assert L.STATS["created"] - before["created"] == 2 and L.STATS["replayed"] == before["replayed"]
    for name in ("sq(x0)", "x1+x2"):
        assert np.array_equal(np.asarray(X_lazy[name]), np.asarray(X_eager[name], dtype=np.float32))
    assert L.STATS["replayed"] - before["replayed"] == 2


def test_a_frame_below_the_floor_is_replayed_eagerly(monkeypatch, seam):
    """With fewer rows than the floor the replay is not deferred."""
    monkeypatch.setenv("MLFRAME_FE_LAZY_REPLAY_MIN_ROWS", "100000")
    out = fe_decide_on_subsample(_fit_with_recipes, _frame(n=4000), np.zeros(4000), subsample_n=500, recipes_at=1)[0]
    assert type(out["sq(x0)"].array) is not L.LazyEngineeredColumn


def test_codes_are_replayed_now_and_keep_their_dtype(monkeypatch):
    """A recipe whose replay is integer codes is not made a float lazy column: it is replayed at once, in its own dtype."""
    from mlframe.feature_selection.filters import engineered_recipes

    monkeypatch.setattr(engineered_recipes, "apply_recipe", lambda recipe, X: (np.asarray(X["k"]) % 3).astype(np.int16))

    def family(X_sub, y_sub, **kwargs):
        """A family that emits one code column."""
        out = X_sub.copy()
        out["codes"] = (X_sub["k"].to_numpy() % 3).astype(np.int16)
        return out, [SimpleNamespace(name="codes")]

    monkeypatch.setenv("MLFRAME_FE_LAZY_REPLAY_MIN_ROWS", "1000")
    out = fe_decide_on_subsample(family, _frame(n=4000), np.zeros(4000), subsample_n=500, recipes_at=1)[0]
    assert out["codes"].dtype == np.int16 and type(out["codes"].array) is not L.LazyEngineeredColumn


def test_a_recipe_that_cannot_be_replayed_falls_back_to_the_full_data_decision(monkeypatch):
    """The first rows are replayed at the decision: a failing recipe sends the family to the full-data decision, as the eager replay did."""
    from mlframe.feature_selection.filters import engineered_recipes

    def broken(recipe, X):
        """A replay that fails."""
        raise RuntimeError("cannot replay")

    monkeypatch.setattr(engineered_recipes, "apply_recipe", broken)
    seen_rows: "list[int]" = []

    def family(X_sub, y_sub, **kwargs):
        """Record the row count the family decided on."""
        seen_rows.append(len(X_sub))
        return _fit_with_recipes(X_sub, y_sub)

    monkeypatch.setenv("MLFRAME_FE_LAZY_REPLAY_MIN_ROWS", "1000")
    fe_decide_on_subsample(family, _frame(n=4000), np.zeros(4000), subsample_n=500, recipes_at=1)
    assert seen_rows == [500, 4000]


# ---- the head of a frame ------------------------------------------------------------------------------------------------------------------------------------------------------------


def _chain(monkeypatch, seam):
    """A frame with two lazy winners and a replay for a third recipe that reads one of them."""
    from mlframe.feature_selection.filters import engineered_recipes

    def replay(recipe, X):
        """The two closed forms of the fixture and ``dep = sq(x0) + 1``."""
        if recipe.kind == "dep":
            return np.asarray(X["sq(x0)"], dtype=np.float64) + 1.0
        return _fake_apply_recipe(recipe, X)

    monkeypatch.setattr(engineered_recipes, "apply_recipe", replay)
    monkeypatch.setenv("MLFRAME_FE_LAZY_REPLAY_MIN_ROWS", "1000")
    return fe_decide_on_subsample(_fit_with_recipes, _frame(n=4000), np.zeros(4000), subsample_n=500, recipes_at=1)[0]


def test_the_head_of_a_frame_does_not_replay_its_lazy_columns_at_full_length(monkeypatch, seam):
    """``X.iloc[:2048]`` (the probe of a recipe on the first rows) reads the head of every lazy column by replaying its recipe on those rows alone: no full-length replay happens, and the head equals the head of the full values."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._helpers import _deferred_replay_column

    X = _chain(monkeypatch, seam)
    L._CACHE.clear()
    before = dict(L.STATS)
    head = X.iloc[:300]
    assert L.STATS["replayed"] == before["replayed"] and L.STATS["head_replays"] - before["head_replays"] == 2
    col = _deferred_replay_column(SimpleNamespace(name="dep", kind="dep"), X, np.float32)
    assert col is not None and L.STATS["replayed"] == before["replayed"]
    full = np.asarray(X["sq(x0)"])
    assert L.STATS["replayed"] - before["replayed"] == 1
    assert np.array_equal(np.asarray(head["sq(x0)"]), full[:300]) and np.array_equal(np.asarray(head["x1+x2"]), np.asarray(X["x1+x2"])[:300])
    assert np.array_equal(np.asarray(col), full.astype(np.float64).astype(np.float32) + np.float32(1.0))


def test_a_head_slice_of_a_known_column_is_cut_from_its_values_and_a_long_one_computes_it(monkeypatch, seam):
    """A column that is cached (or held) is cut, with nothing replayed; a slice past the head bound computes the column once."""
    X = _chain(monkeypatch, seam)
    L._CACHE.clear()
    values = np.asarray(X["sq(x0)"])
    before = dict(L.STATS)
    assert np.array_equal(np.asarray(X["sq(x0)"].array[:50]), values[:50]) and L.STATS["replayed"] == before["replayed"] and L.STATS["head_replays"] == before["head_replays"]
    monkeypatch.setattr(L, "HEAD_MAX_ROWS", 10)
    L._CACHE.clear()
    assert np.array_equal(np.asarray(X["x1+x2"].array[:50]), np.asarray(X["x1+x2"])[:50])
    assert L.STATS["head_replays"] == before["head_replays"]


def test_the_subsample_gather_of_a_lazy_column_replays_it_once_per_draw(monkeypatch, seam):
    """Every family that decides on the same subsample gathers the same rows of the same lazy columns: the column is replayed once, the later gathers are answered by the gather cache, and a different draw gives its own rows."""
    X = _chain(monkeypatch, seam)
    L._CACHE.clear()
    L._GATHER_CACHE.clear()
    expected = np.asarray(X["sq(x0)"]).copy()
    L._CACHE.clear()
    idx = np.random.default_rng(3).choice(len(X), 500, replace=False)
    before = dict(L.STATS)
    first = X.iloc[idx]
    L._CACHE.clear()
    second = X.iloc[idx.copy()]
    assert L.STATS["replayed"] - before["replayed"] == 2 and L.STATS["gather_hits"] - before["gather_hits"] == 2
    assert np.array_equal(np.asarray(first["sq(x0)"]), expected[idx]) and np.array_equal(np.asarray(second["sq(x0)"]), expected[idx])
    other = np.sort(idx)
    assert np.array_equal(np.asarray(X.iloc[other]["sq(x0)"]), expected[other])


def test_a_cache_gives_memory_back_when_the_machine_is_tight(monkeypatch):
    """With little memory available a cache keeps the share of it (never less than the floor); with plenty it keeps its own bound."""
    import types

    cache = L._ColumnCache(max_bytes=10_000)
    for j in range(5):
        cache.put(j, np.zeros(500, dtype=np.float32))
    assert cache._bytes == 10_000
    monkeypatch.setattr(L, "CACHE_FLOOR_BYTES", 2_500)
    monkeypatch.setitem(__import__("sys").modules, "psutil", types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(available=10_000)))
    cache.put(99, np.zeros(500, dtype=np.float32))
    assert cache._bytes <= 2_500 and cache.get(99) is not None and cache.get(0) is None
