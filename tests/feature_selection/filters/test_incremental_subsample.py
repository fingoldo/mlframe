"""The decision subsample of a frame that is its parent plus appended columns reuses the parent's cached subsample and gathers only the appended columns; the frame is the one a full gather gives."""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlframe.feature_selection.filters import _fe_frame_ops as ops
from mlframe.feature_selection.filters import _lazy_engineered as L
from mlframe.feature_selection.filters._mrmr_fit_impl import _helpers as H


@pytest.fixture
def gathers(monkeypatch):
    """A fresh subsample cache and a log of the column counts the gather is asked for."""
    monkeypatch.setitem(H._SUBSAMPLE_CACHE, "X", None)
    monkeypatch.setitem(H._SUBSAMPLE_CACHE, "X_sub", None)
    monkeypatch.setitem(H._SUBSAMPLE_CACHE, "idx", None)
    log: "list[int]" = []
    real = ops.fe_subsample_to_pandas

    def counting(X, idx):
        """The real gather, remembering how many columns it was asked for."""
        log.append(int(X.shape[1]))
        return real(X, idx)

    monkeypatch.setattr(ops, "fe_subsample_to_pandas", counting)
    return log


def _frame(n=2000, k=30, seed=0):
    """A float32 frame of ``k`` columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"c{i}": rng.standard_normal(n).astype(np.float32) for i in range(k)})


def test_a_child_gathers_only_its_appended_columns_and_equals_the_full_gather(gathers):
    """Parent gathered in full (30 columns), the child (+2) gathers 2; three generations in a row, each equal to the frame a full gather of it gives."""
    X0 = _frame()
    y = np.zeros(len(X0))
    idx = np.sort(np.random.default_rng(1).choice(len(X0), 300, replace=False))
    first, _ = H._gathered_subsample(X0, y, idx)
    assert gathers == [30]
    X1 = ops.fe_append_columns(X0, {"e1": X0["c0"].to_numpy() * 2.0, "e2": X0["c1"].to_numpy() + 1.0})
    X2 = ops.fe_append_columns(X1, {"e3": X0["c2"].to_numpy() ** 2})
    X3 = ops.fe_append_columns(X2, {"e4": X0["c3"].to_numpy() - 3.0, "e5": X0["c4"].to_numpy()})
    for child, expect_cols in ((X1, 32), (X2, 33), (X3, 35)):
        got, _ = H._gathered_subsample(child, y, idx)
        assert_frame_equal(got, child.iloc[idx].reset_index(drop=True))
        assert got.shape[1] == expect_cols
    assert gathers == [30, 2, 1, 2]
    assert_frame_equal(first, X0.iloc[idx].reset_index(drop=True))


def test_lazy_columns_are_appended_without_being_computed_by_the_gather_of_the_others(gathers):
    """The appended columns may be lazy engineered ones: they come through the gather of the child only."""
    X0 = _frame(n=3000, k=12)
    y = np.zeros(len(X0))
    idx = np.arange(0, 3000, 11)
    H._gathered_subsample(X0, y, idx)
    calls = []

    def compute():
        """The replay of the lazy column."""
        calls.append(1)
        return (X0["c0"].to_numpy() * 3.0).astype(np.float32)

    lazy = L.LazyEngineeredColumn(compute, len(X0))
    X1 = pd.concat([X0, pd.DataFrame({"lz": lazy}, index=X0.index)], axis=1)
    ops._note_appended(X1, X0, ["lz"])
    got, _ = H._gathered_subsample(X1, y, idx)
    assert gathers == [12, 1] and len(calls) == 1
    assert np.allclose(np.asarray(got["lz"]), X0["c0"].to_numpy()[idx] * 3.0)


def test_an_equal_copy_of_the_draw_is_the_same_draw_another_draw_gathers_in_full(gathers):
    """Every family builds its own array of the fit's one draw: equal values are the same rows (an exact repeat of the frame is answered from the cache, a child gathers its appended columns); other rows gather in full."""
    X0 = _frame()
    y = np.zeros(len(X0))
    idx = np.arange(0, 2000, 7)
    H._gathered_subsample(X0, y, idx)
    H._gathered_subsample(X0, y, idx.copy())
    assert gathers == [30]
    X1 = ops.fe_append_columns(X0, {"e1": X0["c0"].to_numpy() * 2.0})
    H._gathered_subsample(X1, y, idx.copy())
    assert gathers == [30, 1]
    X1b = ops.fe_append_columns(X1, {"e2": X0["c1"].to_numpy()})
    H._gathered_subsample(X1b, y, np.arange(1, 2000, 7))  # other rows: a full gather
    assert gathers == [30, 1, 32]
    idx = np.arange(0, 2000, 7)
    Xo = _frame(seed=5)
    H._gathered_subsample(Xo, y, idx)
    X2 = ops.fe_append_columns(X0, {"e9": X0["c1"].to_numpy()})  # appended to a frame that is not the cached one
    H._gathered_subsample(X2, y, idx)
    assert gathers[-1] == 31
    Xp = _frame(seed=6)
    H._gathered_subsample(Xp, y, idx)
    Xc = ops.fe_append_columns(Xp, {"e": Xp["c0"].to_numpy()})
    del Xp
    gc.collect()
    H._gathered_subsample(Xc, y, idx)
    assert gathers[-1] == 31


def test_a_frame_whose_columns_are_not_the_parents_plus_the_appended_ones_gathers_in_full(gathers, monkeypatch):
    """A recorded lineage that does not match the columns (a column renamed or dropped after the append) is not trusted."""
    X0 = _frame()
    y = np.zeros(len(X0))
    idx = np.arange(0, 2000, 7)
    H._gathered_subsample(X0, y, idx)
    X1 = ops.fe_append_columns(X0, {"e1": X0["c0"].to_numpy() * 2.0})
    X1b = X1.rename(columns={"e1": "other"})
    ops._note_appended(X1b, X0, ["e1"])
    got, _ = H._gathered_subsample(X1b, y, idx)
    assert gathers[-1] == 31 and list(got.columns)[-1] == "other"
