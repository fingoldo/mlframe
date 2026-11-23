"""The usability union keeps the container type of the input: an ndarray in gives an ndarray out, a DataFrame in gives a DataFrame out (AUD23-11)."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._usability_lists as UL
from mlframe.feature_selection.filters.mrmr import MRMR


def _stub(extra_names):
    """An object with just what ``_append_usability_union`` reads of the estimator."""
    cand = [types.SimpleNamespace(name=n) for n in extra_names]
    stub = types.SimpleNamespace(
        support_linear_=cand,
        support_universal_=[],
        get_feature_names_out=lambda: np.asarray(["a", "b", *extra_names], dtype=object),
    )
    stub._usability_union_extra = types.MethodType(MRMR._usability_union_extra, stub)
    return stub


def _patch_materialise(monkeypatch):
    """The materialiser returns one constant column per candidate, indexed like X."""
    monkeypatch.setattr(UL, "materialize_usability_features", lambda extra, X: pd.DataFrame({c.name: np.full(len(X), 7.0) for c in extra}))


def test_ndarray_in_ndarray_out_with_extra_columns(monkeypatch):
    """Two base columns plus one union column: an (n, 3) ndarray, the union values in the last column."""
    _patch_materialise(monkeypatch)
    X = np.arange(10, dtype=float).reshape(5, 2)
    stub = _stub(["u"])
    out = MRMR._append_usability_union(stub, X.copy(), X)
    assert isinstance(out, np.ndarray) and out.shape == (5, 3)
    np.testing.assert_array_equal(out[:, :2], X)
    assert np.all(out[:, 2] == 7.0)
    assert stub.usability_feature_groups_["linear"] == ["u"]


def test_ndarray_in_ndarray_out_without_extra_columns(monkeypatch):
    """The union adds nothing (the linear list is already in the base output): still an ndarray, as the plain transform returns."""
    _patch_materialise(monkeypatch)
    X = np.arange(10, dtype=float).reshape(5, 2)
    stub = _stub([])
    out = MRMR._append_usability_union(stub, X.copy(), X)
    assert isinstance(out, np.ndarray) and out.shape == (5, 2)


def test_dataframe_in_dataframe_out(monkeypatch):
    """A DataFrame stays a DataFrame, with the union column appended."""
    _patch_materialise(monkeypatch)
    X = pd.DataFrame(np.arange(10, dtype=float).reshape(5, 2), columns=["a", "b"])
    out = MRMR._append_usability_union(_stub(["u"]), X.copy(), X)
    assert isinstance(out, pd.DataFrame) and list(out.columns) == ["a", "b", "u"]
