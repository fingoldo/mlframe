"""``group_key_strings`` on string columns skips the per-row token call and returns exactly what the per-row map returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._internals import canonical_group_token, group_key_strings


def _per_row(col: pd.Series) -> np.ndarray:
    """The reference: the canonical token of every row, one Python call each."""
    return np.asarray(col.astype(object).map(canonical_group_token).to_numpy())


@pytest.mark.parametrize("dtype", ["object", "string", "string[pyarrow]", "category"])
def test_string_column_matches_the_per_row_tokens(dtype):
    """Same keys, same order, for plain, pandas-string, arrow-string and categorical string columns."""
    rng = np.random.default_rng(0)
    col = pd.Series(rng.choice(["a", "bb", "1", "1.0", "", " x "], size=5000)).astype(dtype)
    got = group_key_strings(col)
    assert got.dtype == object
    assert got.tolist() == _per_row(col).tolist()


@pytest.mark.parametrize("values", [["a", None, "b"], ["a", np.nan, "b"], ["a", 1, "b"], ["a", 1.0, True], [1, 1.0, True]])
def test_missing_and_mixed_object_columns_keep_the_per_row_tokens(values):
    """A missing value or a mixed type is not taken by the fast path: ``None`` stays ``'None'``, ``1`` and ``1.0`` collapse, ``True`` stays ``'True'``."""
    col = pd.Series(values, dtype=object)
    assert group_key_strings(col).tolist() == _per_row(col).tolist()


def test_the_fast_path_is_taken_for_a_string_column(monkeypatch):
    """A pure-string column never reaches the per-row token function (floor: it is called zero times)."""
    import mlframe.feature_selection.filters._internals as internals

    calls = []
    monkeypatch.setattr(internals, "canonical_group_token", lambda v: calls.append(v) or str(v))
    out = internals.group_key_strings(pd.Series(["u", "v", "u"], dtype="string[pyarrow]"))
    assert out.tolist() == ["u", "v", "u"]
    assert calls == []
