"""raw_mi_noise_floor reads a column subset by name, and keys its memo without copying the matrix.

The missingness family built its floor reference as ``frame[[raw columns]]`` (a sub-frame copy), and the floor then copied the numeric columns
again through ``raw_X[num_cols].to_numpy()`` and a third time through ``arr.tobytes()`` for the memo key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _unified_fe_gate as ug


def _frame(seed: int = 0):
    """Numeric columns of varying relevance, one non-numeric column, and a 3-class target."""
    rng = np.random.default_rng(seed)
    n = 6000
    y = rng.integers(0, 3, size=n)
    cols = {f"x{i}": rng.normal(size=n) + (0.4 * i / 10) * y for i in range(10)}
    cols["label"] = rng.choice(["a", "b"], size=n)
    return pd.DataFrame(cols), y


def _clear_memo():
    """Empty the floor memo so each test measures a fresh computation."""
    with ug._FE_GATE_MEMO_LOCK:
        ug._RAW_MI_FLOOR_MEMO.clear()


def test_columns_argument_equals_subframe_floor():
    """The floor over ``columns`` of the full frame equals the floor over the explicit sub-frame, bit for bit."""
    X, y = _frame()
    subset = ["x1", "x3", "x5", "x7", "label"]
    _clear_memo()
    a = ug.raw_mi_noise_floor(X[subset], y)
    _clear_memo()
    b = ug.raw_mi_noise_floor(X, y, columns=subset)
    assert a == b and a > 0.0


def test_columns_path_builds_no_subframe(monkeypatch):
    """With ``columns`` given, the floor never indexes the frame with a list of column names."""
    X, y = _frame()
    list_keys = []
    real_getitem = pd.DataFrame.__getitem__

    def _spy(self, key):
        """Record list-key column selections, then delegate."""
        if isinstance(key, list):
            list_keys.append(tuple(key))
        return real_getitem(self, key)

    monkeypatch.setattr(pd.DataFrame, "__getitem__", _spy)
    _clear_memo()
    ug.raw_mi_noise_floor(X, y, columns=["x1", "x2", "x3"])
    assert not list_keys, f"sub-frame copies made: {list_keys}"


def test_memo_hits_on_equal_content_and_misses_on_change(monkeypatch):
    """Equal content (a separate copy) is served from the memo; a changed value recomputes."""
    X, y = _frame()
    calls = {"n": 0}
    import mlframe.feature_selection.filters._orthogonal_univariate_fe as ouf

    real = ouf._mi_classif_batch

    def _count(*args, **kwargs):
        """Count scorer calls and delegate."""
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ouf, "_mi_classif_batch", _count)
    monkeypatch.setattr("mlframe.feature_selection.filters._resident_raw_mi.resident_raw_baseline_mi", lambda *a, **k: None)
    _clear_memo()
    first = ug.raw_mi_noise_floor(X, y, columns=["x1", "x2", "x3", "x4"])
    again = ug.raw_mi_noise_floor(X.copy(deep=True), y.copy(), columns=["x1", "x2", "x3", "x4"])
    assert calls["n"] == 1 and first == again
    X2 = X.copy(deep=True)
    X2.loc[0, "x2"] = X2.loc[0, "x2"] + 1.0
    ug.raw_mi_noise_floor(X2, y, columns=["x1", "x2", "x3", "x4"])
    assert calls["n"] == 2


def test_buffer_hash_distinguishes_dtype_and_shape():
    """Same bytes under a different dtype or shape hash differently; equal arrays hash equally."""
    a = np.arange(12, dtype=np.int64)
    assert ug._buffer_hash(a) == ug._buffer_hash(a.copy())
    assert ug._buffer_hash(a) != ug._buffer_hash(a.reshape(3, 4))
    assert ug._buffer_hash(a) != ug._buffer_hash(a.view(np.float64))
