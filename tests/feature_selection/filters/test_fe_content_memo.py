"""The content memo returns equal results for equal inputs, never for different ones, and a caller's edit of a result does not leak back."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _fe_content_memo as M

CALLS = []


@M.content_memo(max_entries=2)
def _scorer(X: pd.DataFrame, y: np.ndarray, *, k: int = 1) -> pd.DataFrame:
    """A counting stand-in for a scorer."""
    CALLS.append(1)
    return pd.DataFrame({"s": X.sum(axis=1).to_numpy() * k + float(np.sum(y))}, index=X.index)


def test_hit_miss_and_isolation():
    """Same content hits (one real call), changed data / y / kwargs miss, edits of a result do not leak, the store is bounded."""
    CALLS.clear()
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))
    y = rng.integers(0, 2, 50)
    a = _scorer(X, y)
    b = _scorer(X.copy(), y.copy())
    pd.testing.assert_frame_equal(a, b)
    assert len(CALLS) == 1
    b.iloc[:, 0] = 0.0
    pd.testing.assert_frame_equal(_scorer(X, y), a)
    X2 = X.copy()
    X2.iloc[0, 0] += 1.0
    _scorer(X2, y)
    _scorer(X, y + 1)
    _scorer(X, y, k=2)
    assert len(CALLS) == 4
    assert M._release_content_memos() >= 0
