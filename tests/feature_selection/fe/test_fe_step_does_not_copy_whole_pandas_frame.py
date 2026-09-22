"""The FE step gets its own frame without duplicating the caller's data.

``fit`` must not mutate the caller's DataFrame, and the pandas path appends engineered columns in place, so the step takes a private frame
first. It took a DEEP copy, duplicating every existing column's data, on frames this package is explicitly built to handle at 100+ GB. Every
write in the step assigns a NEW engineered column name, so the private frame only needs its own column index: sharing the existing buffers
leaves the caller's frame untouched just the same.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def frame():
    """A frame whose column buffers are identifiable, plus a target with real signal."""
    rng = np.random.default_rng(0)
    n = 1500
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": b, "c": rng.normal(size=n), "d": rng.normal(size=n)})
    y = ((a * b) > 0).astype(np.int64)
    return X, y


def test_the_fit_does_not_deep_copy_the_callers_frame(frame, monkeypatch):
    """No full-data copy of the caller's frame is taken during a fit that engineers columns."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = frame
    deep_copies: list = []
    real_copy = pd.DataFrame.copy

    def recording_copy(self, deep=True):
        """Record a deep copy of the caller's own frame, which is the one the FE step used to take."""
        if deep and self.shape == X.shape:
            deep_copies.append(self.shape)
        return real_copy(self, deep=deep)

    monkeypatch.setattr(pd.DataFrame, "copy", recording_copy)
    MRMR._FIT_CACHE.clear()
    MRMR(random_state=0, verbose=0, fe_max_steps=1, full_npermutations=3, baseline_npermutations=2).fit(X, y)
    assert not deep_copies, f"the FE step deep-copied the caller's frame: {deep_copies}"
    # Deep copies of the AUGMENTED frame (wider than the caller's) happen elsewhere in the fit and are a separate question from this one.


def test_the_callers_frame_is_still_not_mutated(frame):
    """The contract the copy exists for: after a fit, the caller's frame is exactly what it was."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = frame
    before = X.copy(deep=True)
    MRMR._FIT_CACHE.clear()
    MRMR(random_state=0, verbose=0, fe_max_steps=1, full_npermutations=3, baseline_npermutations=2).fit(X, y)
    assert list(X.columns) == list(before.columns), f"engineered columns leaked into the caller's frame: {set(X.columns) - set(before.columns)}"
    pd.testing.assert_frame_equal(X, before)


def test_a_shallow_copy_does_not_carry_new_columns_back_to_the_original():
    """The property the fix rests on, pinned directly: adding a column to a shallow copy leaves the original alone."""
    original = pd.DataFrame({"x": np.arange(5.0), "y": np.arange(5.0)})
    private = original.copy(deep=False)
    private["engineered"] = np.arange(5.0) * 2
    assert "engineered" not in original.columns
    assert list(original.columns) == ["x", "y"]
