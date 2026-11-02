"""The RFECV leakage screen keeps scanning after a nullable column and still flags a leaky one behind it (tracker 2026-11-02 wave 9, EXC9-1)."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.wrappers.rfecv._validate import _screen_target_leakage


class _Stub:
    """Carries only the threshold the screen reads."""

    leakage_corr_threshold = 0.95


def test_leaky_column_after_a_nullable_int_column_is_flagged():
    """An Int64 column holding pd.NA used to abort the whole scan, hiding the leak that follows it."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(size=n)
    nullable = pd.array(rng.integers(0, 5, n), dtype="Int64")
    nullable[::7] = pd.NA
    X = pd.DataFrame({"a_nullable": nullable, "z_leak": y + rng.normal(scale=0.01, size=n)})
    found = []
    try:
        _screen_target_leakage(_Stub(), X, y, found)
    except Exception:  # the screen may raise on its configured action; the finding list is what counts
        pass
    assert [c for c, _ in found] == ["z_leak"]
