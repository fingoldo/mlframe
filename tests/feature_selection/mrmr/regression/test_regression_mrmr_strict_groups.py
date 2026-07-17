"""Regression sensor for S30 / A1#4: MRMR ``strict_groups`` toggle.

Pre-fix: MRMR.fit accepted ``groups=`` but silently ignored them (warn-only),
which masked cross-group leakage in MI estimation on panel / session data.

Fix: ``strict_groups: bool`` constructor knob. When True, passing ``groups``
raises NotImplementedError with a clear message; when False, the original
UserWarning fires. Code-quality audit finding #20 (2026-07-17) flipped the
default True -> matching ``sample_weight``, which is ALWAYS consumed rather
than silently dropped; ``strict_groups=False`` remains available as an
explicit opt-out for the legacy warn-only group-naive fallback.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _tiny_xy(seed: int = 0):
    """Build a tiny classification fixture with 6 groups of 10 rows each."""
    rng = np.random.default_rng(seed)
    n = 60
    X = pd.DataFrame({"x0": rng.normal(size=n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    y = pd.Series((X["x0"] + 0.5 * X["x1"] > 0).astype(int), name="y")
    groups = pd.Series(np.repeat(np.arange(6), 10), name="g")
    return X, y, groups


def test_mrmr_strict_groups_true_raises_on_groups():
    """strict_groups=True must raise NotImplementedError when groups is passed without group_aware_mi."""
    X, y, groups = _tiny_xy()
    sel = MRMR(verbose=0, random_seed=42, strict_groups=True)
    with pytest.raises(NotImplementedError, match="groups"):
        sel.fit(X, y, groups=groups)


def test_mrmr_strict_groups_false_warns_on_groups():
    """strict_groups=False must warn-only and stamp groups_ignored_=True instead of raising."""
    X, y, groups = _tiny_xy()
    sel = MRMR(verbose=0, random_seed=42, strict_groups=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            sel.fit(X, y, groups=groups)
        except Exception:
            pass
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning) and "groups" in str(w.message).lower()]
    assert user_warnings, "strict_groups=False must still emit the warn-only fallback"


def test_mrmr_strict_groups_default_is_true():
    """Default now raises on groups (finding #20) -- matches sample_weight's always-consumed contract."""
    sel = MRMR()
    assert sel.strict_groups is True


def test_mrmr_strict_groups_no_op_when_groups_none():
    """strict_groups=True with groups=None must NOT raise."""
    X, y, _ = _tiny_xy()
    sel = MRMR(verbose=0, random_seed=42, strict_groups=True, fe_max_steps=0)
    sel.fit(X, y, groups=None)
    assert hasattr(sel, "support_")


def test_mrmr_strict_groups_setstate_default():
    """Legacy pickle without strict_groups attribute resurfaces with the current ctor default (True) --
    strict_groups is not in _SETSTATE_LEGACY_OVERRIDES, so the corrective default flip (finding #20)
    applies uniformly, including to resurrected legacy pickles (per project convention: a bug-fix
    default flip is not kept at the old wrong value 'for compatibility')."""
    sel = MRMR()
    state = sel.__getstate__() if hasattr(sel, "__getstate__") else dict(sel.__dict__)
    state.pop("strict_groups", None)
    sel2 = MRMR.__new__(MRMR)
    sel2.__setstate__(state)
    assert sel2.strict_groups is True
