"""Cluster 39 regression: the lock module states that stale-lock reclaim is absent (audit 2026-11-20, finding 39-F41).

The docstring no longer promises a reclaim, and the reserved StaleLockReclaimed class is never raised by the lock code.
"""

from pathlib import Path

from mlframe.training.feature_handling import locking


def test_module_docstring_states_reclaim_is_not_performed():
    """The module docstring says reclaim is not performed, not that it is (39-F41)."""
    doc = locking.__doc__ or ""
    assert "Stale-lock reclaim is NOT performed" in doc
    assert "with stale-lock reclaim." not in doc


def test_lock_source_never_raises_stale_lock_reclaimed():
    """No raise of StaleLockReclaimed appears in the lock module source (39-F41)."""
    src = Path(locking.__file__).read_text(encoding="utf-8")
    assert "raise StaleLockReclaimed" not in src
    assert "warnings.warn(StaleLockReclaimed" not in src
