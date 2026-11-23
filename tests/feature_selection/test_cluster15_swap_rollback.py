"""Cluster 15 regression: a failed SFFS swap pass rolls its state back before the argmax reads it (audit 2026-11-20, finding 15-F16).

Pinned: _restore_in_place keeps the dict's identity and restores its contents, and the swap stage snapshots the three dicts before the pass
and restores them in the except handler.
"""

from pathlib import Path

from mlframe.feature_selection.wrappers.rfecv import _finalize as fin

_SRC = Path(fin.__file__).resolve()


def test_restore_in_place_keeps_identity_and_contents():
    """The helper restores the snapshot into the same dict object (15-F16)."""
    target = {1: [0], 2: [0, 1]}
    alias = target
    snapshot = {1: [0], 2: [0]}
    fin._restore_in_place(target, snapshot)
    assert target is alias
    assert target == {1: [0], 2: [0]}


def test_swap_stage_snapshots_before_and_restores_on_failure():
    """The snapshot precedes the swap call and the rollback sits in its except handler (15-F16)."""
    text = _SRC.read_text(encoding="utf-8")
    snap = text.index("_snap_sel = {k: list(v)")
    call = text.index("self._sffs_swap_pass(")
    rollback = text.index("_restore_in_place(selected_features_per_nfeatures, _snap_sel)")
    assert snap < call < rollback
    assert "state rolled back, continuing" in text
