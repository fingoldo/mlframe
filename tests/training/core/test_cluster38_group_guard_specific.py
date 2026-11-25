"""Cluster 38 regression: the pre-screen group guard checks the group column itself (audit 2026-11-20, finding 38-F9).

The old guard tested ``not _protected``, which target names always fill, so it never fired. The guard now tests whether the group column's
name is protected. The explicit group-name field from the extractor is still open (recorded as PARTIAL).
"""

from pathlib import Path

from mlframe.training.core import _phase_train_one_target_pre_screen as ps


def test_guard_tests_the_group_column_not_the_whole_protected_set():
    """The skip condition uses the group column's own protection flag (38-F9)."""
    src = Path(ps.__file__).read_text(encoding="utf-8")
    assert "if _split_cfg_use_groups and not _group_protected:" in src
    assert "if _split_cfg_use_groups and not _protected:" not in src
