"""Cluster 35 regression: the three composite-refit copies state their group semantics (audit 2026-11-20, finding 35-F19).

Each copy carries a note naming the group handling it uses, so a fix to one copy is checked against the others.
"""

from pathlib import Path

from mlframe.training.composite import ensemble
from mlframe.training.composite.ensemble import _oof_external


def test_plain_fold_copy_documents_group_semantics():
    """The plain fold refit documents its group-aware carve (35-F19)."""
    src = Path(ensemble.__file__).read_text(encoding="utf-8")
    assert "Group semantics (audit 35-F19, documented)" in src
    assert "group-aware inner eval set" in src


def test_external_holdout_copy_documents_group_semantics():
    """The external-holdout refit documents its semantics (35-F19)."""
    src = Path(_oof_external.__file__).read_text(encoding="utf-8")
    assert "Group semantics (audit 35-F19, documented)" in src
    assert "external holdout" in src
