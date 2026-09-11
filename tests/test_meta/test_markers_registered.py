"""Every pytest marker used under tests/ is registered.

`py_ci_shared.pytest_markers` does the work. Docstrings are scanned too, because a snippet there gets
pasted into a real decorator. `expect_registered` names markers this repository registers, so a parser
that stops reading pyproject or the conftest fails here instead of reporting a clean tree.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.pytest_markers import assert_markers_registered

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_every_used_marker_is_registered() -> None:
    """No marker is used under tests/ that pyproject or a conftest does not register."""
    assert_markers_registered(REPO_ROOT, expect_registered=("slow", "fast", "gpu", "no_xdist", "fuzz", "biz_transformer"))
