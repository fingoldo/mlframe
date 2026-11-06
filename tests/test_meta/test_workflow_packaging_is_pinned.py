"""Every CI install of ``packaging`` names an exact version.

The matrix jobs install ``packaging`` to decide which versions get tested. An unpinned install lets a bad release run in CI unnoticed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
_UNPINNED = re.compile(r"pip install packaging\s*$", re.MULTILINE)


@pytest.mark.parametrize("workflow", sorted(_WORKFLOWS.glob("*.yml")), ids=lambda p: p.name)
def test_packaging_install_has_an_exact_version(workflow):
    """No workflow installs ``packaging`` without a ``==`` pin."""
    assert not _UNPINNED.search(workflow.read_text(encoding="utf-8")), f"{workflow.name} installs packaging unpinned"
