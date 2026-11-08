"""A CI workflow's ``pyutilz-ref`` input is read from uv.lock, never a hand-copied commit SHA.

uv.lock is the single source for the pyutilz revision. A literal 40-hex value in a ``pyutilz-ref:`` input would be a second copy that
``uv lock --upgrade-package pyutilz`` does not move, so the install would silently stay on the old commit.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.test_meta._scan_guard import assert_scanned_enough

REPO = Path(__file__).resolve().parents[2]
_LITERAL_SHA = re.compile(r"[0-9a-f]{40}")


def test_workflow_pyutilz_ref_inputs_are_not_literal_shas():
    """Every ``pyutilz-ref:`` value in every workflow is an expression that reads the lock step, not a literal commit."""
    refs = []
    workflows = sorted((REPO / ".github" / "workflows").glob("*.yml"))
    assert_scanned_enough(len(workflows), ".github/workflows", minimum=15)
    for wf in workflows:
        for line in wf.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("pyutilz-ref:"):
                refs.append((wf.name, line.strip()))

    assert refs, "expected at least one pyutilz-ref input in the workflows"
    literal = [(name, line) for name, line in refs if _LITERAL_SHA.search(line)]
    assert not literal, f"hand-copied pyutilz SHA(s); read them from uv.lock instead: {literal}"
