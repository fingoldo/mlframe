"""A CI workflow's ``pyutilz-ref`` input is read from uv.lock, never a hand-copied commit SHA.

uv.lock is the single source for the pyutilz revision. A literal 40-hex value in a ``pyutilz-ref:`` input would be a second copy that
``uv lock --upgrade-package pyutilz`` does not move, so the install would silently stay on the old commit.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_LITERAL_SHA = re.compile(r"[0-9a-f]{40}")


def test_workflow_pyutilz_ref_inputs_are_not_literal_shas():
    """Every ``pyutilz-ref:`` value in every workflow is an expression that reads the lock step, not a literal commit."""
    refs = []
    for wf in sorted((REPO / ".github" / "workflows").glob("*.yml")):
        for line in wf.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("pyutilz-ref:"):
                refs.append((wf.name, line.strip()))

    assert refs, "expected at least one pyutilz-ref input in the workflows"
    literal = [(name, line) for name, line in refs if _LITERAL_SHA.search(line)]
    assert not literal, f"hand-copied pyutilz SHA(s); read them from uv.lock instead: {literal}"
