"""A numba cache restore in CI matches the source hash exactly; a prefix fallback can serve kernels compiled against an older callee.

``_numba_cache_deps`` stamps only ``from X import fn`` callees, so a kernel that reaches a changed callee another way keeps
machine code compiled against the old body. A ``restore-keys`` prefix fallback hands such a stale cache to a run with changed
sources, and the failure is wrong results rather than a crash.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))
_NUMBA_CACHE_STEP = re.compile(r"actions/cache@[^\n]*\n(?:(?!- name:).*\n)*?\s+path: \$\{\{ github\.workspace \}\}/\.numba-cache\n(?:(?!- name:).*\n)*", re.MULTILINE)


def _numba_cache_steps() -> list[tuple[str, str]]:
    """Every actions/cache step whose path is the numba cache directory, as (workflow name, step text)."""
    found = []
    for wf in _WORKFLOWS:
        text = wf.read_text(encoding="utf-8")
        for match in _NUMBA_CACHE_STEP.finditer(text):
            found.append((wf.name, match.group(0)))
    return found


def test_the_numba_cache_is_restored_by_exact_key_only():
    """No workflow restores the numba cache through a restore-keys prefix fallback."""
    steps = _numba_cache_steps()
    assert steps, "expected at least one numba cache restore step in the workflows"
    stale = [name for name, step in steps if "restore-keys" in step]
    assert not stale, f"numba cache restore with a prefix fallback in {stale}; a stale cache can serve kernels compiled against old callees"
