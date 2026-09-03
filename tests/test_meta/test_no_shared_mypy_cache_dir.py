"""X_CICD_DEPENDENCIES-4 (2026-08-05 audit): the mypy-full-manual pre-commit hook still used
``--cache-dir=../.mlframe_mypy_cache_shared`` (a cross-worktree shared cache), the exact pattern the two
sibling mypy hooks in this same file were already moved off on 2026-08-03 after proving it can make a
known-buggy revision silently report "Success" (a worktree on a fixed revision can inherit a stale/wrong
cache entry populated by a different, buggy worktree). Meta-test: no mypy hook in
``.pre-commit-config.yaml`` may reference a cache dir outside the checkout (``../...``), so this class of
regression fails CI instead of only surfacing after a real cross-worktree false "Success".
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_precommit_mypy_hook_uses_a_parent_relative_cache_dir():
    """No mypy hook's --cache-dir may point outside this checkout (a ../-relative path)."""
    text = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    offenders = re.findall(r"--cache-dir=(\.\./\S+)", text)
    assert not offenders, (
        f"pre-commit hook(s) reference a cache-dir outside this checkout, which can leak a "
        f"known-buggy worktree's stale cache entries into a different worktree's mypy run: {offenders}"
    )
