"""Meta-test: pre-commit's src-scoped vs tests-scoped hook regexes must partition, not gap/overlap.

Found 2026-07-17 (repeatedly, across black-filtered/bandit/interrogate/codespell): the src-scoped
local hooks' ``exclude:`` regex and the corresponding CI job's ``source-path:``/``check-path:``
scoping were hand-maintained comments claiming "must match CI" with nothing enforcing it -- a
tests/-only commit silently skipped both the src-scoped hook (excluded) AND had no tests-scoped
hook to catch it (didn't exist yet), so tests/ went completely unchecked locally and in CI for
these four tools until each gap was found and fixed one at a time. Now that a tests-scoped
sibling hook exists for each of the four, THIS test asserts the src-hook's ``exclude`` and the
tests-hook's ``files`` regex partition a representative file cleanly: a src/mlframe file matches
exactly the src-scoped hook, a tests/ file matches exactly the tests-scoped hook -- neither a gap
(matched by neither) nor an overlap (matched by both, which would just be wasted double-work but
still worth catching since it signals the regexes have drifted from their intended scope).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# (src-scoped hook id, tests-scoped hook id) pairs that are meant to jointly cover the whole repo.
PAIRED_HOOKS = [
    ("black-filtered-blocking", "black-filtered-tests-blocking"),
    ("bandit-blocking", "bandit-tests-blocking"),
    ("interrogate-blocking", "interrogate-tests-blocking"),
    ("codespell-blocking", "codespell-tests-blocking"),
]

SRC_SAMPLE = "src/mlframe/feature_selection/filters/_mrmr_class.py"
TESTS_SAMPLE = "tests/training/test_trainer.py"


def _load_precommit_hooks() -> dict:
    """Maps hook id -> its pre-commit config dict (repo entry's hooks: list, flattened)."""
    with open(REPO_ROOT / ".pre-commit-config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    hooks = {}
    for repo in config["repos"]:
        for hook in repo["hooks"]:
            hooks[hook["id"]] = hook
    return hooks


def _hook_matches(hook: dict, path: str) -> bool:
    """Whether pre-commit would run this hook against ``path``, per its files=/exclude= regexes.

    Mirrors pre-commit's own matching semantics: a file must match ``files`` (default match-all)
    AND must NOT match ``exclude`` (default match-nothing).
    """
    files_pattern = hook.get("files", "")
    exclude_pattern = hook.get("exclude", "")
    if files_pattern and not re.search(files_pattern, path):
        return False
    if exclude_pattern and re.search(exclude_pattern, path):
        return False
    return True


def test_paired_precommit_hooks_partition_src_and_tests_cleanly():
    """Every (src-hook, tests-hook) pair must match SRC_SAMPLE XOR TESTS_SAMPLE, never both/neither."""
    hooks = _load_precommit_hooks()
    problems = []
    for src_id, tests_id in PAIRED_HOOKS:
        if src_id not in hooks:
            problems.append(f"{src_id}: hook not found in .pre-commit-config.yaml")
            continue
        if tests_id not in hooks:
            problems.append(f"{tests_id}: hook not found in .pre-commit-config.yaml")
            continue
        src_hook, tests_hook = hooks[src_id], hooks[tests_id]

        src_hits = [name for name, hook in (("src-hook", src_hook), ("tests-hook", tests_hook)) if _hook_matches(hook, SRC_SAMPLE)]
        if src_hits != ["src-hook"]:
            problems.append(f"{SRC_SAMPLE} matched by {src_hits or 'neither hook'} (expected exactly src-hook={src_id})")

        tests_hits = [name for name, hook in (("src-hook", src_hook), ("tests-hook", tests_hook)) if _hook_matches(hook, TESTS_SAMPLE)]
        if tests_hits != ["tests-hook"]:
            problems.append(f"{TESTS_SAMPLE} matched by {tests_hits or 'neither hook'} (expected exactly tests-hook={tests_id})")

    assert not problems, "Pre-commit src/tests hook scoping has drifted out of a clean partition:\n  " + "\n  ".join(problems)
