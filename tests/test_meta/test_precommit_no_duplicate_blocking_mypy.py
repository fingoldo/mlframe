"""X_CICD_DEPENDENCIES-7 regression test: mypy-full-manual must not also run blocking on every commit.

mypy-full (whole-project) already runs blocking on every pre-commit/pre-merge-commit and is a strict
superset of mypy-full-manual's changed-files check. Having both hooks in the blocking stages doubled
mypy wall time paid per commit for zero extra coverage. This is a source-inspection test (a YAML config
constant, not something a Python test can invoke and assert on behaviorally) -- pinned so a future edit
accidentally re-adding the blocking stages override is caught.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_precommit_config():
    """Parse .pre-commit-config.yaml and return the loaded dict."""
    return yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))


def _find_hook(config, hook_id):
    """Return the hook dict with the given id from any repo block in the parsed pre-commit config, or None."""
    for repo in config["repos"]:
        for hook in repo.get("hooks", []):
            if hook.get("id") == hook_id:
                return hook
    return None


def test_mypy_full_manual_is_not_blocking_on_every_commit():
    """mypy-full-manual must not carry a pre-commit/pre-merge-commit stages override -- it should fall
    back to the shared hook definition's own default (manual-only), since mypy-full already covers it."""
    config = _load_precommit_config()
    hook = _find_hook(config, "mypy-full-manual")
    assert hook is not None, "mypy-full-manual hook not found in .pre-commit-config.yaml"
    stages = hook.get("stages")
    assert stages != ["pre-commit", "pre-merge-commit"], (
        f"mypy-full-manual still overrides stages to run blocking on every commit ({stages}), duplicating "
        f"mypy-full's whole-project check for zero extra coverage"
    )


def test_mypy_full_still_runs_blocking_on_every_commit():
    """Sanity: the whole-project mypy-full hook (the actual coverage-providing one) must still be blocking."""
    config = _load_precommit_config()
    hook = _find_hook(config, "mypy-full")
    assert hook is not None, "mypy-full hook not found in .pre-commit-config.yaml"
    assert hook.get("stages") == ["pre-commit", "pre-merge-commit"]
