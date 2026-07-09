"""Meta-test: the local mypy pre-commit hook must cover the FULL strict-mode beachhead.

Found 2026-07-09 (CI/CD architecture review): the beachhead had already expanded twice
(calibration -> +safe_pickle -> +_gpu_guard/_numba_params/rank_correlation/_core_precision_mape
in [[tool.mypy.overrides]]) without the pre-commit hook's ``files:`` regex ever being updated to
match. 5 of 6 beachhead modules were invisible to the local hook: an edit to any of them passed
``pre-commit run`` locally and only failed in CI's mypy-beachhead job -- a real, not hypothetical,
local/CI parity gap (the beachhead has already silently drifted like this at least once before).

This test parses pyproject.toml's actual [[tool.mypy.overrides]] list (the source of truth for
"what's in the beachhead": any override block with ``disallow_untyped_defs = true``) and asserts
every one of those files is matched by the pre-commit hook's regex, and also present in
mypy.yml's ``modules:`` list -- so a THIRD, silent drift point (someone expands the beachhead in
pyproject.toml but forgets either the pre-commit regex or the CI workflow's module list) fails
loudly here instead.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - repo's own floor is py39, but this test file itself runs under whatever collects it
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _beachhead_module_files() -> list[str]:
    """Every file covered by a [[tool.mypy.overrides]] block with disallow_untyped_defs=true,
    resolved to repo-relative paths against the real filesystem: a dotted module resolves to
    either a single "<path>.py" file or a "<path>" package directory -- the TOML alone can't
    tell which, so this checks what's actually on disk."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    overrides = data["tool"]["mypy"]["overrides"]
    files: list[str] = []
    for block in overrides:
        if not block.get("disallow_untyped_defs"):
            continue
        modules = block["module"]
        if isinstance(modules, str):
            modules = [modules]
        for mod in modules:
            mod = mod.removesuffix(".*")
            rel = "src/" + mod.replace(".", "/")
            if (REPO_ROOT / f"{rel}.py").is_file():
                files.append(f"{rel}.py")
            elif (REPO_ROOT / rel).is_dir():
                files.append(rel)
            else:
                raise AssertionError(f"Beachhead module {mod!r} (from pyproject.toml) resolves to neither {rel}.py nor a {rel}/ directory -- has it moved?")
    return files


def _precommit_mypy_regex() -> re.Pattern:
    with open(REPO_ROOT / ".pre-commit-config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for repo in config["repos"]:
        if "mirrors-mypy" not in repo.get("repo", ""):
            continue
        for hook in repo["hooks"]:
            if hook["id"] == "mypy":
                return re.compile(hook["files"])
    raise AssertionError("Could not find the mypy hook in .pre-commit-config.yaml -- did its structure change?")


def _mypy_workflow_modules() -> list[str]:
    with open(REPO_ROOT / ".github" / "workflows" / "mypy.yml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    raw = config["jobs"]["mypy"]["with"]["modules"]
    return [line.strip() for line in raw.strip().splitlines() if line.strip()]


def test_precommit_mypy_hook_covers_full_beachhead():
    """Every strict-mode beachhead module (from pyproject.toml) must be matched by the
    pre-commit hook's files= regex -- see this file's module docstring for the incident."""
    pattern = _precommit_mypy_regex()
    beachhead_dirs = _beachhead_module_files()
    unmatched = []
    for mod_path in beachhead_dirs:
        # Directory-style entries (e.g. "src/mlframe/calibration") need a representative file
        # to test the regex against; file-style entries already end in a real filename.
        probe = mod_path if mod_path.endswith(".py") else f"{mod_path}/__init__.py"
        if not pattern.search(probe):
            unmatched.append(probe)
    assert not unmatched, (
        f"The mypy pre-commit hook's files= regex does not match: {unmatched}. "
        f"These are strict-mode beachhead modules per [[tool.mypy.overrides]] in pyproject.toml "
        f"but invisible to the LOCAL pre-commit hook -- an edit to them will pass pre-commit and "
        f"only fail in CI's mypy-beachhead job. Update the regex in .pre-commit-config.yaml."
    )


def test_mypy_workflow_modules_match_beachhead():
    """mypy.yml's `modules:` list (what CI actually type-checks) must stay a superset of the
    strict-mode beachhead declared in pyproject.toml -- the two are meant to describe the same
    set, maintained by hand in two files, and can drift independently."""
    beachhead_dirs = set(_beachhead_module_files())
    workflow_modules = set(_mypy_workflow_modules())

    missing_from_workflow = []
    for mod_path in beachhead_dirs:
        # A beachhead dir entry (e.g. "src/mlframe/calibration") is satisfied by either the bare
        # directory or any file inside it appearing in the workflow's module list.
        if mod_path.endswith(".py"):
            covered = mod_path in workflow_modules
        else:
            covered = mod_path in workflow_modules or any(m.startswith(mod_path + "/") for m in workflow_modules)
        if not covered:
            missing_from_workflow.append(mod_path)

    assert not missing_from_workflow, (
        f"mypy.yml's `modules:` list is missing: {missing_from_workflow}. These are strict-mode "
        f"beachhead modules per [[tool.mypy.overrides]] in pyproject.toml but not type-checked by "
        f"CI's mypy-beachhead job -- update .github/workflows/mypy.yml's `modules:` input."
    )
