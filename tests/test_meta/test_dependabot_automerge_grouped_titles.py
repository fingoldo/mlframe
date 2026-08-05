"""X_CICD_DEPENDENCIES-3 (2026-08-05 audit): dependabot-auto-merge.yml's eligibility check required a
single "from X.Y.Z to A.B.C" pair in the PR title, but dependabot.yml groups the pip ecosystem's
minor/patch bumps into one PR titled "Bump the python-dependencies group ..." (no such pair to parse),
so auto-merge silently never fired for the majority of PRs it exists to handle. Fixed by fast-pathing
that specific group name as eligible (safe because dependabot.yml's own `update-types: [minor, patch]`
restriction on this group already guarantees no major bump can land inside it) while leaving the
github-actions group (unrestricted `patterns: ["*"]`, can include majors) still gated through manual
review. This test extracts the workflow's actual shell script and runs it against representative titles
so the logic itself is pinned, not just re-implemented in Python.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "dependabot-auto-merge.yml"


def _run_eligibility_script(pr_title: str) -> str:
    """Extract the workflow's 'Decide whether this is a minor/patch bump' step's shell script and run
    it in a subshell with PR_TITLE set, returning the eligible= value written to GITHUB_OUTPUT."""
    doc = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    script = doc["jobs"]["auto-merge"]["steps"][0]["run"]

    output_file = REPO_ROOT / "tests" / "test_meta" / "_tmp_github_output.txt"
    try:
        env = {"PR_TITLE": pr_title, "GITHUB_OUTPUT": str(output_file), "PATH": __import__("os").environ["PATH"]}
        subprocess.run(["bash", "-c", script], env=env, check=True, capture_output=True, text=True)
        content = output_file.read_text(encoding="utf-8")
    finally:
        output_file.unlink(missing_ok=True)

    for line in content.splitlines():
        if line.startswith("eligible="):
            return line.split("=", 1)[1]
    raise AssertionError(f"script did not write an eligible= line for title {pr_title!r}")


def test_grouped_python_dependencies_title_is_eligible():
    """The exact title shape GitHub generates for a dependabot group PR must now be auto-merge eligible."""
    assert _run_eligibility_script("Bump the python-dependencies group across 1 directory with 3 updates") == "true"


def test_grouped_github_actions_title_stays_ineligible():
    """The github-actions group has no minor/patch restriction, so it must NOT be fast-pathed."""
    assert _run_eligibility_script("Bump the github-actions group with 2 updates") == "false"


def test_ungrouped_minor_patch_bump_is_eligible():
    """The original single-dependency from-X-to-Y path must still work."""
    assert _run_eligibility_script("Bump requests from 2.28.1 to 2.31.0") == "true"


def test_ungrouped_major_bump_stays_ineligible():
    """A major-version bump must still require manual review."""
    assert _run_eligibility_script("Bump numpy from 1.24.0 to 2.0.0") == "false"
