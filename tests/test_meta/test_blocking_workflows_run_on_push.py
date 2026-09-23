"""Meta-test: a gate that mirrors a blocking local hook must be able to fail a PUSH, not only a PR.

This repository pushes straight to master and opens no pull requests. A workflow triggered only by
`pull_request` therefore fires never -- it reads in the file like a gate, appears in the workflow list
like a gate, and cannot fail anything.

Found 2026-09-23. A sweep disabled the push trigger on six workflows while a macOS pytest abort was being
chased (`audits/ci_review_2026-09-08/_TRACKER.md`, X5). `ci.yml` was restored afterwards; four ubuntu-only
workflows that run no tests at all -- and so could never have been affected by a macOS pytest crash --
were not. The consequence was not hypothetical: twelve whole-project mypy errors reached master between
2026-09-11 and 2026-09-22 while `mypy-full.yml` had no automatic trigger of any kind, and the published
documentation stopped deploying at the same time because `docs.yml` deploys on push.

The failure is quiet in both directions, which is why it needs a test rather than a comment: the local
hook still passes, the CI tab still lists the workflow, and the only visible symptom is that the workflow
has no recent runs -- something nobody checks until a push is already red for an unrelated reason.

A workflow may be exempt, but it must SAY it is, by name and with a reason, in `PUSH_EXEMPT` below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

#: Workflows that legitimately do not run on every push, each with the reason it does not.
#: A scheduled or manually-dispatched job is not a gate on a push and is not pretending to be one.
PUSH_EXEMPT: Dict[str, str] = {
    "codecov-full.yml": "scheduled coverage upload; the blocking coverage gate lives in ci.yml",
    "deep-nightly.yml": "nightly long-running sweep, far too slow for a push",
    "fs-benchmark-nightly.yml": "nightly measurement; a measurement that can fail a build gets skipped",
    "gpu-extras-install-matrix.yml": "install matrix over GPU extras, scheduled",
    "gpu-matrix.yml": "needs a GPU runner, scheduled",
    "numba-coverage.yml": "nightly; NUMBA_DISABLE_JIT makes it far slower than a push can absorb",
    "release.yml": "triggered by a tag, not by a push to master",
    "macos-abort-probe.yml": "on-demand probe for the X5 investigation",
    "dependabot-auto-merge.yml": "reacts to Dependabot pull requests by design",
    "dependency-review.yml": "GitHub's action compares a PR against its base; it has no push semantics",
    "dep-floors.yml": "resolves the declared floors; scheduled and dispatch, not a per-push gate",
}


def _workflow_files() -> List[Path]:
    """Return every workflow file, sorted, so a failure names the same file on every machine."""
    return sorted(WORKFLOWS.glob("*.yml"))


def _triggers(path: Path) -> Dict[str, Any]:
    """Return a workflow's `on:` mapping.

    PyYAML parses a bare `on` key as the BOOLEAN True (YAML 1.1 treats on/off as booleans), which is the
    trap that makes a naive `doc["on"]` lookup return nothing and this whole test pass vacuously. Both
    spellings are looked up.
    """
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), f"{path.name} is not a mapping"
    triggers = doc.get("on", doc.get(True))
    assert isinstance(triggers, dict), f"{path.name} has no parseable `on:` mapping"
    return triggers


def _pushes_to_master(triggers: Dict[str, Any]) -> bool:
    """Whether the workflow runs on a push to master."""
    push = triggers.get("push")
    if push is None:
        return False
    if not isinstance(push, dict):
        return True
    branches = push.get("branches")
    if branches is None:
        return True
    return "master" in [str(b) for b in branches]


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_a_workflow_either_runs_on_push_or_declares_why_not(path: Path) -> None:
    """Every workflow runs on a push to master, or names itself in `PUSH_EXEMPT` with a reason."""
    if path.name in PUSH_EXEMPT:
        assert PUSH_EXEMPT[path.name].strip(), f"{path.name} is exempt with an empty reason"
        return
    assert _pushes_to_master(_triggers(path)), (
        f"{path.name} does not run on a push to master. This repository opens no pull requests, so a "
        "pull_request-only trigger fires never and this workflow cannot fail anything. Restore its push "
        "block, or add it to PUSH_EXEMPT with the reason it is not a per-push gate."
    )


def test_the_four_gates_that_silently_stopped_running_are_covered() -> None:
    """The specific workflows the 2026-09-08 sweep left disabled must run on push and not be exempt.

    Named individually rather than left to the parametrised check above: the cheapest way for this
    regression to return is for one of them to be quietly moved into `PUSH_EXEMPT`, which the general
    test would then accept.
    """
    for name in ("mypy-full.yml", "black-filtered.yml", "docs.yml", "hooks-not-in-ci.yml"):
        path = WORKFLOWS / name
        assert path.exists(), f"{name} has gone missing"
        assert name not in PUSH_EXEMPT, f"{name} mirrors a blocking local hook and may not be exempted"
        assert _pushes_to_master(_triggers(path)), f"{name} must run on a push to master"


def test_no_workflow_is_left_disabled_by_the_temporary_x5_marker() -> None:
    """No workflow may carry the sweep's marker AND be unable to run on a push.

    The marker said "restore once macOS is confirmed stable", and on an ubuntu workflow that runs no
    tests there is no macOS result that could ever be the condition -- so a marker sitting next to a
    missing trigger is a disable nobody is tracking. `ci.yml` keeps its copy legitimately: there the
    marker narrows the push MATRIX to macOS while X5 is open, which is a live workaround on the one
    workflow X5 is actually about, not a gate that stopped running.
    """
    offenders: List[str] = []
    for path in _workflow_files():
        text = path.read_text(encoding="utf-8")
        if "TEMPORARY" in text and "X5" in text and not _pushes_to_master(_triggers(path)):
            offenders.append(path.name)
    assert offenders == [], f"these workflows are still disabled by the X5 temporary marker: {offenders}"


def test_mypy_full_stays_a_blocking_gate() -> None:
    """`advisory: false` is what makes the whole-project mypy run able to fail a push at all.

    Restoring the trigger is only half the fix: a workflow that runs on every push and reports advisory
    is the same silence with more log output.
    """
    text = (WORKFLOWS / "mypy-full.yml").read_text(encoding="utf-8")
    assert "advisory: false" in text, "mypy-full.yml must stay blocking; advisory would make its runs decorative"
