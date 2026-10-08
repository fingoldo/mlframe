"""Structural gates over the CI workflows, the install extras they use, uv.lock and text-mode ``open``.

Each test parses YAML/TOML (or runs the real tool) and asserts on structure or behaviour, never on prose.

* master CI runs must not be cancelled by a newer push, or no commit ever gets a completed run;
* every workflow's installed extras must be enough for ``import mlframe`` and for the entry points the workflow runs;
* a self-hosted runner executes repo code, so only a manual dispatch may reach it;
* uv.lock records pyutilz as a git commit from the branch pyproject names;
* no ``open()`` / ``Path.read_text`` / ``Path.write_text`` in src relies on the platform default encoding.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest
import yaml
from packaging.requirements import Requirement
from py_ci_shared.ci_default_branch_never_cancelled import assert_ci_default_branch_never_cancelled
from py_ci_shared.ci_install_covers_entry_imports import assert_ci_install_covers_entry_imports, assert_entry_imports_without_extras

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from tests.test_meta._scan_guard import assert_scanned_enough

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
EXTRAS_RE = re.compile(r"\.\[([^\]]+)\]")


def _load_workflow(name: str) -> Dict[Any, Any]:
    """Parse one workflow file into a dict."""
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _all_workflows() -> List[Path]:
    """Every workflow file, failing closed when the directory is not where it should be."""
    files = sorted(WORKFLOWS.glob("*.yml"))
    assert_scanned_enough(len(files), "workflow files", minimum=10)
    return files


def _triggers(doc: Dict[Any, Any]) -> Set[str]:
    """The event names of a workflow (YAML 1.1 reads the ``on`` key as the boolean True)."""
    on = doc.get("on", doc.get(True))
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return set(on)
    return set(on)


def _steps(doc: Dict[Any, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """(job name, step) pairs of every step in the workflow."""
    return [(job_name, step) for job_name, job in doc.get("jobs", {}).items() for step in job.get("steps", [])]


def _workflow_extras(doc: Dict[Any, Any]) -> Set[str]:
    """Extras this workflow installs, from ``project-extras`` inputs and ``.[extras]`` in run scripts."""
    extras: Set[str] = set()
    for _, step in _steps(doc):
        specs = [str(step.get("with", {}).get("project-extras", ""))]
        specs.append(str(step.get("run", "")))
        for spec in specs:
            for group in EXTRAS_RE.findall(spec):
                extras.update(part.strip() for part in group.split(",") if part.strip())
    return extras


def _pyproject() -> Dict[str, Any]:
    """The parsed pyproject.toml."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


def _expand_extras(requested: Set[str], optional: Dict[str, List[str]]) -> Set[str]:
    """Distribution names provided by the requested extras, following ``mlframe[...]`` self-references."""
    seen: Set[str] = set()
    dists: Set[str] = set()
    todo = list(requested)
    while todo:
        extra = todo.pop()
        if extra in seen:
            continue
        seen.add(extra)
        for spec in optional.get(extra, []):
            req = Requirement(spec)
            if req.name.lower() == "mlframe":
                todo.extend(req.extras)
            else:
                dists.add(req.name.lower().replace("_", "-"))
    return dists


def _dry_run_entry_points(doc: Dict[Any, Any]) -> List[List[str]]:
    """``python -m mlframe...`` invocations of the workflow that carry ``--dry-run``, as argv lists (expressions read as ``nightly``)."""
    found: List[List[str]] = []
    for _, step in _steps(doc):
        script = re.sub(r"\$\{\{.*?\}\}", "nightly", str(step.get("run", "")).replace("\\\n", " "))
        for line in script.splitlines():
            if "python -m mlframe" in line and "--dry-run" in line:
                argv = [tok.strip('"') for tok in line.split()]
                found.append(argv[argv.index("-m") + 1 :])
    return found


def test_master_ci_runs_are_never_cancelled_by_a_newer_push() -> None:
    """Only pull_request events may cancel an in-progress CI run; a push to master must run to completion."""
    group = _load_workflow("ci.yml")["concurrency"]
    cancel = group["cancel-in-progress"]
    if isinstance(cancel, bool):
        assert cancel is False
        return
    expr = str(cancel).strip()
    assert expr.startswith("${{") and expr.endswith("}}"), expr
    body = expr[3:-2].strip()
    assert body == "github.event_name == 'pull_request'", f"cancel-in-progress must be limited to pull_request, got {body!r}"


def test_workflow_install_extras_exist_in_pyproject() -> None:
    """Every extra a workflow installs is declared in pyproject.toml's optional-dependencies."""
    optional = _pyproject()["project"]["optional-dependencies"]
    seen = 0
    for path in _all_workflows():
        for extra in _workflow_extras(_load_workflow(path.name)):
            if "${{" in extra or "$" in extra:
                continue
            seen += 1
            assert extra in optional, f"{path.name} installs unknown extra {extra!r}"
    assert seen >= 5, "no workflow installs any extra; the extras parser reads nothing"


def test_the_benchmark_workflow_installs_the_signal_extra_wavelet_code_needs() -> None:
    """The nightly benchmark installs the extras providing pywavelets (lazy wavelet arms) and the boosters its dry run imports."""
    optional = _pyproject()["project"]["optional-dependencies"]
    provided = _expand_extras(_workflow_extras(_load_workflow("fs-benchmark-nightly.yml")), optional)
    assert {"pywavelets", "catboost"} <= provided


@pytest.mark.parametrize("workflow", ["fs-benchmark-nightly.yml", "deep-nightly.yml", "ci.yml"])
def test_workflow_install_is_enough_to_import_mlframe_and_run_its_dry_run_entry(workflow: str) -> None:
    """With every extra the workflow does NOT install unimportable, ``import mlframe`` and its dry-run entry still start."""
    doc = _load_workflow(workflow)
    optional = _pyproject()["project"]["optional-dependencies"]
    installed_dists = _expand_extras(_workflow_extras(doc), optional)
    # An extra shares no distribution with the install exactly when blocking its libraries cannot break something the job provides.
    blocked_extras = sorted(extra for extra in optional if not _expand_extras({extra}, optional) & installed_dists)
    assert blocked_extras, "nothing to block; the test would pass vacuously"
    assert_entry_imports_without_extras(REPO_ROOT, ["mlframe", *_dry_run_entry_points(doc)], blocked_extras=blocked_extras)


def test_every_workflow_installs_what_the_modules_it_runs_import() -> None:
    """The extras a job installs cover the module-level imports of each entry command (python -m, python script.py).

    Test files are not entries here: following every one of them multiplies the import walk several times over, and the extras a test
    module needs are already judged by the conftest install gate and by importorskip in the tests themselves.
    """
    assert_ci_install_covers_entry_imports(REPO_ROOT, include_pytest=False, min_files=10)


def test_no_workflow_that_runs_on_a_master_push_cancels_its_own_master_runs() -> None:
    """Every workflow that fires on a push to master either cancels pull requests only or says why it may cancel."""
    assert_ci_default_branch_never_cancelled(REPO_ROOT, default_branches=["master"], min_files=10)


def test_self_hosted_runners_are_reachable_only_by_manual_dispatch() -> None:
    """A workflow with a self-hosted job fires on workflow_dispatch alone, never on push, pull_request or schedule."""
    checked = 0
    for path in _all_workflows():
        doc = _load_workflow(path.name)
        for job in doc.get("jobs", {}).values():
            runs_on = job.get("runs-on")
            labels = runs_on if isinstance(runs_on, list) else [runs_on]
            if "self-hosted" in [str(label) for label in labels]:
                checked += 1
                assert _triggers(doc) == {"workflow_dispatch"}, f"{path.name} exposes a self-hosted runner to {_triggers(doc)}"
    assert checked >= 1, "no self-hosted job found; the GPU workflow moved or was renamed"


def test_uv_lock_records_pyutilz_as_a_pinned_commit_of_the_branch_pyproject_names() -> None:
    """uv.lock stores pyutilz as a 40-hex commit of the repository and branch in [tool.uv.sources]."""
    source = _pyproject()["tool"]["uv"]["sources"]["pyutilz"]
    with open(REPO_ROOT / "uv.lock", "rb") as fh:
        lock = tomllib.load(fh)
    entries = [pkg for pkg in lock["package"] if pkg["name"] == "pyutilz"]
    assert len(entries) == 1
    git = entries[0]["source"]["git"]
    match = re.fullmatch(r"(?P<repo>[^?#]+)\?branch=(?P<branch>[^#]+)#(?P<sha>[0-9a-f]{40})", git)
    assert match, git
    assert match["repo"] == source["git"]
    assert match["branch"] == source["branch"]


def _ruff_cmd() -> List[str]:
    """Command prefix that runs the pinned ruff."""
    exe = shutil.which("ruff")
    return [exe] if exe else [sys.executable, "-m", "ruff"]


def test_no_text_mode_open_in_src_relies_on_the_platform_default_encoding() -> None:
    """ruff PLW1514 (unspecified-encoding) reports nothing under src/mlframe: text I/O names its encoding."""
    src = REPO_ROOT / "src" / "mlframe"
    assert_scanned_enough(len(list(src.rglob("*.py"))), "mlframe source files")
    proc = subprocess.run(
        [*_ruff_cmd(), "check", str(src), "--isolated", "--preview", "--select", "PLW1514", "--output-format", "concise"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
