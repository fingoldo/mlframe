"""Mutation teeth over the production lines a commit changed.

For every ``src/mlframe`` module the diff against ``MLFRAME_MUTATION_BASE`` (default ``HEAD~1``) touched, the shared
harness mutates only the changed lines and checks that the module's own test files notice each mutant. A surviving
mutant is a change the tests do not observe; it is either fixed by an assertion or accepted in ``mutation_survivors.json``
with a note saying why it cannot be observed. The baseline is a ratchet: a new survivor fails, an accepted one passes.

A module with no ``test_<stem>.py`` anywhere under ``tests/`` is skipped here, since the test-source parity gate already
demands that a suite exists for it.

The sweep is slow (minutes per mutant), so it runs only when ``MLFRAME_RUN_MUTATION=1`` is set. Run one module by hand
the same way as in pyutilz: see the mutation section of CLAUDE.md. Use a non-editable install, or every mutant reads as a
survivor, and a relative path, since the harness refuses absolute ones.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from py_ci_shared.baseline_ratchet import Baseline
from py_ci_shared.git_changed_lines import changed_lines, lines_for
from py_ci_shared.mutation_teeth import assert_no_new_surviving_mutant

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"
BASE_REV = os.environ.get("MLFRAME_MUTATION_BASE", "HEAD~1")
BASELINE = Baseline("mutation_survivors", directory=str(Path(__file__).resolve().parent))


def _suites_for(module: Path) -> list[Path]:
    """Repo-relative ``test_*.py`` files under ``tests/`` that import *module* by its dotted path, shortest path first.

    A suite is any test file that names the module's dotted import path, so a module whose tests are not called
    ``test_<stem>.py`` (most of mlframe) still gets its suites; a file that never names the module is not its suite.
    """
    dotted = ".".join(module.with_suffix("").parts[1:])
    found = []
    for path in TESTS_DIR.rglob("test_*.py"):
        if dotted in path.read_text(encoding="utf-8", errors="replace"):
            found.append(path)
    found.sort(key=lambda p: (len(p.parts), p.as_posix()))
    return [p.relative_to(REPO_ROOT) for p in found]


def _changed_production_modules() -> list[Path]:
    """Repo-relative ``src/mlframe`` modules with a changed line and at least one suite under ``tests/``."""
    try:
        changed = changed_lines(REPO_ROOT, rev=BASE_REV, include_untracked=False)
    except (OSError, subprocess.CalledProcessError, RuntimeError):
        # A shallow or history-less checkout has no BASE_REV: changed_lines raises RuntimeError, and with no
        # diff there is nothing to sweep, so the meta-test skips instead of erroring at collection.
        return []
    modules = []
    for path in sorted(changed):
        if path.parts[:2] != ("src", "mlframe") or path.suffix != ".py" or path.name == "__init__.py":
            continue
        if not lines_for(changed, path):
            continue
        if _suites_for(path):
            modules.append(path)
    return modules


_MODULES = _changed_production_modules()


@pytest.mark.skipif(os.environ.get("MLFRAME_RUN_MUTATION") != "1", reason="mutation sweep runs only when MLFRAME_RUN_MUTATION=1 (it is minutes per mutant)")
@pytest.mark.skipif(not _MODULES, reason=f"no changed src/mlframe module with a suite under tests/ against {BASE_REV}")
@pytest.mark.parametrize("module", _MODULES, ids=lambda p: p.as_posix())
def test_changed_lines_are_killed_by_their_module_tests(module, request):
    """The changed lines of *module* are killed by its suites, or the survivor is accepted in the baseline."""
    changed = changed_lines(REPO_ROOT, rev=BASE_REV, include_untracked=False)
    assert_no_new_surviving_mutant(
        module,
        _suites_for(module),
        REPO_ROOT,
        BASELINE,
        request,
        lines=lines_for(changed, module),
    )
