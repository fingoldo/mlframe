"""L4.4 (audits/ci_review_2026-09-08/_TRACKER.md): pydoclint stays advisory-only in ``lint-advisory.yml``
because a straight flip to blocking would red the schedule on the ~1000-finding pre-existing backlog
rather than on a regression. This baselines that backlog the same way ``_source_text_baseline.json`` and
every other ``py_ci_shared.baseline_ratchet`` consumer in this directory do: the check fails only on a
finding absent from the frozen set, so a NEW docstring/signature mismatch is caught immediately while the
backlog is paid down separately (or not at all).

Refresh after fixing (or deliberately accepting) findings::

    pytest tests/test_meta/test_pydoclint_baseline.py --refresh-pydoclint-baseline
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from py_ci_shared.baseline_ratchet import Baseline

TEST_META_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_META_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src" / "mlframe"

_FINDING_RE = re.compile(r"^\s+(\d+): (DOC\d+): (.*)$")
REFRESH_FLAG = "--refresh-pydoclint-baseline"


def _run_pydoclint() -> dict[str, str]:
    """Run ``pydoclint src/mlframe`` and parse its text report into ``{path:line:code: message}``.

    pydoclint prints every scanned file's path unconditionally (even with zero findings) followed by an
    indented ``LINE: CODE: message`` line per violation, so only the indented lines are findings; the most
    recent unindented, non-"Skipping" line is the file the following findings belong to.
    """
    import shutil

    pydoclint_exe = shutil.which("pydoclint")
    if pydoclint_exe is None:
        # console-script entry point, not a runnable `python -m pydoclint` package.
        pydoclint_exe = str(Path(sys.executable).with_name("pydoclint"))
    proc = subprocess.run(
        [pydoclint_exe, "src/mlframe"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # pydoclint writes its report to stderr, not stdout.
    report = proc.stdout + proc.stderr
    assert report.strip(), f"pydoclint invocation produced no output at all (rc={proc.returncode})"
    found: dict[str, str] = {}
    current_file: str | None = None
    for line in report.splitlines():
        if line.startswith("Skipping files"):
            continue
        match = _FINDING_RE.match(line)
        if match:
            line_no, code, message = match.groups()
            key = f"{current_file}:{line_no}:{code}"
            found[key] = message.strip()
        elif line.strip() and not line.startswith(" "):
            current_file = line.strip().replace("\\", "/")
    return found


def regenerate_baseline(path: Path = TEST_META_DIR / "_pydoclint_baseline.json") -> None:
    """Rewrite the pydoclint baseline from today's findings, for ``regen_baselines.py``."""
    Baseline("_pydoclint_baseline", directory=str(TEST_META_DIR), refresh_command=f"pytest tests/test_meta/test_pydoclint_baseline.py {REFRESH_FLAG}").regenerate(_run_pydoclint())


def test_no_new_pydoclint_findings() -> None:
    """A docstring/signature mismatch absent from the frozen baseline fails the build."""
    baseline = Baseline("_pydoclint_baseline", directory=str(TEST_META_DIR), refresh_command=f"pytest tests/test_meta/test_pydoclint_baseline.py {REFRESH_FLAG}")
    found = _run_pydoclint()
    if REFRESH_FLAG in sys.argv:
        baseline.regenerate(found)
        return
    assert (
        baseline.enforce(
            found,
            label="pydoclint baseline",
            guidance="Fix the docstring (preferred), or if pydoclint is flagging a correct terse docstring per this repo's own style, add it with a note.",
        )
        == 0
    )
