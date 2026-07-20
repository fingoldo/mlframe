"""Advisory report: complex PRIVATE functions/methods without a docstring.

[tool.interrogate]'s ``ignore-private = true`` is a sound blanket default (avoids reopening the
whole 100%-coverage campaign for every trivial internal helper), but it has a real blind spot: a
private function can be genuinely complex (this codebase's own mccabe threshold comment already
notes a median complexity of 30, with real outliers past 100) and STILL carry zero docstring,
since interrogate never looks at it. Flagged in the 2026-07-18 architecture review as worth
closing, but lower priority than a live bug -- this is a coverage-SHAPE gap, not a correctness
one, so it stays advisory (a warning, not a failure) rather than blocking the whole test suite on
writing docstrings for dozens of legacy complex private functions in one pass.

Cross-references ruff's own C901 (mccabe complexity, --select C90) findings against each flagged
function's actual docstring presence (ast.get_docstring) -- reuses the SAME threshold as
[tool.ruff.lint.mccabe] max-complexity so "complex" means the same thing here as it does in the
mccabe advisory report already wired into CI (lint-advisory.yml).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import orjson

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ruff_c901_findings() -> list[dict]:
    """Every C901 finding under src/mlframe, at the repo's own configured mccabe threshold."""
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src/mlframe", "--select", "C90", "--output-format", "json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if not proc.stdout.strip():
        return []
    return orjson.loads(proc.stdout)


def _has_docstring(file_path: Path, lineno: int) -> bool:
    """Whether the function/method definition starting at (or containing) ``lineno`` has a docstring."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno == lineno:
            return ast.get_docstring(node) is not None
    return True  # couldn't locate the node precisely -- don't false-positive-flag it


def test_report_complex_private_functions_missing_docstrings():
    """Advisory (never fails): logs every private (leading-underscore) function/method that ruff's
    mccabe check flags as complex AND that has no docstring, so this gap is visible without
    blocking the suite on writing dozens of docstrings for legacy complex helpers in one pass."""
    findings = _ruff_c901_findings()
    undocumented = []
    for finding in findings:
        func_name = finding["message"].split("`")[1] if "`" in finding["message"] else ""
        if not func_name.startswith("_") or func_name.startswith("__"):
            continue  # only private (single-leading-underscore); dunders are interrogate's ignore-magic territory already
        file_path = Path(finding["filename"])
        lineno = finding["location"]["row"]
        if not _has_docstring(file_path, lineno):
            undocumented.append(f"{file_path.relative_to(REPO_ROOT)}:{lineno} {func_name} (complexity in: {finding['message']})")

    if undocumented:
        print(
            f"\n[complex-private-docstrings] {len(undocumented)} complex private function(s) have no docstring "
            f"(advisory only, not blocking):\n  " + "\n  ".join(undocumented),
            file=sys.stderr,
        )
    # Advisory: no assertion. The print above is the signal; see this file's module docstring for why.
