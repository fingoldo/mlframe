"""Guard-rail regression for 05_concurrency_and_statistics.md finding #4.

``_evaluate_candidates_inner`` (in ``_evaluation_driver.py``) assumes the caller has already
republished the main thread's MI thread-local toggles (SU/JMIM/bur-lambda/relax-mrmr/PID/CMI-perm/
CPT/Miller-Madow/group-MI) into the current thread -- ``evaluate_candidates`` is the ONLY function
that does this republish/restore dance before calling ``_evaluate_candidates_inner``. Nothing in the
language enforces that discipline; a future joblib dispatch site added elsewhere in the codebase that
calls ``_evaluate_candidates_inner`` directly (bypassing ``evaluate_candidates``) would silently run
with default (False/0.0) toggles regardless of the caller's actual settings. This test is a static
tripwire: it fails the day a second call site appears anywhere in ``src/mlframe`` outside
``_evaluation_driver.py`` itself.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

def _find_src_root() -> Path:
    """Walk up from this test file to the repo root (the first ancestor containing ``src/mlframe``),
    robust to running from a plain clone or a worktree checkout at an arbitrary nesting depth."""
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / "src" / "mlframe"
        if candidate.is_dir():
            return candidate
    raise RuntimeError(f"could not locate src/mlframe above {__file__}")


_SRC_ROOT = _find_src_root()
_ALLOWED_CALLER_SUFFIX = "_evaluation_driver.py"


def _calls_evaluate_candidates_inner(path: Path) -> bool:
    """True if ``path`` contains a direct call/reference to ``_evaluate_candidates_inner``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "_evaluate_candidates_inner":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "_evaluate_candidates_inner":
            return True
    return False


@pytest.mark.fast
def test_evaluate_candidates_inner_has_no_bypass_callers():
    """Only ``_evaluation_driver.py`` (which defines it AND republishes the thread-locals before
    calling it) may reference ``_evaluate_candidates_inner``. Any other caller has bypassed the
    thread-local republish/restore contract."""
    assert _SRC_ROOT.is_dir(), f"expected src root at {_SRC_ROOT}"
    offenders = [
        str(path.relative_to(_SRC_ROOT))
        for path in _SRC_ROOT.rglob("*.py")
        if not str(path).endswith(_ALLOWED_CALLER_SUFFIX) and _calls_evaluate_candidates_inner(path)
    ]
    assert not offenders, (
        f"_evaluate_candidates_inner referenced outside _evaluation_driver.py in {offenders} -- "
        "any new call site MUST go through evaluate_candidates() so the MI thread-local toggles "
        "are republished into the worker thread first (05_concurrency_and_statistics.md finding #4)."
    )
