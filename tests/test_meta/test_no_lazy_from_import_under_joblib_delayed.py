"""Meta-test (E1.1, 2026-05-22): no ``from X import name`` inside a function
that gets dispatched via ``joblib.delayed(...)`` to a threaded ``Parallel``.

The TVT-2026-05-21 prod log surfaced a NameError race in
_composite_screening_tiny: a lazy ``from .composite_estimator import
_y_train_clip_bounds`` inside a per-fold inner function ran under
``backend='threading'`` Parallel; two threads racing on the partial
``composite_estimator`` module left the local name unbound, the outer
``except Exception`` caught the NameError, and the fold silently
returned NaN. We fixed that one site by hoisting to module top, but the
SAME race class can re-appear anywhere a future contributor adds a
lazy import inside a joblib-dispatched callee.

This meta-test walks every ``.py`` under src/, finds every
``delayed(callee)(...)`` AST call, resolves ``callee`` to its
function-def if it's at module scope, and fails if the body contains a
``from X import name`` ImportFrom inside any nested function that
runs per-task.

To opt out at a site: add ``# joblib-import-race-ok`` to the
ImportFrom's source line (e.g. for delayed imports that are
demonstrably safe because they only touch already-loaded modules, or
because the dispatch uses backend != "threading"). Bare keep this
in mind:
- ``import X as Y`` (NOT ``from X import name``) is safe under racing
  threads — Python's import lock + binding to module object protects
  against partial-init races. Only ``from X import name`` carries the
  binding gotcha. The test below only flags ImportFrom for that reason.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from tests.test_meta._shared_ast_cache import parsed_ast, source_text

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe"
IGNORE_MARKER = "joblib-import-race-ok"


def _function_defs_by_name(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    """Return module-scope function defs keyed by name (top-level only -- nested
    closures don't matter because they can't be the ``delayed(callee)`` target
    referenced by name at module-level dispatch sites)."""
    out: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = node
    return out


def _find_lazy_from_imports(func: ast.FunctionDef, lines: list[str]) -> list[tuple[int, str]]:
    """Walk the function body + every nested function/closure; collect every
    ImportFrom node whose source line lacks the opt-out marker. Returns
    (lineno, source_line) tuples."""
    violations: list[tuple[int, str]] = []
    for node in ast.walk(func):
        if isinstance(node, ast.ImportFrom):
            ln = getattr(node, "lineno", 1)
            line_text = lines[ln - 1] if 0 < ln <= len(lines) else ""
            if IGNORE_MARKER in line_text:
                continue
            violations.append((int(ln), line_text.strip()))
    return violations


def _scan_module(path: pathlib.Path) -> list[tuple[str, int, str]]:
    """Return (callee_name, lineno, source_line) for every lazy-import
    violation inside a callee that this module passes to ``delayed(...)``."""
    tree = parsed_ast(path)
    if tree is None:
        return []
    func_defs = _function_defs_by_name(tree)
    src = source_text(path)
    lines = src.splitlines() if src is not None else []
    delayed_callees: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match ``delayed(callee)`` or ``X.delayed(callee)``.
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name not in ("delayed", "_delayed"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Name):
            delayed_callees.add(first.id)
        elif isinstance(first, ast.Attribute):
            delayed_callees.add(first.attr)

    out: list[tuple[str, int, str]] = []
    for callee in delayed_callees:
        if callee not in func_defs:
            # Callee is imported / on self / referenced by attribute -- skip.
            # The meta-test is best-effort; cross-module callees aren't
            # statically resolvable without a real type system.
            continue
        violations = _find_lazy_from_imports(func_defs[callee], lines)
        for lineno, line in violations:
            out.append((callee, lineno, line))
    return out


@pytest.mark.timeout(300)
def test_no_lazy_from_imports_inside_joblib_delayed_callees():
    """Walk src/mlframe; flag any ``from X import name`` inside a function
    dispatched via ``delayed(...)``. Opt out per-line with
    ``# joblib-import-race-ok`` if the import is demonstrably safe."""
    all_violations: list[tuple[str, str, int, str]] = []
    for path in SRC_ROOT.rglob("*.py"):
        for callee, lineno, line in _scan_module(path):
            rel = str(path.relative_to(SRC_ROOT.parent.parent))
            all_violations.append((rel, callee, lineno, line))
    if all_violations:
        msg_lines = [
            "E1.1 (2026-05-22) joblib-delayed callees must NOT carry ``from X import name`` "
            "inside their body -- two threads racing on a partial-init module leave the "
            "local binding unset and produce silent NameErrors (TVT-2026-05-21 root cause). "
            "Hoist to module top, or add ``# joblib-import-race-ok`` to the line.",
            "Violations:",
        ]
        for path, callee, ln, line in all_violations:
            msg_lines.append(f"  {path}:{ln}  in callee {callee!r}: {line}")
        pytest.fail("\n".join(msg_lines))
