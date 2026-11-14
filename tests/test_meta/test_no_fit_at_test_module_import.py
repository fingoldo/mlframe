"""No test module trains a model while it is being imported.

Every shard imports every test module to collect it, so a fit at import time is paid by all shards before the first test, outside any
test timeout. One full fit there kept the macOS shards in collection until the job cap cancelled them, three hours with nothing run.
A one-off setup belongs in a fixture, which runs only when a test that needs it is selected.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CALLS = {"fit", "fit_transform", "fit_predict", "train_mlframe_models_suite"}


def _call_name(node: ast.Call) -> str:
    """Name of the called function or method, or an empty string for any other callee."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _trains(nodes: list) -> bool:
    """True when any statement in ``nodes`` calls a training entry point, looking through nested blocks but not into nested definitions."""
    for node in nodes:
        stack = [node]
        while stack:
            current = stack.pop()
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(current, ast.Call) and _call_name(current) in TRAINING_CALLS:
                return True
            stack.extend(ast.iter_child_nodes(current))
    return False


def _is_main_guard(node: ast.stmt) -> bool:
    """True for ``if __name__ == "__main__":``, whose body does not run when the module is imported."""
    return isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"


def _calls_outside_lambdas(node: ast.AST):
    """Every call under ``node`` that runs when the node does: a lambda body is not run until the lambda is called."""
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, ast.Lambda):
            continue
        if isinstance(current, ast.Call):
            yield current
        stack.extend(ast.iter_child_nodes(current))


def _import_time_fits(tree: ast.Module) -> list:
    """Lines of the module-level statements that train, directly or through a function defined in the same module."""
    training_functions = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _trains(n.body)}
    lines = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or _is_main_guard(node):
            continue
        if _trains([node]):
            lines.append(node.lineno)
            continue
        if any(isinstance(call.func, ast.Name) and call.func.id in training_functions for call in _calls_outside_lambdas(node)):
            lines.append(node.lineno)
    return lines


def test_import_time_scan_sees_a_fit_run_through_a_module_function():
    """The scan reports a module-level call to a helper that fits, which is the shape that stalled collection."""
    source = "def warm():\n    model.fit(x, y)\n\nwarm()\n"
    assert _import_time_fits(ast.parse(source)) == [4]
    assert _import_time_fits(ast.parse("def warm():\n    model.fit(x, y)\n")) == []
    assert _import_time_fits(ast.parse("model.fit(x, y)\n")) == [1]
    assert _import_time_fits(ast.parse("if __name__ == '__main__':\n    model.fit(x, y)\n")) == []
    assert _import_time_fits(ast.parse("def f():\n    m.fit(x)\nS = {'a': lambda: f()}\n")) == []


def test_no_test_module_trains_a_model_at_import():
    """Collecting the suite runs no fit: no module-level statement of any test file trains."""
    offenders = []
    for path in sorted(p for pattern in ("test_*.py", "conftest.py") for p in TESTS_ROOT.rglob(pattern)):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for line in _import_time_fits(tree):
            offenders.append(f"{path.relative_to(TESTS_ROOT).as_posix()}:{line}")
    assert not offenders, "test modules that train at import time (move the call into a fixture):\n  " + "\n  ".join(offenders)
