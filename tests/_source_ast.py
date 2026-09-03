"""Structural assertions about production code, without `inspect.getsource`.

Some contracts are genuinely structural and have no behavioural equivalent: a constant that must not be
written out a second time, dead scaffolding that must stay deleted, a helper that must be called rather than
reimplemented. The repo's rule bans asserting on source TEXT for these -- a substring check breaks on any
harmless rewrite and passes for an implementation that is actually wrong -- and bans reading the text back
through `read_text().find(...)` too.

The sanctioned form is to parse the module and assert on its STRUCTURE. These helpers read the module's own
file (never `inspect.getsource`, which is what the meta-linter flags) and hand back an AST to query, so the
assertion survives reformatting, comment edits and renamed locals, and fails only when the structure really
changes.

Prefer a behavioural test wherever the contract is observable through the public API; reach for this only when
it is not.
"""

from __future__ import annotations

import ast
import pathlib
from functools import lru_cache
from types import ModuleType


@lru_cache(maxsize=64)
def _parse(path: str) -> ast.Module:
    """Parse the file at ``path`` once per test session."""
    return ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))


def module_ast(mod: ModuleType) -> ast.Module:
    """The parsed AST of ``mod``'s own source file."""
    file = getattr(mod, "__file__", None)
    assert file, f"{mod!r} has no __file__, so its source cannot be parsed"
    return _parse(file)


def function_ast(mod: ModuleType, qualname: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """The definition of ``qualname`` (``"func"`` or ``"Class.method"``) inside ``mod``."""
    tree = module_ast(mod)
    parts = qualname.split(".")
    node: ast.AST = tree
    for i, part in enumerate(parts):
        wanted: tuple = (ast.ClassDef,) if i < len(parts) - 1 else (ast.FunctionDef, ast.AsyncFunctionDef)
        found = next((n for n in ast.walk(node) if isinstance(n, wanted) and getattr(n, "name", None) == part), None)
        assert found is not None, f"{qualname!r} not found in {getattr(mod, '__name__', mod)!r} (looking for {part!r})"
        node = found
    return node  # type: ignore[return-value]


def called_names(node: ast.AST) -> list[str]:
    """Every called name under ``node``, attribute calls reported by their attribute (``a.b()`` -> ``"b"``)."""
    out: list[str] = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        if isinstance(fn, ast.Attribute):
            out.append(fn.attr)
        elif isinstance(fn, ast.Name):
            out.append(fn.id)
    return out


def loaded_names(node: ast.AST) -> set[str]:
    """Every identifier READ under ``node`` -- what the code depends on, ignoring what it binds."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def assigned_names(node: ast.AST) -> set[str]:
    """Every identifier bound by a plain assignment under ``node``."""
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign):
            out.update(t.id for t in sub.targets if isinstance(t, ast.Name))
        elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)) and isinstance(sub.target, ast.Name):
            out.add(sub.target.id)
    return out


def numeric_literals(node: ast.AST) -> list[float]:
    """Every int/float literal under ``node``, excluding booleans."""
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)]


def string_literals(node: ast.AST) -> list[str]:
    """Every string literal under ``node``, docstrings included."""
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
