"""AST-level probes for tests that pin a static property of a module (a name must be gone, an import must not exist).

Those properties cannot be observed by calling the code, but they should be read off the syntax tree rather than grepped
out of the source text, so a comment, a docstring or a reformatting cannot make the check pass or fail by accident.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType


def module_tree(module: ModuleType) -> ast.Module:
    """Parse the file *module* was loaded from."""
    return ast.parse(Path(module.__file__).read_bytes())


def identifiers(*modules: ModuleType) -> set[str]:
    """Every identifier bound or referenced in *modules*: names, attributes, arguments, keywords, imports, defs and classes."""
    found: set[str] = set()
    for module in modules:
        for node in ast.walk(module_tree(module)):
            if isinstance(node, ast.Name):
                found.add(node.id)
            elif isinstance(node, ast.Attribute):
                found.add(node.attr)
            elif isinstance(node, ast.arg):
                found.add(node.arg)
            elif isinstance(node, ast.keyword) and node.arg is not None:
                found.add(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    found.add(alias.asname or alias.name)
    return found


def string_literals(*modules: ModuleType) -> list[str]:
    """Every string constant (docstrings and log-message fragments included) in *modules*."""
    return [node.value for module in modules for node in ast.walk(module_tree(module)) if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def import_sources(module: ModuleType) -> list[tuple[int, str]]:
    """Return ``(relative_level, module_name)`` for every ``from X import ...`` statement of *module*."""
    return [(node.level, node.module or "") for node in ast.walk(module_tree(module)) if isinstance(node, ast.ImportFrom)]


def call_keywords(module: ModuleType, func_name: str) -> list[set[str]]:
    """For each call of *func_name* (bare name or attribute) in *module*, the set of keyword-argument names it passes."""
    calls: list[set[str]] = []
    for node in ast.walk(module_tree(module)):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == func_name:
                calls.append({kw.arg for kw in node.keywords if kw.arg is not None})
    return calls
