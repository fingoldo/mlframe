"""Nothing in the tree may need a Python newer than the floor `pyproject.toml` advertises.

Two CI failures on the 3.9 shard turned out to be different instances of one class: a walrus operator in the
subscript of an assignment TARGET (a 3.9 syntax error, which made a whole module unimportable on a supported
runtime) and a `__slots__` read that assumed `@dataclass(slots=True)` had applied, which on 3.9 silently
yielded an EMPTY set and turned half the assertions around it into vacuous passes.

Both only surfaced because a shard happened to run that interpreter, and both are cheap to catch statically.
This pins the whole class instead of waiting for the next one.

Deliberately AST-based rather than grep: `match` and `tomllib` appear in comments, docstrings and string
literals all over a codebase that talks about them, and a text search cannot tell those from real uses.

A guarded use is not a finding. `import tomllib` inside a `try` with an `ImportError` handler is the
documented way to reach a 3.11 stdlib module from 3.9, and the tree already does that in four places.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCANNED_ROOTS = ("src/mlframe", "tests", "benchmarks")

# (module name, first version that ships it in the stdlib). Reaching these from the floor needs a fallback.
NEWER_STDLIB = {"tomllib": "3.11", "graphlib": "3.9"}

# Call keyword arguments that do not exist at the floor.
NEWER_CALL_KWARGS = {"zip": {"strict"}, "dataclass": {"slots", "kw_only"}}


def _python_floor() -> tuple[int, int]:
    """The (major, minor) floor declared by `requires-python` in pyproject.toml."""
    import re

    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'requires-python\s*=\s*"[^0-9]*(\d+)\.(\d+)', text)
    assert match, "requires-python is missing from pyproject.toml, so there is no floor to check against"
    return int(match.group(1)), int(match.group(2))


def _iter_modules():
    """Every Python file under the scanned roots, as (repo-relative path, parsed tree)."""
    for root in SCANNED_ROOTS:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                # Reported by its own test below, not swallowed here.
                continue
            yield path.relative_to(REPO_ROOT).as_posix(), tree


def _guarded_import_lines(tree: ast.Module) -> set[int]:
    """Line numbers of imports the module reaches only behind an explicit guard.

    Two guard forms count, because both are correct and the tree uses both: a `try` whose handler catches
    ImportError, and an `if sys.version_info >= (...)` branch. The version check is the better of the two --
    it says WHICH version the fallback is for instead of inferring it from a failed import -- so a check that
    recognised only the try/except form would have pushed authors toward the weaker one.
    """
    guarded: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.If) and any(isinstance(n, ast.Attribute) and n.attr == "version_info" for n in ast.walk(node.test)):
            for branch in (node.body, node.orelse):
                for stmt in branch:
                    for inner in ast.walk(stmt):
                        if isinstance(inner, (ast.Import, ast.ImportFrom)):
                            guarded.add(inner.lineno)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        handles_import_error = any(handler.type is None or any(isinstance(n, ast.Name) and n.id in {"ImportError", "ModuleNotFoundError"} for n in ast.walk(handler.type)) for handler in node.handlers)
        if not handles_import_error:
            continue
        for stmt in node.body:
            for inner in ast.walk(stmt):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    guarded.add(inner.lineno)
    return guarded


def test_every_module_in_the_tree_parses():
    """A module that will not parse is not merely untested -- it is unimportable wherever it lands.

    The slugify meta test AST-walks the tree and fails on anything it cannot read, so an unparseable module
    takes an unrelated test down with it rather than reporting itself.
    """
    broken = []
    for root in SCANNED_ROOTS:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                broken.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{exc.lineno}: {exc.msg}")
    assert not broken, "modules that do not parse:\n  " + "\n  ".join(broken)


def test_no_walrus_in_an_assignment_target_subscript():
    """`cols[name := x] = v` is a syntax error on 3.9, so the module cannot even be imported there.

    It parses on 3.10+, which is exactly why it reached master: every local check and most shards accept it.
    """
    offenders = []
    for rel, tree in _iter_modules():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for inner in ast.walk(target):
                    if isinstance(inner, ast.NamedExpr):
                        offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, "walrus inside an assignment-target subscript (3.9 SyntaxError):\n  " + "\n  ".join(offenders)


def test_no_match_statement_below_the_floor():
    """`match` is 3.10 syntax; below that floor it is a parse error for the whole module."""
    if _python_floor() >= (3, 10):
        pytest.skip("the declared floor is 3.10 or newer, so match statements are fine")
    offenders = [f"{rel}:{node.lineno}" for rel, tree in _iter_modules() for node in ast.walk(tree) if node.__class__.__name__ == "Match"]
    assert not offenders, "match statements below the declared Python floor:\n  " + "\n  ".join(offenders)


def test_no_unguarded_import_of_a_newer_stdlib_module():
    """Reaching a stdlib module the floor does not ship needs an ImportError fallback, not a bare import."""
    floor = _python_floor()
    offenders = []
    for rel, tree in _iter_modules():
        guarded = _guarded_import_lines(tree)
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                root_name = (name or "").split(".")[0]
                needed = NEWER_STDLIB.get(root_name)
                if needed and tuple(int(p) for p in needed.split(".")) > floor and node.lineno not in guarded:
                    offenders.append(f"{rel}:{node.lineno}: {root_name} needs {needed}")
    assert not offenders, "unguarded imports of a stdlib module newer than the floor:\n  " + "\n  ".join(offenders)


def test_no_call_keyword_newer_than_the_floor():
    """`zip(strict=)` and `@dataclass(slots=/kw_only=)` are 3.10+; below that they raise TypeError at runtime.

    `_training_context.py` reaches `slots=True` correctly, by building the keyword dict behind a version
    check, which is invisible to this scan precisely because it is not a literal keyword in the call.
    """
    if _python_floor() >= (3, 10):
        pytest.skip("the declared floor is 3.10 or newer")
    offenders = []
    for rel, tree in _iter_modules():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fname = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            for kw in node.keywords:
                if kw.arg and kw.arg in NEWER_CALL_KWARGS.get(fname, set()):
                    offenders.append(f"{rel}:{node.lineno}: {fname}({kw.arg}=...)")
    assert not offenders, "call keywords newer than the declared Python floor:\n  " + "\n  ".join(offenders)
