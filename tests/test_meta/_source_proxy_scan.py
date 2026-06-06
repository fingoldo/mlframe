"""AST scanner: find behavioral-proxy assertions of the form `"literal" in <var>`
where <var> is bound (in the SAME function scope, or module scope) to a `*.read_text()`
result -- asserting prod CODE CONTAINS a string instead of asserting RUNTIME BEHAVIOUR.

Scoping matters: a variable named ``text`` bound from ``_format_for_log(...)`` in one
function must NOT be conflated with a ``text`` bound from ``path.read_text()`` in another.
The scanner therefore resolves read_text bindings per enclosing function scope (with
module scope as the fallback), not across the whole module.

Legitimate read_text uses (LOC budgets, CHANGELOG cross-walk, meta-linters, docstring/
annotation scanners) are NOT flagged: only a string-constant membership test against a
read_text-derived name matches.
"""
from __future__ import annotations

import ast
from pathlib import Path


def _contains_read_text(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "read_text":
            return True
    return False


def _root_name(node: ast.expr) -> str | None:
    cur = node
    while isinstance(cur, (ast.Attribute, ast.Subscript)):
        cur = cur.value
    if isinstance(cur, ast.Name):
        return cur.id
    return None


def _iter_scope_nodes(scope: ast.AST):
    """Yield statement nodes lexically inside ``scope`` WITHOUT descending into nested
    function/class definitions (those are their own scope)."""
    stack = list(getattr(scope, "body", []))
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            stack.append(child)


def _scope_read_text_names(scope: ast.AST) -> set[str]:
    """Names assigned from a read_text() expression directly inside ``scope`` (not nested
    function/class bodies). Nested scopes get their own analysis."""
    names: set[str] = set()
    for node in _iter_scope_nodes(scope):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            value = node.value
            if value is None or not _contains_read_text(value):
                continue
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                targets = [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def find_source_proxy_sites(path: Path) -> list[tuple[int, str]]:
    """Return [(lineno, literal)] for each `"literal" in <read_text-var>` membership test,
    where the var is read_text-bound in the same function scope (or module scope)."""
    try:
        src = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []

    module_names = _scope_read_text_names(tree)
    hits: list[tuple[int, str]] = []
    seen: set[int] = set()

    # Walk every function scope independently so a same-named var in another scope
    # never leaks its read_text binding here. A Compare is attributed to the nearest
    # enclosing function whose scope (or module) binds the RHS root name.
    func_scopes: list[ast.AST] = [tree]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_scopes.append(node)

    for scope in func_scopes:
        names = module_names | _scope_read_text_names(scope)
        if not names:
            continue
        for node in _iter_scope_nodes(scope):
            if not isinstance(node, ast.Compare) or id(node) in seen:
                continue
            if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.In, ast.NotIn)):
                continue
            left = node.left
            if not (isinstance(left, ast.Constant) and isinstance(left.value, (str, bytes))):
                continue
            root = _root_name(node.comparators[0])
            if root is not None and root in names:
                seen.add(id(node))
                lit = left.value if isinstance(left.value, str) else left.value.decode("latin-1", "replace")
                hits.append((node.lineno, lit))

    hits.sort()
    return hits
