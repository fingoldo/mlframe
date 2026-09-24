"""Find bounded-eviction sites that discard the evicted entry, and registries whose readers treat a miss as "nothing to do".

A cache may evict freely: a missing entry is recomputed. A registry of pending work may not: each entry is an
obligation (fill this buffer, release this handle), and a reader that finds nothing simply returns, so a dropped
entry silently cancels the work. ``_DEFERRED_HOST_FILL`` was such a registry; its eviction dropped unfilled host
buffers and the CPU kernel then read uninitialised memory.
"""

from __future__ import annotations

import ast

from tests.test_meta._module_mutable_state import aliases_in, base_name, function_defs, module_level_dicts

EVICT_OK_MARKER = "evict-ok:"


def _is_next_iter_of(node: ast.AST, name_ok) -> bool:
    """True for ``next(iter(X))`` where ``X`` satisfies ``name_ok``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "next"
        and len(node.args) >= 1
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "iter"
        and len(node.args[0].args) == 1
        and bool(name_ok(node.args[0].args[0]))
    )


def discarded_evictions(tree: ast.Module, imported: dict[str, str] | None = None) -> list[tuple[int, str, str]]:
    """``[(lineno, func, dict_name), ...]`` for evictions of a module-level dict whose evicted value is thrown away.

    Matches ``X.popitem(...)`` / ``X.pop(next(iter(X)))`` as a bare statement, and ``del X[next(iter(X))]``.
    An eviction whose result is used (``_fill(X.popitem()[1])``) is not discarded and is not reported.
    ``imported`` maps local names bound to ANOTHER module's dict to an ``"origin.module:NAME"`` label, which is what
    the row then reports, so an eviction in one module of a dict defined in another is seen too.
    """
    imported = imported or {}
    names = set(module_level_dicts(tree)) | set(imported)
    out: list[tuple[int, str, str]] = []
    for fn in function_defs(tree):
        alias = aliases_in(fn, names)

        def resolve(node: ast.AST, alias: dict = alias) -> str:
            """Module dict a node refers to, or ""; ``alias`` is bound per function."""
            b = base_name(node) if isinstance(node, (ast.Name, ast.Subscript)) else ""
            b = alias.get(b, b)
            return b if b in names else ""

        for node in ast.walk(fn):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                call, f = node.value, node.value.func
                target = resolve(f.value)
                if not target:
                    continue
                if f.attr == "popitem" or (f.attr == "pop" and call.args and _is_next_iter_of(call.args[0], resolve)):
                    out.append((node.lineno, fn.name, imported.get(target, target)))
            elif isinstance(node, ast.Delete):
                for t in node.targets:
                    if isinstance(t, ast.Subscript) and resolve(t.value) and _is_next_iter_of(t.slice, resolve):
                        out.append((node.lineno, fn.name, imported.get(resolve(t.value), resolve(t.value))))
    # Nested defs are walked from both the outer and inner function; keep one row per site.
    return sorted(set(out))


def _is_bare_none_return(stmt: ast.stmt) -> bool:
    """True for ``return`` / ``return None``."""
    return isinstance(stmt, ast.Return) and (stmt.value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None))


def silent_miss_readers(tree: ast.Module) -> dict[str, list[str]]:
    """``{dict_name: [func, ...]}``: functions that look an entry up and, on a miss, return nothing.

    The shape is ``h = X.get(k)`` (or ``X.pop(k, None)``) followed by ``if h is None: return``, in a function that
    never returns a value. That is a procedure acting on the entry, so a missing entry means the action is skipped.
    A cache reader returns the value (or computes it), which this does not match.
    """
    names = set(module_level_dicts(tree))
    out: dict[str, list[str]] = {}
    for fn in function_defs(tree):
        if any(isinstance(n, ast.Return) and not _is_bare_none_return(n) for n in ast.walk(fn)):
            continue
        alias = aliases_in(fn, names)
        looked_up: dict[str, str] = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                v = node.value
                if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr in ("get", "pop"):
                    b = base_name(v.func.value)
                    b = alias.get(b, b)
                    if b in names:
                        looked_up[node.targets[0].id] = b
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id in looked_up
                and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Is)
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value is None
                and node.body
                and _is_bare_none_return(node.body[0])
            ):
                out.setdefault(looked_up[node.test.left.id], []).append(fn.name)
    return out


def marked_evict_ok(source_lines: list[str], lineno: int) -> bool:
    """True when the eviction line or the line above carries ``# evict-ok: <reason>``."""
    for ln in (lineno, lineno - 1):
        if 1 <= ln <= len(source_lines):
            line = source_lines[ln - 1]
            i = line.find("#")
            if i >= 0 and EVICT_OK_MARKER in line[i:]:
                reason = line[i:].split(EVICT_OK_MARKER, 1)[1].strip()
                if reason:
                    return True
    return False
