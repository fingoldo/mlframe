"""Registry-transform ``fit`` / ``forward`` / ``inverse`` calls go through ``call_transform``, which forwards what the transform takes.

A bare ``transform.fit(y, base)`` drops ``groups`` and ``sample_weight`` silently: a grouped transform refit without its
groups raised mid-discovery, per-fold refits fell back to global parameters, and chain fits ignored the weights.
``call_transform`` passes each optional argument exactly when the transform declares it and fails loudly when a required
one is missing. Existing bare calls are recorded in ``_transform_gateway_baseline.json`` and may only shrink.
"""

from __future__ import annotations

import ast
import orjson
from collections import Counter
from pathlib import Path

import mlframe

from tests.test_meta._shared_ast_cache import parsed_ast

_PKG = Path(mlframe.__file__).resolve().parent
_SCOPE = (_PKG / "training" / "composite", _PKG / "training" / "core")
_BASELINE = Path(__file__).resolve().parent / "_transform_gateway_baseline.json"
_OPS = frozenset({"fit", "forward", "inverse"})
_TRANSFORM_NAMES = frozenset({"transform", "tr", "_t", "_transform", "bivariate", "unary", "tf", "chain_tf", "_tf"})
_SOURCES = frozenset({"get_transform", "TRANSFORMS_REGISTRY", "_TRANSFORMS_REGISTRY"})


def _bound_transforms(func: ast.AST) -> set[str]:
    """Local names bound from ``get_transform(...)`` or a registry subscript inside ``func``."""
    out = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = node.value
            src = v.func if isinstance(v, ast.Call) else (v.value if isinstance(v, ast.Subscript) else None)
            name = getattr(src, "id", getattr(src, "attr", None))
            if name in _SOURCES:
                out.add(node.targets[0].id)
    return out


def bare_calls(tree: ast.Module) -> list[tuple[str, str, int]]:
    """``(function, op, line)`` for every bare ``<transform>.fit/forward/inverse(...)`` call."""
    out = []
    for func in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        names = _TRANSFORM_NAMES | _bound_transforms(func)
        for node in ast.walk(func):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _OPS
                    and isinstance(node.func.value, ast.Name) and node.func.value.id in names):
                out.append((func.name, node.func.attr, node.lineno))
    return sorted(set(out), key=lambda t: t[2])


def _scan() -> Counter:
    """``path::function::op`` counts over composite and core."""
    counts: Counter = Counter()
    scanned = 0
    for root in _SCOPE:
        for path in sorted(root.rglob("*.py")):
            if "_benchmarks" in path.parts or path.name == "_call_gateway.py":
                continue
            tree = parsed_ast(path)
            if tree is None:
                continue
            scanned += 1
            rel = path.relative_to(_PKG).as_posix()
            for func, op, _line in bare_calls(tree):
                counts[f"{rel}::{func}::{op}"] += 1
    assert scanned >= 250, f"scanned only {scanned} modules; the scope no longer matches the tree"
    return counts


def test_no_new_bare_transform_calls():
    """Every bare call is recorded; the record may only shrink (a converted call must leave it)."""
    got = _scan()
    recorded = Counter(orjson.loads(_BASELINE.read_text(encoding="utf-8")))
    new = {k: v - recorded.get(k, 0) for k, v in got.items() if v > recorded.get(k, 0)}
    gone = {k: v - got.get(k, 0) for k, v in recorded.items() if v > got.get(k, 0)}
    assert not new, f"route these transform calls through call_transform(transform, op, ..., groups=..., sample_weight=...): {new}"
    assert not gone, f"these recorded bare calls are gone; lower them in {_BASELINE.name}: {gone}"


def test_the_scan_sees_a_bare_call_and_not_the_gateway():
    """Canary: a bare fit on a ``get_transform`` result is flagged; the gateway call is not."""
    src = '''
def f(name, y, base, groups):
    t = get_transform(name)
    p = t.fit(y, base)
    q = call_transform(t, "fit", y, base, groups=groups)
    return t.inverse(y, base, p)
'''
    assert [(fn, op) for fn, op, _ in bare_calls(ast.parse(src))] == [("f", "fit"), ("f", "inverse")]
