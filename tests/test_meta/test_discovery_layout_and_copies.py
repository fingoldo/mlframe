"""Discovery hot loops must not read C-order matrices by column, re-copy the feature matrix, or pay the GIL per resample.

Each rule is a shape that measured slow in the discovery screen: a row-major matrix read column by column inside a loop
(every column a strided gather), the whole feature matrix upcast to float64 again for every base or spec, and a Python loop
that calls an ``@njit`` kernel once per permutation with a fresh RNG draw (the per-resample loop stays GIL-bound; the draws
belong up front and the loop inside one kernel). Existing hits are recorded in ``_discovery_layout_baseline.json`` and may
only shrink.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import mlframe

from tests.test_meta._shared_ast_cache import parsed_ast

_PKG = Path(mlframe.__file__).resolve().parent
_SCOPE = _PKG / "training" / "composite" / "discovery"
_BASELINE = Path(__file__).resolve().parent / "_discovery_layout_baseline.json"
_RNG_DRAWS = frozenset({"permutation", "shuffle", "choice", "integers", "random", "normal", "uniform"})
_MATRIX_HINTS = ("matrix", "_mat", "x_", "X")


def _call_name(node: ast.AST) -> str:
    """The called attribute or name (``np.empty`` -> ``empty``)."""
    f = getattr(node, "func", None)
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "") or ""


def _njit_names(tree: ast.Module) -> set[str]:
    """Functions in the module decorated with something named like njit / jit."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for d in node.decorator_list:
                name = _call_name(d) if isinstance(d, ast.Call) else (d.attr if isinstance(d, ast.Attribute) else getattr(d, "id", ""))
                if "jit" in (name or "").lower():
                    out.add(node.name)
    return out


def _loop_bodies(func: ast.AST):
    """Every ``for``/``while`` loop inside ``func``."""
    return [n for n in ast.walk(func) if isinstance(n, (ast.For, ast.While))]


def _is_matrix_name(node: ast.AST) -> bool:
    """A name that reads like a whole feature matrix."""
    name = node.id if isinstance(node, ast.Name) else (node.attr if isinstance(node, ast.Attribute) else "")
    return bool(name) and (name in ("X", "x") or any(h in name for h in _MATRIX_HINTS if h not in ("X",)))


def findings(tree: ast.Module) -> list[tuple[str, str, int]]:
    """``(function, rule, line)`` for every hit of the three rules in one module."""
    njit = _njit_names(tree)
    out: list[tuple[str, str, int]] = []
    for func in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        c_order = {}
        for node in ast.walk(func):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                v = node.value
                two_d_empty = _call_name(v) in ("empty", "zeros") and v.args and isinstance(v.args[0], ast.Tuple) and len(v.args[0].elts) == 2
                stacked = _call_name(v) == "column_stack"
                f_order = any(k.arg == "order" and isinstance(k.value, ast.Constant) and k.value.value == "F" for k in v.keywords)
                if (two_d_empty or stacked) and not f_order:
                    c_order[node.targets[0].id] = node.lineno
        for loop in _loop_bodies(func):
            body = ast.Module(body=list(loop.body), type_ignores=[])
            for node in ast.walk(body):
                # (a) column READ of a C-order matrix inside a loop (a column write into a C buffer is left alone: the consumer
                # usually gathers rows per fold, which the C layout serves).
                if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name) and node.value.id in c_order
                        and isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2
                        and isinstance(node.slice.elts[0], ast.Slice) and node.slice.elts[0].lower is None and node.slice.elts[0].upper is None):
                    out.append((func.name, "c_order_column_read", node.lineno))
                # (b) whole-matrix float64 re-copy inside a loop.
                if (isinstance(node, ast.Call) and _call_name(node) in ("asarray", "ascontiguousarray", "array") and node.args
                        and _is_matrix_name(node.args[0])
                        and any(k.arg == "dtype" and "float64" in ast.unparse(k.value) for k in node.keywords)):
                    out.append((func.name, "matrix_upcast_in_loop", node.lineno))
            # (c) a loop that draws from an RNG and calls an njit kernel on every iteration.
            calls = [n for n in ast.walk(body) if isinstance(n, ast.Call)]
            if any(_call_name(c) in _RNG_DRAWS for c in calls) and any(_call_name(c) in njit for c in calls):
                out.append((func.name, "njit_per_resample_loop", loop.lineno))
    return sorted(set(out), key=lambda t: t[2])


def _scan() -> Counter:
    """``path::function::rule`` counts over the discovery package."""
    counts: Counter = Counter()
    scanned = 0
    for path in sorted(_SCOPE.rglob("*.py")):
        if "_benchmarks" in path.parts:
            continue
        tree = parsed_ast(path)
        if tree is None:
            continue
        scanned += 1
        rel = path.relative_to(_PKG).as_posix()
        for func, rule, _line in findings(tree):
            counts[f"{rel}::{func}::{rule}"] += 1
    assert scanned >= 50, f"scanned only {scanned} discovery modules; the scope no longer matches the tree"
    return counts


def test_no_new_layout_copy_or_gil_loop_hits():
    """Every hit is recorded; the record may only shrink (a fixed hit must leave it)."""
    got = _scan()
    recorded = Counter(json.loads(_BASELINE.read_text(encoding="utf-8")))
    new = {k: v - recorded.get(k, 0) for k, v in got.items() if v > recorded.get(k, 0)}
    fixed = {k: v - got.get(k, 0) for k, v in recorded.items() if v > got.get(k, 0)}
    assert not new, f"new hot-loop layout / copy / GIL-loop shapes in discovery: {new}"
    assert not fixed, f"these recorded hits are gone; lower them in {_BASELINE.name}: {fixed}"


def test_the_rules_fire_on_their_shapes():
    """Canary: each rule fires on its shape and not on the fixed form."""
    src = '''
import numpy as np
from numba import njit

@njit
def kern(a):
    return a.sum()

def bad(n, m, x_matrix, rng):
    mat = np.empty((n, m))
    for j in range(m):
        s = mat[:, j].sum()
        xf = np.asarray(x_matrix, dtype=np.float64)
        p = rng.permutation(n)
        kern(p)

def good(n, m, x_matrix, rng):
    mat = np.empty((n, m), order="F")
    xf = np.asarray(x_matrix, dtype=np.float64)
    perms = np.stack([rng.permutation(n) for _ in range(m)])
    for j in range(m):
        s = mat[:, j].sum() + xf[0, j]
'''
    rules = sorted({(f, r) for f, r, _ in findings(ast.parse(src))})
    assert rules == [("bad", "c_order_column_read"), ("bad", "matrix_upcast_in_loop"), ("bad", "njit_per_resample_loop")], rules
