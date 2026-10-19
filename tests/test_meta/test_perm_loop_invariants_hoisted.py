"""Work that does not depend on the shuffle is computed once outside the permutation loop, or the function is on the reviewed baseline.

Scoring something against K shuffled targets repeats the same statements K times when the shuffle-independent part (the x-marginal, the joint code of a pair, a sorted/binned copy of a
feature) is rebuilt inside ``for k in range(K)``. The fixed form builds those once and reuses them: ``_pair_mi_perm_cuda.py`` builds the pair's joint code and x-marginal once per block
and redoes only the (joint, y_perm) histogram per shuffle, and ``_pair_mi_perm_scalar`` in ``filters/info_theory/_batch_kernels.py`` hoists ``cls_x`` and ``freqs_x_int`` out of the loop.

Detector (AST): a ``for <v> in range(<bound>)`` loop whose bound mentions a permutation count (K, n_perm, n_permutations, nperm, n_perms, npermutations, perms, n_shuffles and
similar). A body statement is reported when it is an assignment whose value contains a numpy / cupy / numba array routine, an array-method call (sum, argsort, bincount, ...) or a
comprehension, and it neither reads the loop variable, nor a name derived from it earlier in the body, nor a name the body (re)assigns elsewhere, and it draws nothing random
(``rng`` / ``random`` / ``shuffle`` / ``permutation`` calls differ per iteration by design). Cheap scalar statements are ignored. Hits are keyed ``<module path>::<function>``.

The detector is deliberately conservative, so a baseline entry is a lead and not a proof: check that the statement really is loop-invariant (the usual false positive is a call that
fills a buffer in place, which the detector cannot see). A new unlisted hit fails the test; a baseline entry whose hit is gone fails too, so the list only shrinks.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "mlframe"
BASELINE = Path(__file__).with_name("_perm_loop_invariants_baseline.json")

_PERM_BOUND_RE = re.compile(r"^(K|n_?perms?|n_?permutations?|n_?shuffles?|num_?perm\w*|perms|permutations)$", re.I)
_ARRAY_ROOTS = {"np", "numpy", "cp", "cupy", "xp", "nb", "numba", "jnp", "torch"}
_ARRAY_METHODS = {
    "sum", "argsort", "bincount", "cumsum", "sort", "unique", "astype", "reshape", "take", "copy", "ravel", "flatten",
    "searchsorted", "nonzero", "mean", "var", "std", "dot", "matmul", "max", "min", "argmax", "argmin", "clip",
}  # fmt: skip
_CHEAP_ATTRS = {"dtype", "uint64", "uint32", "int64", "int32", "int8", "uint8", "float32", "float64", "bool_", "intp", "errstate", "finfo", "iinfo"}
_ALLOCATORS = {"copy", "empty", "zeros", "ones", "full", "empty_like", "zeros_like", "ones_like", "full_like"}
_RANDOM_RE = re.compile(r"rng|random|shuffle|permutation|seed|sample|choice|randint|default_rng", re.I)


def _names(node: ast.AST, ctx: type) -> "set[str]":
    """Names under ``node`` with the given ``ast.Load`` / ``ast.Store`` context."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ctx)}


def _stored(node: ast.AST) -> "set[str]":
    """Names (re)bound under ``node``, including the base array of ``a[i] = ...`` / ``a.x = ...`` and ``a += ...`` writes."""
    out = _names(node, ast.Store)
    for n in ast.walk(node):
        if isinstance(n, (ast.Subscript, ast.Attribute)) and isinstance(n.ctx, ast.Store):
            base = n.value
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if isinstance(base, ast.Name):
                out.add(base.id)
    return out


def _is_heavy(value: ast.AST) -> bool:
    """True when the expression calls an array routine / array method or builds a comprehension.

    A bare allocation (``.copy()``, ``np.empty`` / ``zeros`` / ``full``) or a scalar / dtype constructor is not heavy: a per-iteration buffer that the body then fills or shuffles in place
    is not shared work, and the constructors cost nothing.
    """
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr in _ALLOCATORS:
        return False
    for n in ast.walk(value):
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            root = n.func
            while isinstance(root, ast.Attribute):
                root = root.value
            if n.func.attr in _CHEAP_ATTRS:
                continue
            if (isinstance(root, ast.Name) and root.id in _ARRAY_ROOTS) or n.func.attr in _ARRAY_METHODS:
                return True
    return False


def _bound_is_perm(loop: ast.For) -> bool:
    """True for ``for v in range(...)`` whose range arguments name a permutation count."""
    it = loop.iter
    if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"):
        return False
    return any(_PERM_BOUND_RE.match(name) for arg in it.args for name in _names(arg, ast.Load) | {a.attr for a in ast.walk(arg) if isinstance(a, ast.Attribute)})


def _invariant_statements(loop: ast.For) -> "list[int]":
    """Line numbers of loop-body assignments that are heavy and independent of the shuffle."""
    body = loop.body
    stores_by_stmt = [_stored(s) for s in body]
    tainted = _names(loop.target, ast.Store)
    mutated_in_place = {
        n.value.id for s in body for n in ast.walk(s) if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name)
    } | {n.target.id for s in body for n in ast.walk(s) if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)}
    hits: list[int] = []
    for i, stmt in enumerate(body):
        if not isinstance(stmt, ast.Assign):
            tainted |= stores_by_stmt[i]
            continue
        loads = _names(stmt.value, ast.Load)
        others_store = set().union(*(s for j, s in enumerate(stores_by_stmt) if j != i)) if len(body) > 1 else set()
        subscript_stores = any(isinstance(t, (ast.Subscript, ast.Attribute)) for t in stmt.targets)
        invariant = (
            not subscript_stores
            and _is_heavy(stmt.value)
            and not (loads & tainted)
            and not (loads & others_store)
            and not (stores_by_stmt[i] & loads)
            and not (stores_by_stmt[i] & mutated_in_place)
            and not _RANDOM_RE.search(ast.unparse(stmt.value))
        )
        if invariant:
            hits.append(stmt.lineno)
        else:
            tainted |= stores_by_stmt[i]
    return hits


def perm_loop_invariants(source: str) -> "list[tuple[str, int]]":
    """``[(function name, line)]`` of shuffle-independent heavy statements inside permutation loops."""
    if "range(" not in source:
        return []
    hits: list[tuple[str, int]] = []

    def visit(node: ast.AST, func: str) -> None:
        """Walk ``node`` carrying the name of the enclosing function."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name if func == "<module>" else f"{func}.{child.name}")
                continue
            if isinstance(child, ast.For) and func != "<module>" and _bound_is_perm(child):
                hits.extend((func, line) for line in _invariant_statements(child))
            visit(child, func)

    visit(ast.parse(source), "<module>")
    return hits


def _current() -> "set[str]":
    """``{"<module path>::<function>"}`` over the package source."""
    found: set[str] = set()
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "_benchmarks" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            hits = perm_loop_invariants(text)
        except SyntaxError:
            continue
        rel = str(path.relative_to(SRC_ROOT.parent.parent)).replace("\\", "/")
        found.update(f"{rel}::{func}" for func, _ in hits)
    return found


def _baseline() -> "set[str]":
    """The reviewed ``{"<module path>::<function>"}`` list."""
    return set(json.loads(BASELINE.read_text(encoding="utf-8")))


def test_no_new_unhoisted_permutation_invariants():
    """A shuffle-independent heavy statement inside a permutation loop, outside the baseline, fails the test; hoist it above the loop."""
    new = sorted(_current() - _baseline())
    assert not new, f"loop-invariant work inside a permutation loop (hoist it out, or review and baseline): {new}"


def test_baseline_has_no_stale_entries():
    """A baseline entry whose statement was hoisted or removed is deleted, so the baseline only shrinks."""
    stale = sorted(_baseline() - _current())
    assert not stale, f"baseline entries no longer needed (remove them): {stale}"


@pytest.mark.parametrize("rel", sorted({e.split("::")[0] for e in _baseline()}))
def test_baselined_module_exists(rel):
    """Every baseline path points at a real module, so a rename cannot hide a hit."""
    assert (SRC_ROOT.parent.parent / rel).is_file(), rel


def test_checker_flags_unhoisted_marginal_and_accepts_hoisted_form():
    """Rebuilding the x-marginal in every shuffle is flagged; hoisting it, or only touching shuffled data, is accepted."""
    bad = (
        "import numpy as np\n"
        "def f(x, y, K, rng):\n"
        "    for k in range(K):\n"
        "        yp = rng.permutation(y)\n"
        "        fx = np.bincount(x)\n"
        "        out[k] = score(fx, yp)\n"
    )
    good = (
        "import numpy as np\n"
        "def f(x, y, K, rng):\n"
        "    fx = np.bincount(x)\n"
        "    for k in range(K):\n"
        "        yp = rng.permutation(y)\n"
        "        h = np.bincount(yp)\n"
        "        out[k] = score(fx, h)\n"
    )
    assert perm_loop_invariants(bad) == [("f", 5)]
    assert not perm_loop_invariants(good)
