"""A Python ``for`` loop that issues several small cupy calls per iteration is launch-bound and is on the reviewed baseline or batched.

The old pair-floor code looped ``for p in range(n_pairs)`` and called ``cp.bincount`` / ``cp.sum`` / ``cp.where`` / ``cp.log`` for every pair: tens of thousands of kernel launches, each
doing microseconds of work, so the wall time is the launch latency and not the arithmetic. The fix is one batched kernel over all iterations (a grid dimension instead of the Python
loop) or a vectorised expression over a stacked array.

Detector (AST, no cupy or mlframe import at test time). In modules that mention cupy, a ``for`` loop (not ``while``) whose iterable is not a small literal ``range`` (bound <= 16) or a
literal tuple/list of at most 16 items is examined. The cupy calls counted for the loop are:

1. direct calls through a cupy alias in the loop body (``cp``, ``cupy``, ``xp``, any ``import cupy as X`` alias), nested loops included;
2. bare calls to names imported from cupy (``from cupy import bincount, sum as csum``);
3. one level of callees: a call in the loop body whose target is a function defined in the same module, or imported from an mlframe module (``from mlframe.x import f``, a relative
   ``from .x import f``, a re-export through a package ``__init__``, or ``import mlframe.x as m; m.f()``). Callees are followed up to ``_CALL_DEPTH`` levels (loop, callee, callee of callee); ``self.f`` and ``cls.f`` resolve to a method of the same module.

The loop is flagged when the count is at least 3, or at least 2 when one of them is a data-size-dependent routine (bincount, unique, argsort, sort, where, searchsorted, cumsum).
Loops whose bound is small by construction (``range(nd)``, degrees, streams) are skipped only when the bound is a literal; others are reviewed by hand.

Severity. A chunk or tile loop (a 3-argument ``range(start, stop, step)``, or a bound or iterable whose name ends in chunk/tile/block, optionally plural) is reported with severity
``low`` instead of ``high``. The policy is to report, not exclude: these loops usually run a handful of iterations over big vectorised bodies, but nothing guarantees a small count (a tile
loop can run tens of thousands of times), and an exclusion would silently drop them. ``low`` hits are baselined and fail the test like any other new hit; the tag only tells the reviewer
which worklist entries to take first.

Known limitations (findings A3, A4 and B2 in ``audits/tiny_cupy_detector_2026-10-19``): calls through attribute receivers other than ``self``, ``cls`` or a
module alias are not followed; a loop with a single cupy call per iteration is not flagged even over a huge range; cupy operator launches
(``a[p] * b[p]``) are not counted; a once-only branch inside the loop still counts (a known false positive, reviewed through the baseline).

Benchmark scripts under ``_benchmarks`` are skipped: their loops time launches on purpose.

The baseline is a worklist of launch-bound loops for future optimisation, not a verdict: some loops run a handful of iterations in practice. A new loop outside the baseline fails the
test, and a baseline entry whose loop is gone fails too, so the list only shrinks.
"""

from __future__ import annotations

import ast
import functools
import json
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "mlframe"
BASELINE = Path(__file__).with_name("_tiny_cupy_loops_baseline.json")

_CUPY_ROOTS = {"cp", "cupy", "xp"}
_HEAVY = {"bincount", "unique", "argsort", "sort", "where", "searchsorted", "cumsum"}
_SMALL_RANGE = 16
_CHUNK_SUFFIXES = ("chunk", "chunks", "tile", "tiles", "block", "blocks")
_PKG = "mlframe"
#: How many levels of calls a loop body is followed through (loop -> callee -> callee of callee).
_CALL_DEPTH = 2


def _is_small_literal_loop(loop: ast.For) -> bool:
    """True when the iterable is ``range(<int literal <= 16>)`` or a literal tuple/list of at most 16 items."""
    it = loop.iter
    if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range" and it.args:
        last = it.args[0] if len(it.args) == 1 else it.args[1]
        return isinstance(last, ast.Constant) and isinstance(last.value, int) and last.value <= _SMALL_RANGE
    if isinstance(it, (ast.Tuple, ast.List)):
        return len(it.elts) <= _SMALL_RANGE
    return False


def _is_chunk_loop(loop: ast.For) -> bool:
    """True for a 3-argument ``range`` or a bound/iterable whose name ends in chunk/tile/block (see the module docstring for the severity policy)."""
    it = loop.iter
    if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range" and it.args:
        if len(it.args) == 3:
            return True
        bound = it.args[-1]
    else:
        bound = it
    if isinstance(bound, ast.Name):
        name = bound.id
    elif isinstance(bound, ast.Attribute):
        name = bound.attr
    else:
        name = ""
    return name.lower().endswith(_CHUNK_SUFFIXES)


def _cupy_names(tree: ast.AST) -> "tuple[set[str], set[str]]":
    """``(attribute roots, bare names)``: aliases bound to the cupy module, and names imported from it."""
    roots = set(_CUPY_ROOTS)
    bare: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "cupy" or alias.name.startswith("cupy."):
                    roots.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and (node.module == "cupy" or node.module.startswith("cupy.")):
            bare.update(alias.asname or alias.name for alias in node.names)
    return roots, bare


def _cupy_calls(stmts: list, roots: set, bare: set) -> "list[str]":
    """Names of every cupy call (attribute or bare imported name) inside ``stmts``, nested nodes included."""
    names: list[str] = []
    for stmt in stmts:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in roots:
                names.append(func.attr)
            elif isinstance(func, ast.Name) and func.id in bare:
                names.append(func.id)
    return names


def _import_target(node: ast.ImportFrom, pkg: "tuple[str, ...] | None") -> "tuple[str, ...] | None":
    """Package-relative module parts an ``ImportFrom`` points at, or None when it does not point into mlframe."""
    if node.level == 0:
        mod = node.module or ""
        if mod != _PKG and not mod.startswith(_PKG + "."):
            return None
        return tuple(mod.split("."))[1:]
    if pkg is None or node.level - 1 > len(pkg):
        return None
    base = pkg[: len(pkg) - (node.level - 1)]
    return base + tuple(node.module.split(".")) if node.module else base


def _call_refs(stmts: list) -> "list[tuple]":
    """Outgoing call references in ``stmts``: ``("name", f)`` for ``f(...)`` and ``("attr", base, attr)`` for ``base.attr(...)``; resolved later against a module summary."""
    refs: list[tuple] = []
    for stmt in stmts:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                refs.append(("name", func.id))
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                refs.append(("attr", func.value.id, func.attr))
    return refs


def _summarise(tree: ast.AST, pkg: "tuple[str, ...] | None") -> dict:
    """Per-module facts used for call-following: each unique function's cupy calls and outgoing call references, its import table, and module aliases into mlframe."""
    roots, bare = _cupy_names(tree)
    nodes: dict[str, list] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes.setdefault(node.name, []).append(node)
    funcs = {name: _cupy_calls(defs[0].body, roots, bare) for name, defs in nodes.items() if len(defs) == 1}
    refs = {name: _call_refs(defs[0].body) for name, defs in nodes.items() if len(defs) == 1}
    imports: dict[str, tuple] = {}
    modalias: dict[str, tuple] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target = _import_target(node, pkg)
            if target is not None:
                for alias in node.names:
                    imports[alias.asname or alias.name] = (target, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname and (alias.name == _PKG or alias.name.startswith(_PKG + ".")):
                    modalias[alias.asname] = tuple(alias.name.split("."))[1:]
    return {"funcs": funcs, "refs": refs, "imports": imports, "modalias": modalias}


def _module_file(root: Path, parts: tuple) -> "Path | None":
    """The ``.py`` file or package ``__init__.py`` for module ``parts`` under ``root``."""
    base = root.joinpath(*parts)
    if parts:
        cand = base.with_name(base.name + ".py")
        if cand.is_file():
            return cand
    init = base / "__init__.py"
    return init if init.is_file() else None


@functools.cache
def _file_summary(root: str, parts: tuple) -> "dict | None":
    """Cached summary of one mlframe module; each file is parsed once per scan."""
    path = _module_file(Path(root), parts)
    if path is None:
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return None
    return _summarise(tree, parts if path.name == "__init__.py" else parts[:-1])


def _locate(ref: tuple, summ: dict, root: str, hops: int = 2) -> "tuple[dict, str] | None":
    """The module summary and function name a call reference denotes, or ``None`` when it is not an mlframe function.

    ``self.f`` / ``cls.f`` resolve to a method of the same module; ``mod.f`` through a module alias into mlframe; a bare ``f`` through a local def or an import (following
    a package re-export up to ``hops`` hops).
    """
    if ref[0] == "name":
        name = ref[1]
        if name in summ["funcs"]:
            return summ, name
        if name in summ["imports"] and hops:
            target, orig = summ["imports"][name]
            tsum = _file_summary(root, target)
            if tsum is not None:
                return _locate(("name", orig), tsum, root, hops - 1)
        return None
    _, base, attr = ref
    if base in ("self", "cls") and attr in summ["funcs"]:
        return summ, attr
    if base in summ["modalias"]:
        tsum = _file_summary(root, summ["modalias"][base])
        if tsum is not None and attr in tsum["funcs"]:
            return tsum, attr
    return None


def _follow(refs: "list[tuple]", summ: dict, root: str, depth: int) -> "list[str]":
    """Cupy calls made by the mlframe functions referenced in ``refs``, followed through ``depth`` levels of calls."""
    extra: list[str] = []
    for ref in refs:
        found = _locate(ref, summ, root)
        if found is None:
            continue
        callee_summ, name = found
        extra.extend(callee_summ["funcs"][name])
        if depth > 1:
            extra.extend(_follow(callee_summ["refs"].get(name, []), callee_summ, root, depth - 1))
    return extra


def _callee_calls(stmts: list, summ: dict, root: str, depth: int = _CALL_DEPTH) -> "list[str]":
    """Cupy calls made by the functions invoked inside ``stmts``, followed through up to ``depth`` levels."""
    return _follow(_call_refs(stmts), summ, root, depth)


def tiny_cupy_loops(source: str, rel: "tuple[str, ...] | None" = None, root: Path = SRC_ROOT) -> "list[tuple[str, int, str]]":
    """``[(function name, line, severity)]`` of launch-bound cupy loops in ``source``.

    ``rel`` is the module's path parts relative to ``root`` (needed to resolve relative imports); ``root`` is the mlframe package directory used for absolute imports.
    """
    if not any(tok in source for tok in ("cupy", "cp.", "xp.")):
        return []
    tree = ast.parse(source)
    roots, bare = _cupy_names(tree)
    summ = _summarise(tree, tuple(rel[:-1]) if rel else None)
    root_str = str(root)
    hits: list[tuple[str, int, str]] = []

    def visit(node: ast.AST, func: str) -> None:
        """Walk ``node`` carrying the name of the enclosing function."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name if func == "<module>" else f"{func}.{child.name}")
                continue
            if isinstance(child, ast.For) and not _is_small_literal_loop(child):
                calls = _cupy_calls(child.body, roots, bare) + _callee_calls(child.body, summ, root_str)
                if len(calls) >= 3 or (len(calls) >= 2 and _HEAVY & set(calls)):
                    hits.append((func, child.lineno, "low" if _is_chunk_loop(child) else "high"))
            visit(child, func)

    visit(tree, "<module>")
    return hits


def _current_hits() -> "dict[str, str]":
    """``{"<module path>::<function>": severity}`` over the package source; a function with any high hit is high."""
    found: dict[str, str] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "_benchmarks" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rel_parts = path.relative_to(SRC_ROOT).parts
        try:
            hits = tiny_cupy_loops(text, rel=rel_parts)
        except SyntaxError:
            continue
        rel = str(path.relative_to(SRC_ROOT.parent.parent)).replace("\\", "/")
        for func, _, severity in hits:
            key = f"{rel}::{func}"
            if found.get(key) != "high":
                found[key] = severity
    return found


def _current() -> "set[str]":
    """``{"<module path>::<function>"}`` over the package source."""
    return set(_current_hits())


def _baseline() -> "set[str]":
    """The reviewed ``{"<module path>::<function>"}`` worklist."""
    return set(json.loads(BASELINE.read_text(encoding="utf-8")))


def test_no_new_launch_bound_cupy_loops():
    """A launch-bound cupy loop outside the reviewed baseline fails the test; batch it into one kernel or a vectorised expression."""
    hits = _current_hits()
    new = sorted(key for key in hits if key not in _baseline())
    detail = [f"{key} [{hits[key]}]" for key in new]
    assert not new, f"Python loops of small cupy calls outside the baseline (batch them into one kernel / vectorised op, or review and baseline): {detail}"


def test_baseline_has_no_stale_entries():
    """A baseline entry whose loop is gone or batched is removed, so the baseline only shrinks."""
    stale = sorted(_baseline() - _current())
    assert not stale, f"baseline entries no longer needed (remove them): {stale}"


@pytest.mark.parametrize("rel", sorted({e.split("::")[0] for e in _baseline()}))
def test_baselined_module_exists(rel):
    """Every baseline path points at a real module, so a rename cannot hide a loop."""
    assert (SRC_ROOT.parent.parent / rel).is_file(), rel


def _fake_pkg(tmp_path: Path) -> Path:
    """A throwaway ``mlframe`` tree with a package re-export, so the import resolver is exercised without touching the repo."""
    root = tmp_path / "mlframe"
    (root / "fs").mkdir(parents=True)
    (root / "fs" / "__init__.py").write_text("from .helpers_impl import reexported\n", encoding="utf-8")
    (root / "fs" / "helpers_impl.py").write_text(
        "import cupy as cp\n"
        "def heavy_helper(x):\n"
        "    return cp.log(cp.sum(cp.bincount(x)))\n"
        "def reexported(x):\n"
        "    return cp.where(x, cp.sum(x), cp.unique(x))\n",
        encoding="utf-8",
    )
    return root


def test_checker_flags_per_pair_loop_and_accepts_batched_form():
    """The old per-pair bincount/sum/where/log loop is flagged; a vectorised expression and a tiny literal loop are accepted."""
    bad = (
        "import cupy as cp\n"
        "def pair_floor(codes, n_pairs):\n"
        "    for p in range(n_pairs):\n"
        "        h = cp.bincount(codes[p])\n"
        "        s = cp.sum(h)\n"
        "        out[p] = cp.log(s)\n"
    )
    good = (
        "import cupy as cp\n"
        "def pair_floor(codes):\n"
        "    h = cp.bincount(codes.ravel())\n"
        "    return cp.log(cp.sum(h))\n"
        "def small(a):\n"
        "    for d in range(4):\n"
        "        a = cp.sum(cp.where(a, 1, 0))\n"
    )
    assert tiny_cupy_loops(bad) == [("pair_floor", 3, "high")]
    assert not tiny_cupy_loops(good)


def test_method_call_through_self_is_flagged():
    """A loop that calls ``self._step(x)`` counts the cupy calls of the method ``_step`` defined in the same module (finding A1)."""
    bad = (
        "import cupy as cp\n"
        "class K:\n"
        "    def _step(self, x):\n"
        "        return cp.log(cp.sum(cp.bincount(x)))\n"
        "    def run(self, xs):\n"
        "        for x in xs:\n"
        "            self._step(x)\n"
    )
    assert tiny_cupy_loops(bad) == [("run", 6, "high")]


def test_call_chain_two_levels_deep_is_flagged():
    """Cupy calls two helper levels below the loop (loop -> helper -> helper) count against it (finding A2)."""
    bad = (
        "import cupy as cp\n"
        "def _leaf(x):\n"
        "    return cp.log(cp.sum(cp.bincount(x)))\n"
        "def _mid(x):\n"
        "    return _leaf(x)\n"
        "def driver(xs):\n"
        "    for x in xs:\n"
        "        _mid(x)\n"
    )
    assert tiny_cupy_loops(bad) == [("driver", 7, "high")]


def test_call_chain_beyond_the_depth_limit_is_not_followed():
    """Three helper levels below the loop are past the depth limit, so the chain is not counted (the limit is the documented cost bound)."""
    deep = (
        "import cupy as cp\n"
        "def _leaf(x):\n"
        "    return cp.log(cp.sum(cp.bincount(x)))\n"
        "def _mid2(x):\n"
        "    return _leaf(x)\n"
        "def _mid1(x):\n"
        "    return _mid2(x)\n"
        "def driver(xs):\n"
        "    for x in xs:\n"
        "        _mid1(x)\n"
    )
    assert tiny_cupy_loops(deep) == []


def test_call_hidden_in_same_module_helper_is_flagged():
    """Three cupy calls inside a module-level helper, called once per iteration, count against the loop."""
    bad = (
        "import cupy as cp\n"
        "def _step(x):\n"
        "    return cp.log(cp.sum(cp.bincount(x)))\n"
        "def driver(codes, n):\n"
        "    for p in range(n):\n"
        "        out = _step(codes[p])\n"
    )
    assert tiny_cupy_loops(bad) == [("driver", 5, "high")]


def test_call_hidden_in_mlframe_helper_imported_inside_function_is_flagged(tmp_path):
    """A function-local ``from mlframe.<pkg>.<mod> import f`` resolves to the helper file under the package root, as the blind-spot caller does."""
    root = _fake_pkg(tmp_path)
    bad = (
        "import cupy as cp\n"
        "def driver(codes, n):\n"
        "    from mlframe.fs.helpers_impl import heavy_helper\n"
        "    for p in range(n):\n"
        "        heavy_helper(codes[p])\n"
    )
    assert tiny_cupy_loops(bad, rel=("fs", "caller.py"), root=root) == [("driver", 4, "high")]


def test_relative_and_reexported_helpers_are_flagged(tmp_path):
    """A relative import of a helper, and a helper re-exported by a package ``__init__``, both resolve to the defining file."""
    root = _fake_pkg(tmp_path)
    relative = "import cupy as cp\nfrom .helpers_impl import heavy_helper\ndef driver(xs):\n    for x in xs:\n        heavy_helper(x)\n"
    reexport = "import cupy as cp\nfrom mlframe.fs import reexported\ndef driver(xs):\n    for x in xs:\n        reexported(x)\n"
    assert tiny_cupy_loops(relative, rel=("fs", "caller.py"), root=root) == [("driver", 4, "high")]
    assert tiny_cupy_loops(reexport, rel=("fs", "caller.py"), root=root) == [("driver", 4, "high")]


def test_chunk_step_loop_is_low_severity():
    """A 3-argument ``range`` (chunk loop) is reported with severity low; it is baselined, not excluded."""
    chunked = (
        "import cupy as cp\n"
        "def pooled(x, nperm, perm_chunk):\n"
        "    for p0 in range(0, nperm, perm_chunk):\n"
        "        a = cp.bincount(x)\n"
        "        b = cp.sum(a)\n"
        "        c = cp.log(b)\n"
    )
    assert tiny_cupy_loops(chunked) == [("pooled", 3, "low")]


def test_chunk_named_bound_loop_is_low_severity():
    """A bound whose name ends in chunk/tile/block also gets the low severity."""
    named = "import cupy as cp\ndef f(x, n_chunks):\n    for c in range(n_chunks):\n        a = cp.bincount(x); b = cp.sum(a); cp.log(b)\n"
    assert tiny_cupy_loops(named) == [("f", 3, "low")]


def test_module_alias_and_bare_cupy_import_are_counted():
    """``import cupy as X`` aliases and names imported from cupy count, so renaming the import does not hide a loop."""
    aliased = (
        "import cupy as cupy_mod\n"
        "from cupy import bincount, sum as csum\n"
        "def f(xs):\n"
        "    for x in xs:\n"
        "        s = csum(bincount(x))\n"
        "        cupy_mod.log(s)\n"
    )
    assert tiny_cupy_loops(aliased) == [("f", 4, "high")]
