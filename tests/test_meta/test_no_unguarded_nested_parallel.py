"""A thread fan-out must not let two threads into a numba ``parallel=True`` kernel at once.

numba's default threading layer is not safe to enter concurrently from several Python threads. Linux
mostly tolerates it; macOS aborts the process. mlframe's first three-OS CI run crashed 92 xdist workers
with ``Fatal Python error: Aborted``, every faulthandler dump showing multiple pool threads stopped at the
same ``parallel=True`` call site in ``per_feature_edges``.

That instance is fixed, and this gate exists because the NEXT one is easy to add by accident: the pattern
is ordinary and reads as obviously good -- fan work out over columns or chunks, and let each worker call
the fast kernel. Nothing about it looks wrong until it runs on macOS.

The rule: a module that BOTH starts a thread pool AND can REACH a prange kernel -- transitively, not just
by calling one directly -- has to import ``mlframe._numba_parallel_guard``. Transitively matters, and is
the whole reason this file is not two lines shorter: the crash that motivated it ran
``per_feature_edges`` -> ``edges_fayyad_irani`` -> ``mdlp_bin_edges`` -> ``_mdlp_recurse_validated_bfs``
-> the kernel, four hops from the pool, and a same-module check sees none of that.

Whether the guard is held at exactly the right call is not something a static check can decide, so this
asserts that the author was made to think about it, and the allowlist records the modules where the answer
was "this one cannot race" along with why.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "mlframe"
_SKIP_DIRS = frozenset({"_benchmarks", "legacy", "benchmarks", "profiling", "__pycache__"})

#: Modules that start a pool and reach a prange kernel, but cannot have two threads inside one, with the
#: reason each is safe. A new entry needs that reason, not just a path.
_ALLOWLIST = {
    # Watchdog, not parallelism: the pool has one worker and the CALLING thread blocks on
    # ``_future.result(timeout=...)`` until it finishes, so exactly one thread is ever running work.
    "feature_selection/filters/_mrmr_fe_step/_step_pairmi.py",
}


def _is_parallel_kernel(node: ast.AST) -> bool:
    """True for a function decorated ``@njit(..., parallel=True)``."""
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
        isinstance(d, ast.Call) and any(k.arg == "parallel" and isinstance(k.value, ast.Constant) and k.value.value is True for k in d.keywords)
        for d in node.decorator_list
    )


def _modules():
    """Every production module, parsed."""
    for path in sorted(SRC.rglob("*.py")):
        if _SKIP_DIRS & set(path.parts):
            continue
        try:
            yield path, ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue


def _call_graph():
    """``(parallel kernel names, function -> names it calls, function -> defining modules)``."""
    kernels, calls, where = set(), {}, {}
    for path, tree in _modules():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            where.setdefault(node.name, []).append(path)
            if _is_parallel_kernel(node):
                kernels.add(node.name)
            calls[(path, node.name)] = {getattr(c.func, "id", None) or getattr(c.func, "attr", None) for c in ast.walk(node) if isinstance(c, ast.Call)} - {
                None
            }
    return kernels, calls, where


def _guards(path) -> bool:
    """Does this module take the guard? Checked per MODULE on the path, not only at the pool."""
    return "_numba_parallel_guard" in path.read_text(encoding="utf-8")


def _reaches_kernel(path, fname, kernels, calls, where, depth=4, seen=None):
    """The first UNGUARDED prange kernel reachable from this function within ``depth`` hops, or None.

    A path stops being a finding as soon as it passes through a module that takes the guard: guarding at a
    shared dispatcher covers every caller, which is better than making each caller guard separately, and a
    check that only looked at the pool's own module would push authors the other way.
    """
    seen = seen if seen is not None else set()
    if (path, fname) in seen or depth == 0 or _guards(path):
        return None
    seen.add((path, fname))
    for callee in calls.get((path, fname), ()):
        if callee in kernels:
            return callee
        for other in where.get(callee, ())[:2]:  # a name defined in many modules is not a useful edge
            found = _reaches_kernel(other, callee, kernels, calls, where, depth - 1, seen)
            if found:
                return found
    return None


@pytest.fixture(scope="module")
def call_graph():
    """Parsed once: the scan walks every production module."""
    return _call_graph()


def test_a_module_that_threads_into_a_prange_kernel_takes_the_guard(call_graph):
    """Starting a pool whose workers can enter a prange kernel is what aborts the process on macOS."""
    kernels, calls, where = call_graph
    offenders = []
    for path, tree in _modules():
        pool_fns = [
            n.name
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(isinstance(c, ast.Call) and (getattr(c.func, "id", None) or getattr(c.func, "attr", None)) == "ThreadPoolExecutor" for c in ast.walk(n))
        ]
        if not pool_fns:
            continue
        rel = path.relative_to(SRC).as_posix()
        if rel in _ALLOWLIST or "_numba_parallel_guard" in path.read_text(encoding="utf-8"):
            continue
        for fn in pool_fns:
            kernel = _reaches_kernel(path, fn, kernels, calls, where)
            if kernel:
                offenders.append(f"{rel}::{fn} -> {kernel}")
                break

    assert not offenders, (
        f"{offenders} start a thread pool and call a numba parallel=True kernel without importing "
        "mlframe._numba_parallel_guard. Two threads inside one prange region aborts the process on macOS. "
        "Hold parallel_kernel_entry() around the kernel call, or add the module to _ALLOWLIST with the "
        "reason it cannot race."
    )


def test_the_allowlist_still_describes_real_modules():
    """An entry for a module that no longer exists is archaeology, not an exemption."""
    missing = sorted(rel for rel in _ALLOWLIST if not (SRC / rel).exists())
    assert not missing, f"allowlisted modules no longer exist: {missing}"


def test_the_guard_serialises_entry():
    """The guard is only worth importing if it actually excludes a second thread."""
    from mlframe._numba_parallel_guard import parallel_kernel_entry

    lock = parallel_kernel_entry()
    assert lock is parallel_kernel_entry(), "each call handed back a different lock; nothing would be serialised"
    with lock:
        assert not lock.acquire(blocking=False), "a second thread could enter while the first was inside"
