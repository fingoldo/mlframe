"""List the call paths on which two Python threads can end up inside one numba ``parallel=True`` kernel.

    python -m mlframe._nested_parallel_scan            # every thread fan-out in the package
    python -m mlframe._nested_parallel_scan --from check_prospective_fe_pairs   # one entry point

numba's default threading layer is not safe to enter concurrently from several Python threads. Linux
mostly tolerates it; macOS aborts the process. mlframe's first three-OS CI run crashed 92 xdist workers
with ``Fatal Python error: Aborted``, every faulthandler dump showing several pool threads stopped at the
same prange call site.

``tests/test_meta/test_no_unguarded_nested_parallel.py`` is the gate that fails a build when a new such
path appears. This module is the investigation tool behind it, and exists separately because the two jobs
are different: a gate answers "is anything unguarded", while fixing a site needs the actual paths, the
kernels at the end of them, and what is left after each guard lands. Guarding the pair sweep took four
rounds, and after each one this printed what remained.

Both share the same rule for what counts as guarded: a path stops being a finding as soon as it passes
through a module importing ``mlframe._numba_parallel_guard``, because guarding at a shared dispatcher
covers every caller and is better than making each caller guard separately.

One deliberate difference: this prints RAW paths, while the gate additionally honours its allowlist of
sites judged unable to race (today: a single-worker watchdog pool whose caller blocks on the future, so
only one thread is ever running). So a path here that the gate does not fail on is expected -- read the
allowlist entry before assuming it is a miss.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

#: Directories that are not production code, so a fan-out inside them cannot reach a user.
SKIP_DIRS = frozenset({"_benchmarks", "legacy", "benchmarks", "profiling", "__pycache__"})

#: How many call hops to follow.
#:
#: Started at 4, which was what the first crash needed (``per_feature_edges`` -> ``edges_fayyad_irani`` ->
#: ``mdlp_bin_edges`` -> ``_mdlp_recurse_validated_bfs`` -> kernel). CI then found a fifth-hop path this
#: missed entirely -- ``check_prospective_fe_pairs`` -> ``_fit_prewarp_and_gate_med`` ->
#: ``_prewarp_generalises`` -> ``build_basis_matrix`` -> ``fit_pair_prewarp_als`` -> kernel -- so a depth
#: tuned to the last known bug is a depth that finds only the last known bug. 6 covers every path seen so
#: far with room above it; raise it rather than trimming a path that reaches a kernel.
MAX_DEPTH = 6


def _is_parallel_kernel(node: ast.AST) -> bool:
    """True for a function decorated ``@njit(..., parallel=True)``."""
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
        isinstance(d, ast.Call) and any(k.arg == "parallel" and isinstance(k.value, ast.Constant) and k.value.value is True for k in d.keywords)
        for d in node.decorator_list
    )


def _guarded_call_nodes(node: ast.AST) -> Set[int]:
    """``id()`` of every Call lexically inside a ``with parallel_kernel_entry():`` block."""
    guarded: Set[int] = set()
    for with_node in ast.walk(node):
        if not isinstance(with_node, (ast.With, ast.AsyncWith)):
            continue
        holds = any(
            isinstance(item.context_expr, ast.Call)
            and (getattr(item.context_expr.func, "id", None) or getattr(item.context_expr.func, "attr", None)) == "parallel_kernel_entry"
            for item in with_node.items
        )
        if holds:
            guarded.update(id(c) for c in ast.walk(with_node) if isinstance(c, ast.Call))
    return guarded


def _called_names(node: ast.AST) -> Set[str]:
    """Every name called inside a function body that is NOT already under the guard.

    Per CALL, not per module. Marking a whole module safe because one of its functions takes the guard is
    what let a real crash through: ``_hermite_prewarp`` guards its fourier replay, and the walk then
    treated ``fit_pair_prewarp_als`` in the same file as covered when it was not.
    """
    guarded = _guarded_call_nodes(node)
    out = {getattr(c.func, "id", None) or getattr(c.func, "attr", None) for c in ast.walk(node) if isinstance(c, ast.Call) and id(c) not in guarded}
    return {n for n in out if n}


def _dispatch_tables(tree: ast.AST) -> Dict[str, Set[str]]:
    """``{dict name: the function names it holds}`` for module-level ``{"a": fn_a, ...}`` literals.

    A kernel reached through a lookup table is invisible to a plain call-graph walk: the call site reads
    ``builder(x)``, and ``builder`` came out of a dict. That is not a corner case here --
    ``_BASIS_BUILDERS`` maps four basis names to four ``parallel=True`` builders, and
    ``build_basis_matrix`` calls whichever one it looked up. The first version of this scan missed it, and
    CI found the crash instead.
    """
    tables: Dict[str, Set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        names = {v.id for v in node.value.values if isinstance(v, ast.Name)}
        if not names:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                tables.setdefault(target.id, set()).update(names)
    return tables


class Package:
    """The parsed call graph of one source tree."""

    def __init__(self, root: Path) -> None:
        """Parse every production module under ``root``."""
        self.kernels: Set[str] = set()
        self.calls: Dict[Tuple[Path, str], Set[str]] = {}
        self.where: Dict[str, List[Path]] = {}
        self.pools: List[Tuple[Path, str]] = []
        for path in sorted(root.rglob("*.py")):
            if SKIP_DIRS & set(path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
                tree = ast.parse(text)
            except (SyntaxError, UnicodeDecodeError):
                continue
            tables = _dispatch_tables(tree)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                self.where.setdefault(node.name, []).append(path)
                edges = _called_names(node)
                # A function that touches a dispatch table can call anything in it, so treat every entry
                # as an edge. Referencing the table is enough -- the lookup and the call are usually two
                # statements apart, and tracking the variable between them buys nothing here.
                # A dispatch-table edge is added from a bare REFERENCE to the table, because the lookup
                # and the call are usually separate statements. That bypasses the per-call guard check,
                # so a function that guards its dispatched call (``with ...: builder(x)``) would still be
                # reported for every member. If the function guards anything at all, take that as guarding
                # the dispatch too -- an over-approximation, but the alternative reports four false paths
                # for every guarded table.
                if not _guarded_call_nodes(node):
                    referenced = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                    for table, members in tables.items():
                        if table in referenced:
                            edges |= members
                self.calls[(path, node.name)] = edges
                if _is_parallel_kernel(node):
                    self.kernels.add(node.name)
                if "ThreadPoolExecutor" in _called_names(node):
                    self.pools.append((path, node.name))

    def reachable_kernels(self, path: Path, fname: str) -> Dict[str, str]:
        """``{kernel name: the call path that reaches it}`` for every UNGUARDED kernel below this function.

        A kernel call wrapped in ``with parallel_kernel_entry():`` is not reported, and neither is anything
        below it -- holding the guard around a call covers everything that call goes on to do.
        """
        found: Dict[str, str] = {}

        def walk(p: Path, f: str, depth: int, seen: Set[Tuple[Path, str]], trail: Tuple[str, ...]) -> None:
            """Depth-first over the call graph, stopping at guarded modules."""
            if (p, f) in seen or depth == 0:
                return
            seen.add((p, f))
            for callee in self.calls.get((p, f), ()):
                if callee in self.kernels:
                    found.setdefault(callee, " -> ".join((*trail[-2:], f, callee)))
                for other in self.where.get(callee, ())[:2]:
                    walk(other, callee, depth - 1, seen, (*trail, f))

        walk(path, fname, MAX_DEPTH, set(), ())
        return found


def main(argv: Optional[List[str]] = None) -> int:
    """Print every thread fan-out that can reach an unguarded prange kernel."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent), help="source tree to scan")
    parser.add_argument("--from", dest="entry", default=None, help="only this function, by name")
    args = parser.parse_args(argv)

    pkg = Package(Path(args.root))
    starts = [(p, f) for p, f in pkg.pools if args.entry is None or f == args.entry]
    if args.entry and not starts:
        # A named entry point that starts no pool is worth reporting: it is usually a typo, and silently
        # printing "0 findings" for it would read as "this one is clean".
        print(f"{args.entry!r} does not start a thread pool anywhere under {args.root}")
        return 2

    total = 0
    for path, fname in starts:
        found = pkg.reachable_kernels(path, fname)
        if not found:
            continue
        total += len(found)
        print(f"\n{path}")
        for kernel, trail in sorted(found.items()):
            print(f"    {kernel:42} via {trail}")
    print(f"\n{len(pkg.kernels)} parallel kernels, {len(pkg.pools)} thread fan-outs, {total} unguarded path(s).")
    return 1 if total else 0


if __name__ == "__main__":  # pragma: no cover -- CLI entry
    raise SystemExit(main())
