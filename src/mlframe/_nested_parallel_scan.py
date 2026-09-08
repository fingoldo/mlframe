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


def _called_names(node: ast.AST) -> Set[str]:
    """Every name called inside a function body, attribute calls included."""
    out = {getattr(c.func, "id", None) or getattr(c.func, "attr", None) for c in ast.walk(node) if isinstance(c, ast.Call)}
    return {n for n in out if n}


class Package:
    """The parsed call graph of one source tree."""

    def __init__(self, root: Path) -> None:
        """Parse every production module under ``root``."""
        self.kernels: Set[str] = set()
        self.calls: Dict[Tuple[Path, str], Set[str]] = {}
        self.where: Dict[str, List[Path]] = {}
        self.guarded: Set[Path] = set()
        self.pools: List[Tuple[Path, str]] = []
        for path in sorted(root.rglob("*.py")):
            if SKIP_DIRS & set(path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
                tree = ast.parse(text)
            except (SyntaxError, UnicodeDecodeError):
                continue
            if "_numba_parallel_guard" in text:
                self.guarded.add(path)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                self.where.setdefault(node.name, []).append(path)
                self.calls[(path, node.name)] = _called_names(node)
                if _is_parallel_kernel(node):
                    self.kernels.add(node.name)
                if "ThreadPoolExecutor" in _called_names(node):
                    self.pools.append((path, node.name))

    def reachable_kernels(self, path: Path, fname: str) -> Dict[str, str]:
        """``{kernel name: the call path that reaches it}`` for everything unguarded below this function."""
        found: Dict[str, str] = {}

        def walk(p: Path, f: str, depth: int, seen: Set[Tuple[Path, str]], trail: Tuple[str, ...]) -> None:
            """Depth-first over the call graph, stopping at guarded modules."""
            if (p, f) in seen or depth == 0 or p in self.guarded:
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
