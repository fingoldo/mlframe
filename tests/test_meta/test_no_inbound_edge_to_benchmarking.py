"""Blocking meta-test: ``mlframe.benchmarking`` must stay a pure SINK of the import graph.

The benchmarking harness imports ``feature_selection`` (a ~10-25s cold import) and is
research-facing code with a much looser change cadence than the production packages. A
single inbound edge from production would (a) drag that cold-import cost into every
consumer of the importing package, and (b) let the harness join a strongly-connected
component it can never be refactored out of.

There IS an import-linter ``forbidden`` contract expressing this rule in ``pyproject.toml``,
but it does not enforce anything: ``run-import-linter`` sits in the ``lint-advisory`` CI job
and the local pre-commit hook is ``stages: [manual]`` -- the ``[tool.importlinter]`` section
header says so explicitly. This module is the real gate, and it blocks on every shard.

Deliberately mirrors ``test_no_import_cycles.py::_internal_imports``: only TOP-LEVEL imports
count (``ast.iter_child_nodes``, not ``ast.walk``). A lazy import inside a function body does
not participate in the module-load dependency graph, so it neither costs cold-import time at
package load nor forms a load-time cycle -- the same deliberate choice the cycle detector makes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import mlframe

from tests.test_meta._shared_ast_cache import parsed_ast

# Anchored on THIS checkout, not on ``mlframe.__file__``. Several editable installs share one site-packages here, so
# ``import mlframe`` can resolve to a sibling worktree and a gate keyed off it then silently audits the wrong source.
MLFRAME_DIR = Path(__file__).resolve().parents[2] / "src" / "mlframe"
IMPORTED_MLFRAME_DIR = Path(mlframe.__file__).resolve().parent
PKG_NAME = "mlframe"
BENCHMARKING_PKG = f"{PKG_NAME}.benchmarking"


def _module_name_from_path(path: Path) -> str:
    """``<src>/mlframe/training/core.py`` -> ``mlframe.training.core``.

    Args:
        path: Absolute path to a ``.py`` file under ``MLFRAME_DIR``.

    Returns:
        The dotted module name, with a trailing ``.__init__`` collapsed to the package name.
    """
    rel = path.relative_to(MLFRAME_DIR)
    parts = list(rel.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][: -len(".py")]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([PKG_NAME, *parts])


def _targets_benchmarking(name: str) -> bool:
    """Whether a resolved dotted import target is ``mlframe.benchmarking`` or a submodule of it.

    Args:
        name: A fully-qualified dotted module name.

    Returns:
        True if ``name`` is the benchmarking package itself or lives inside it.
    """
    return name == BENCHMARKING_PKG or name.startswith(BENCHMARKING_PKG + ".")


def _top_level_internal_imports(tree: ast.AST, current: str) -> set[str]:
    """Collect in-package module names imported by ``tree`` at TOP LEVEL only.

    Mirrors ``test_no_import_cycles.py::_internal_imports``: walks ``ast.iter_child_nodes``
    rather than ``ast.walk`` so lazy in-function imports are ignored, but descends into
    top-level ``if`` bodies (``try:``/``if`` guarded optional imports do execute at load).

    Args:
        tree: Parsed module AST.
        current: Dotted name of the module being scanned, used to resolve relative imports.

    Returns:
        Set of fully-qualified ``mlframe.*`` module names imported at module load time.
    """
    out: set[str] = set()
    current_parts = current.split(".")
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PKG_NAME):
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:  # absolute import
                if node.module and node.module.startswith(PKG_NAME):
                    out.add(node.module)
            else:
                # ``from .X import Y`` -- resolve relative to the current module.
                base_parts = current_parts[: -node.level]
                if node.module:
                    base_parts.append(node.module)
                if base_parts and base_parts[0] == PKG_NAME:
                    out.add(".".join(base_parts))
        elif isinstance(node, ast.If):
            # ``if TYPE_CHECKING:`` bodies never run; ``try: import optdep`` bodies do. The cycle
            # detector accepts the same conservative over-count here rather than miss a real edge.
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        if alias.name.startswith(PKG_NAME):
                            out.add(alias.name)
                elif isinstance(sub, ast.ImportFrom):
                    if sub.level == 0 and sub.module and sub.module.startswith(PKG_NAME):
                        out.add(sub.module)
    return out


def _scan_for_inbound_edges() -> list[str]:
    """Scan every production module outside ``benchmarking/`` for a top-level import of it.

    Returns:
        Human-readable ``"<importer> -> <target>"`` strings, one per offending edge.
    """
    benchmarking_dir = MLFRAME_DIR / "benchmarking"
    offenders: list[str] = []
    for py in sorted(MLFRAME_DIR.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        if py.name.endswith(".py.old"):
            continue
        if py.is_relative_to(benchmarking_dir):
            continue  # intra-package imports are fine; only INBOUND edges are forbidden
        tree = parsed_ast(py)
        if tree is None:
            continue
        mod_name = _module_name_from_path(py)
        for target in sorted(_top_level_internal_imports(tree, mod_name)):
            if _targets_benchmarking(target):
                offenders.append(f"{mod_name} -> {target}")
    return offenders


def test_gate_audits_this_checkout_not_a_sibling_worktree() -> None:
    """``import mlframe`` must resolve inside this checkout, or every source-scanning gate is auditing other code.

    Under pytest this holds because ``pythonpath = ["src"]`` puts this checkout first, so it is normally a
    no-op. It is worth pinning anyway: several editable installs share one site-packages on this machine and
    the most recent ``pip install -e`` wins for all of them, so a bare ``python -c "import mlframe"`` here
    resolves to a sibling worktree. Any check run outside pytest -- a smoke import in a script, a hand-run
    validation -- silently reads that other tree. If the ``pythonpath`` setting is ever dropped, gates keyed
    off ``mlframe.__file__`` (``test_no_import_cycles`` among them) would start passing against code nobody
    is editing; this assertion turns that from silent into loud.
    """
    assert MLFRAME_DIR.is_dir(), f"package dir not found: {MLFRAME_DIR}"
    assert IMPORTED_MLFRAME_DIR == MLFRAME_DIR, (
        f"`import mlframe` resolves to {IMPORTED_MLFRAME_DIR}, not this checkout's {MLFRAME_DIR}. "
        f"Source-scanning gates keyed off `mlframe.__file__` are auditing the wrong tree. "
        f"Re-run with PYTHONPATH={MLFRAME_DIR.parent}, or reinstall: pip install -e {MLFRAME_DIR.parents[1]}"
    )


def test_no_production_module_top_level_imports_benchmarking() -> None:
    """No ``src/mlframe/**`` module outside ``benchmarking/`` may top-level-import it.

    Passes vacuously while the package does not exist yet -- that is the point: the gate is in
    place BEFORE the first line of the harness lands, so the very first inbound edge fails CI
    instead of being discovered once it is load-bearing.
    """
    assert MLFRAME_DIR.is_dir(), f"package dir not found: {MLFRAME_DIR}"
    offenders = _scan_for_inbound_edges()
    if offenders:
        pytest.fail(
            f"{len(offenders)} top-level import(s) of {BENCHMARKING_PKG} from production code -- "
            f"the harness must stay a pure sink (it pulls feature_selection, a ~10-25s cold import). "
            "Move the import inside the function that needs it, or invert the dependency:\n  " + "\n  ".join(offenders)
        )
