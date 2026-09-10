"""Blocking meta-test: monolith-split siblings must not import their own facade back.

There is an import-linter ``forbidden`` contract expressing this rule in ``pyproject.toml``
(the ``[tool.importlinter]`` "Kernel-sibling modules split off the 1k-LOC ceiling..." contract),
but it does not enforce anything -- ``run-import-linter`` sits in the ``lint-advisory`` CI job and
the local pre-commit hook is ``stages: [manual]``, exactly the shape
``test_no_inbound_edge_to_benchmarking.py`` already found for the sibling contract next to it. This
module is the real gate for the first one and blocks on every shard, mirroring that file's pattern.

Both siblings were carved out of an over-1k-LOC parent (CLAUDE.md's monolith-split convention)
specifically so the parent could re-export their names -- a back-import would create the real
circular-import risk the split was designed to avoid. Add each new facade/sibling pair to
``_FORBIDDEN_BACKIMPORTS`` as the pattern is applied to more files.

Only TOP-LEVEL imports count, matching ``test_no_inbound_edge_to_benchmarking.py`` and
``test_no_import_cycles.py``: a lazy import inside a function body does not participate in the
module-load dependency graph.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import mlframe

from tests.test_meta._shared_ast_cache import parsed_ast

# Anchored on THIS checkout, not on ``mlframe.__file__`` -- see test_no_inbound_edge_to_benchmarking.py
# for why (several editable installs can share one site-packages).
MLFRAME_DIR = Path(__file__).resolve().parents[2] / "src" / "mlframe"
IMPORTED_MLFRAME_DIR = Path(mlframe.__file__).resolve().parent
PKG_NAME = "mlframe"

#: sibling module -> the facade(s) it must not top-level-import back. Mirrors
#: ``[[tool.importlinter.contracts]]``'s "Kernel-sibling modules..." contract in pyproject.toml.
_FORBIDDEN_BACKIMPORTS: dict[str, tuple[str, ...]] = {
    "mlframe.feature_selection.filters._gpu_resident_select_kernels": ("mlframe.feature_selection.filters._gpu_resident_select",),
    "mlframe.training.composite.provenance_formulas": ("mlframe.training.composite.provenance",),
}


def _module_name_from_path(path: Path) -> str:
    """``<src>/mlframe/a/b.py`` -> ``mlframe.a.b``."""
    rel = path.relative_to(MLFRAME_DIR)
    parts = list(rel.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][: -len(".py")]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([PKG_NAME, *parts])


def _top_level_internal_imports(tree: ast.AST, current: str) -> set[str]:
    """Collect in-package module names imported by ``tree`` at TOP LEVEL only.

    Deliberately duplicated from ``test_no_inbound_edge_to_benchmarking.py`` rather than shared:
    that copy is tied to its own benchmarking-specific scan, and a shared extraction helper
    is a bigger refactor than this two-pair check needs.
    """
    out: set[str] = set()
    current_parts = current.split(".")
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PKG_NAME):
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and node.module.startswith(PKG_NAME):
                    out.add(node.module)
            else:
                base_parts = current_parts[: -node.level]
                if node.module:
                    base_parts.append(node.module)
                if base_parts and base_parts[0] == PKG_NAME:
                    out.add(".".join(base_parts))
        elif isinstance(node, ast.If):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        if alias.name.startswith(PKG_NAME):
                            out.add(alias.name)
                elif isinstance(sub, ast.ImportFrom):
                    if sub.level == 0 and sub.module and sub.module.startswith(PKG_NAME):
                        out.add(sub.module)
    return out


def _module_path(dotted: str) -> Path:
    """``mlframe.a.b`` -> ``<src>/mlframe/a/b.py`` (or ``.../a/b/__init__.py`` if it's a package)."""
    rel_parts = dotted.split(".")[1:]
    as_file = MLFRAME_DIR.joinpath(*rel_parts).with_suffix(".py")
    if as_file.is_file():
        return as_file
    return MLFRAME_DIR.joinpath(*rel_parts, "__init__.py")


def test_gate_audits_this_checkout_not_a_sibling_worktree() -> None:
    """Same guard as ``test_no_inbound_edge_to_benchmarking.py`` -- see that module for the full rationale."""
    assert MLFRAME_DIR.is_dir(), f"package dir not found: {MLFRAME_DIR}"
    assert IMPORTED_MLFRAME_DIR == MLFRAME_DIR, (
        f"`import mlframe` resolves to {IMPORTED_MLFRAME_DIR}, not this checkout's {MLFRAME_DIR}. "
        f"Source-scanning gates keyed off `mlframe.__file__` are auditing the wrong tree."
    )


def test_no_sibling_top_level_imports_its_own_facade() -> None:
    """No monolith-split sibling may top-level-import the facade it was carved out of."""
    offenders: list[str] = []
    for sibling, facades in _FORBIDDEN_BACKIMPORTS.items():
        path = _module_path(sibling)
        assert path.is_file(), f"{sibling} -> expected source at {path}, which does not exist -- update _FORBIDDEN_BACKIMPORTS"
        tree = parsed_ast(path)
        if tree is None:
            continue
        imported = _top_level_internal_imports(tree, sibling)
        for facade in facades:
            if facade in imported:
                offenders.append(f"{sibling} -> {facade}")
    if offenders:
        pytest.fail(
            f"{len(offenders)} monolith-split sibling(s) import their own facade back at top level, "
            "re-creating the circular-import risk the split was meant to avoid:\n  " + "\n  ".join(offenders)
        )
