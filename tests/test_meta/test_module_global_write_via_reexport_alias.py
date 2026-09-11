"""Blocking meta-test: a private ALL_CAPS module global must be written through the module that
actually DEFINES it, not one that merely re-exports the name.

Ports ``.semgrep.yml``'s ``module-global-write-via-reexport-alias`` rule so it is enforced on every
shard instead of only in the manual, advisory ``semgrep-warn`` pre-commit hook. Real bug this catches
(2026-07-18, ``_gpu_resident_radix_ktc.py``): ``from . import _gpu_resident_select as _sel; _sel.X =
value`` silently rebinds the FACADE's own copy of ``X`` when the facade only re-exports ``X`` from a
sibling module (``from .sibling import X``) -- the sibling's actual global, which other code reads,
is never touched. A write through the OWNING module (``from . import sibling as _s; _s.X = value``)
is the only form that reaches the real global.

Scope matches the semgrep rule exactly: ``$ALIAS.$CONST = $VALUE`` where ``$CONST`` is a private
ALL_CAPS name (``_[A-Z][A-Z0-9_]*``) and ``$ALIAS`` is a private lowercase name (``_[a-z]...``),
resolved only for the ``from . import <module> as <alias>`` package-relative import shape this
codebase actually uses for this pattern -- a symbol-level alias (``from .x import Y as alias``)
targets a NAME, not a module, and is out of scope for a "which module defines this global" question.

Suppressible with a trailing ``# nosemgrep: module-global-write-via-reexport-alias`` comment, mirroring
the two suppressed lines in ``_gpu_resident_radix_ktc.py`` that are legitimate (already writing through
the owning module) -- the same escape hatch semgrep itself honours, kept working across the port.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import mlframe

from tests.test_meta._scan_guard import assert_scanned_enough
from tests.test_meta._shared_ast_cache import parsed_ast, source_text

MLFRAME_DIR = Path(__file__).resolve().parents[2] / "src" / "mlframe"
IMPORTED_MLFRAME_DIR = Path(mlframe.__file__).resolve().parent

_CONST_RE = re.compile(r"^_[A-Z][A-Z0-9_]*$")
_ALIAS_RE = re.compile(r"^_[a-z]")
_SUPPRESS_MARKER = "nosemgrep: module-global-write-via-reexport-alias"


def _resolve_relative_module_import(tree: ast.Module, alias: str, current_file: Path) -> Path | None:
    """Find ``from . import <module> as <alias>`` (or without ``as``, name == alias) at top level,
    and return the imported module's own file path, or None if no such import exists.

    Only single-dot ``from . import X`` is resolved -- the one shape this rule's real fix used.
    A deeper relative import (``from .. import X``) or an absolute import would need a different
    base-directory resolution and has not been observed for this pattern.
    """
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None:
            for a in node.names:
                bound = a.asname or a.name
                if bound == alias:
                    return current_file.parent / f"{a.name}.py"
    return None


def _module_defines_name(module_path: Path, name: str) -> bool | None:
    """True if ``module_path`` assigns ``name`` at module (top) level; False if it only imports/
    re-exports it; None if the module could not be read/parsed at all (caller should not flag it).
    """
    if not module_path.is_file():
        return None
    tree = parsed_ast(module_path)
    if tree is None:
        return None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return True
    return False


def _line_is_suppressed(path: Path, lineno: int) -> bool:
    """Whether the flagged line carries the ``# nosemgrep: module-global-write-via-reexport-alias`` marker."""
    text = source_text(path)
    if text is None:
        return False
    lines = text.splitlines()
    if not (1 <= lineno <= len(lines)):
        return False
    return _SUPPRESS_MARKER in lines[lineno - 1]


def _scan_for_reexport_alias_writes() -> list[str]:
    """Every ``$alias.$CONST = ...`` write where ``$alias`` is a `from . import X as alias` module
    import and the imported module does NOT itself define ``$CONST`` (it only re-exports it)."""
    offenders: list[str] = []
    scanned = 0
    for py in sorted(MLFRAME_DIR.rglob("*.py")):
        if "__pycache__" in py.parts or py.name.endswith(".py.old"):
            continue
        tree = parsed_ast(py)
        if tree is None:
            continue
        scanned += 1
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)):
                continue
            const, alias = target.attr, target.value.id
            if not (_CONST_RE.match(const) and _ALIAS_RE.match(alias)):
                continue
            if _line_is_suppressed(py, node.lineno):
                continue
            owning_module = _resolve_relative_module_import(tree, alias, py)
            if owning_module is None:
                continue  # alias isn't a `from . import X as alias` module import -- out of scope
            defines = _module_defines_name(owning_module, const)
            if defines is False:
                offenders.append(
                    f"{py.relative_to(MLFRAME_DIR)}:{node.lineno}  {alias}.{const} = ...  "
                    f"(imported module {owning_module.relative_to(MLFRAME_DIR)} does not define {const} itself)"
                )
    assert_scanned_enough(scanned, "src/mlframe")
    return offenders


def test_gate_audits_this_checkout_not_a_sibling_worktree() -> None:
    """Same guard as ``test_no_inbound_edge_to_benchmarking.py`` -- see that module for the full rationale."""
    assert MLFRAME_DIR.is_dir(), f"package dir not found: {MLFRAME_DIR}"
    assert IMPORTED_MLFRAME_DIR == MLFRAME_DIR, (
        f"`import mlframe` resolves to {IMPORTED_MLFRAME_DIR}, not this checkout's {MLFRAME_DIR}. "
        f"Source-scanning gates keyed off `mlframe.__file__` are auditing the wrong tree."
    )


def test_no_module_global_write_via_reexport_alias() -> None:
    """A write to a private ALL_CAPS module global must target the module that defines it."""
    offenders = _scan_for_reexport_alias_writes()
    if offenders:
        pytest.fail(
            f"{len(offenders)} write(s) to a module global through an alias that does not define it -- "
            "the write silently rebinds the RE-EXPORTING module's own copy, never reaching the real "
            "global other code reads. Import the OWNING module directly, or suppress with a trailing "
            f"'# {_SUPPRESS_MARKER}' comment if this alias genuinely IS the owning module:\n  " + "\n  ".join(offenders)
        )
