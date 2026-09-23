"""Composite status comes from the spec-name set, and a target slot is filled through one helper.

Two defects shared a shape: code decided something a registry or spec set already knew. ``is_composite_target_name`` guesses
from the name, so it missed runtime-registered chain targets and matched dashed raw ones; a bare ``target_by_type[tt][name] =``
overwrote whatever sat in that slot, so a spec named like an existing target replaced that target's values in silence.
Both are structural, so both are checked here by AST rather than by a test per consumer.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mlframe

from tests.test_meta._shared_ast_cache import parsed_ast

_PKG = Path(mlframe.__file__).resolve().parent
_TRAINING = _PKG / "training"
# naming.py owns the heuristic; the legacy-pickle loader has no spec set to consult.
_HEURISTIC_ALLOWED = {"training/composite/transforms/naming.py"}
_SLOT_WRITE_ALLOWED = {"training/core/_target_slots.py"}
_SLOT_NAMES = frozenset({"target_by_type", "new_target_by_type", "_target_by_type"})


def heuristic_calls(tree: ast.Module) -> list[int]:
    """Lines calling ``is_composite_target_name(...)``."""
    return [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "is_composite_target_name"]


def slot_writes(tree: ast.Module) -> list[int]:
    """Lines assigning into a target-slot dict by a double subscript, ``target_by_type[tt][name] = ...``."""
    out = []
    for node in ast.walk(tree):
        for target in getattr(node, "targets", []) if isinstance(node, ast.Assign) else []:
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Subscript)
                    and getattr(target.value.value, "id", None) in _SLOT_NAMES):
                out.append(node.lineno)
    return out


def _scan(finder) -> dict[str, list[int]]:
    """``{relative path: lines}`` over the training package, skipping benchmarks."""
    hits = {}
    for path in sorted(_TRAINING.rglob("*.py")):
        if "_benchmarks" in path.parts:
            continue
        tree = parsed_ast(path)
        if tree is None:
            continue
        found = finder(tree)
        if found:
            hits[path.relative_to(_PKG).as_posix()] = found
    return hits


def test_composite_status_is_read_from_the_spec_names():
    """Only ``naming.py`` may call the name heuristic; every consumer passes the suite's spec-name set instead (INT-08)."""
    offenders = {p: lines for p, lines in _scan(heuristic_calls).items() if p not in _HEURISTIC_ALLOWED}
    assert not offenders, f"call is_composite_target(name, composite_names) with the suite's spec names instead: {offenders}"


def test_a_target_slot_is_filled_through_the_insert_helper():
    """Only ``_target_slots.py`` writes a target slot directly; it drops a colliding name instead of overwriting (INT-18)."""
    offenders = {p: lines for p, lines in _scan(slot_writes).items() if p not in _SLOT_WRITE_ALLOWED}
    assert not offenders, f"insert through insert_composite_targets(), which refuses to overwrite an existing target: {offenders}"


def test_the_scans_see_what_they_claim_to_see():
    """Canary: both finders fire on the shapes they forbid and stay quiet on the sanctioned ones."""
    src = (
        "def f(name, target_by_type, tt, values, composite_names):\n"
        "    if is_composite_target_name(name):\n"
        "        target_by_type[tt][name] = values\n"
        "    ok = is_composite_target(name, composite_names)\n"
        "    target_by_type[tt] = dict(values)\n"
        "    return ok\n"
    )
    tree = ast.parse(src)
    assert heuristic_calls(tree) == [2]
    assert slot_writes(tree) == [3]
