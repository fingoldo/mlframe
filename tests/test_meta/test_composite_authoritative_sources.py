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
    return [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "is_composite_target_name"]


def slot_writes(tree: ast.Module) -> list[int]:
    """Lines assigning into a target-slot dict by a double subscript, ``target_by_type[tt][name] = ...``."""
    out = []
    for node in ast.walk(tree):
        for target in getattr(node, "targets", []) if isinstance(node, ast.Assign) else []:
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Subscript) and getattr(target.value.value, "id", None) in _SLOT_NAMES:
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


# (c) One construction site per registry adapter. The extended registry once rebuilt six adapter tuples out of the same
# fit/forward/inverse functions the base registry had wrapped, so the two copies could drift (one carried a fix, the other
# did not). ``Transform(...)`` is built only in the registry modules and the chain factory, and a registry ``fit`` is
# wrapped into a ``Transform`` in one module only. Sharing forward/inverse under a different fit is a different transform
# (``linear_residual_multi_robust`` reuses the multi-base apply with a robust fit), so the adapter's identity is its fit.
_TRANSFORM_BUILD_ALLOWED = {
    "training/composite/transforms/registry.py",
    "training/composite/transforms/_registry_extended.py",
    "training/composite/transforms/nonlinear.py",  # the chain factories: they build from their arguments, not a fixed function
}


def transform_builds(tree: ast.Module) -> list[tuple[int, set[str]]]:
    """``(line, {the module-level function passed as fit=})`` for every ``Transform(...)`` call."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "Transform":
            fits = [k.value for k in n.keywords if k.arg == "fit"]
            out.append((n.lineno, {v.id for v in fits if isinstance(v, ast.Name) and v.id.startswith("_")}))
    return out


def _adapter_findings(root: Path) -> tuple[list[str], dict[str, set[str]]]:
    """Builds outside the allowed modules, and each wrapped function's set of building modules."""
    outside, sites = [], {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if "_benchmarks" in path.parts:
            continue
        for line, funcs in transform_builds(parsed_ast(path)):
            if rel not in _TRANSFORM_BUILD_ALLOWED:
                outside.append(f"{rel}:{line}")
            for f in funcs:
                sites.setdefault(f, set()).add(rel)
    return outside, sites


def test_registry_adapters_have_one_construction_site():
    """No ``Transform(...)`` outside the registry modules, and no fit wrapped into a ``Transform`` in two modules."""
    outside, sites = _adapter_findings(_PKG)
    assert not outside, f"Transform(...) built outside the registry modules: {outside}"
    twice = {f: sorted(m) for f, m in sites.items() if len(m) > 1}
    assert not twice, f"registry fits wrapped into a Transform in more than one module: {twice}"
    assert sum(len(transform_builds(parsed_ast(_PKG / p))) for p in _TRANSFORM_BUILD_ALLOWED) >= 40, "the scan lost its subject"


def test_the_adapter_rule_sees_a_rebuilt_adapter(tmp_path):
    """Canary: a second module wrapping a registry function it does not own is reported."""
    pkg = tmp_path / "mlframe" / "training" / "composite" / "transforms"
    pkg.mkdir(parents=True)
    (pkg / "registry.py").write_text("T = Transform(name='a', fit=_a_fit, forward=_a_fwd)\n", encoding="utf-8")
    (pkg / "_registry_extended.py").write_text("T2 = Transform(name='a2', fit=_a_fit, forward=_a_fwd)\n", encoding="utf-8")
    (pkg / "elsewhere.py").write_text("T3 = Transform(name='b', fit=_b_fit)\n", encoding="utf-8")
    outside, sites = _adapter_findings(tmp_path / "mlframe")
    assert outside == ["training/composite/transforms/elsewhere.py:1"]
    assert sites["_a_fit"] == {"training/composite/transforms/registry.py", "training/composite/transforms/_registry_extended.py"}
