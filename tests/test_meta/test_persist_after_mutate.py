"""Nothing in the suite orchestrators mutates models or metadata after the last save.

``finalize_suite`` saved metadata and persisted the ``_CT_ENSEMBLE__*`` entries, and then composite post-processing
created those entries, wrapped the composite models in place and stamped their metadata, so a reloaded suite served the
unwrapped inner models in T-scale and lacked the ensemble. This walks ``training/core``: a mutator is any function that
assigns into ``models[...]`` / ``metadata[...]`` or sets ``.model``, or calls one (transitively); a persister is a save
entry point, or a function whose last save follows its last mutation. In every orchestrator function that saves, no
mutator may be called after the last persister.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mlframe

from tests.test_meta._shared_ast_cache import parsed_ast

_CORE = Path(mlframe.__file__).resolve().parent / "training" / "core"
_PERSIST = {"finalize_suite", "_finalize_and_save_metadata", "_persist_ct_ensemble_entries", "persist_after_composite_post"}
_MUTATED = {"models", "metadata"}
# A mutation after the last save that is intentionally not persisted, with the reason. Empty: none is known.
_ALLOWED_LATE: dict[tuple[str, str], str] = {}


def _calls(node: ast.AST) -> list[tuple[int, str]]:
    """``(line, name)`` of every call under ``node``, by the called name or attribute."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            name = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
            if name:
                out.append((n.lineno, name))
    return out


def _assigns_into_suite_state(fn: ast.AST) -> bool:
    """True when ``fn`` assigns into ``models[...]`` / ``metadata[...]`` (bare or as an attribute) or sets ``.model``."""
    for n in ast.walk(fn):
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign)) else []
        for t in targets:
            if isinstance(t, ast.Attribute) and t.attr == "model":
                return True
            if isinstance(t, ast.Subscript):
                base = t.value
                while isinstance(base, ast.Subscript):
                    base = base.value
                if (isinstance(base, ast.Name) and base.id in _MUTATED) or (isinstance(base, ast.Attribute) and base.attr in _MUTATED):
                    return True
    return False


def _core_functions() -> dict[str, list[ast.AST]]:
    """Every function defined under ``training/core``, by name."""
    funcs: dict[str, list[ast.AST]] = {}
    for path in sorted(_CORE.rglob("*.py")):
        if "_benchmarks" in path.parts:
            continue
        tree = parsed_ast(path)
        if tree is None:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.setdefault(n.name, []).append(n)
    return funcs


def _classify(funcs: dict[str, list[ast.AST]]) -> tuple[set[str], set[str]]:
    """``(mutators, persisters)``: mutation is closed transitively over calls; a persister saves after it mutates."""
    direct = {name for name, defs in funcs.items() if any(_assigns_into_suite_state(fn) for fn in defs)}
    persisters = set(_PERSIST)
    for name, defs in funcs.items():
        for fn in defs:
            calls = _calls(fn)
            saves = [ln for ln, c in calls if c in _PERSIST]
            muts = [ln for ln, c in calls if c in direct] + ([fn.lineno] if name in direct else [])
            if saves and (not muts or max(saves) >= max(muts)):
                persisters.add(name)
    mutators, changed = set(direct), True
    while changed:
        changed = False
        for name, defs in funcs.items():
            if name not in mutators and any(c in mutators for fn in defs for _, c in _calls(fn)):
                mutators.add(name)
                changed = True
    return mutators - persisters, persisters


def _late_mutations(tree: ast.AST, mutators: set[str], persisters: set[str]) -> list[str]:
    """``function:line:call`` for every mutator called after the last persister in a function that persists."""
    late = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        calls = _calls(fn)
        saves = [ln for ln, c in calls if c in persisters]
        if saves:
            last = max(saves)
            late += [f"{fn.name}:{ln}:{c}" for ln, c in sorted(set(calls)) if c in mutators and ln > last and (fn.name, c) not in _ALLOWED_LATE]
    return late


def test_no_orchestrator_mutates_after_its_last_save():
    """In ``_main_train_suite*.py`` every mutation of models / metadata is followed by a save."""
    funcs = _core_functions()
    assert len(funcs) >= 300, f"only {len(funcs)} functions found under training/core; the scan no longer fits the tree"
    mutators, persisters = _classify(funcs)
    assert {"run_composite_post_processing", "_run_composite_target_wrapping"} <= mutators, "the composite wrap must count as a mutation"
    orchestrators = sorted(_CORE.glob("_main_train_suite*.py"))
    assert len(orchestrators) >= 3
    late = [f"{p.name}:{x}" for p in orchestrators for x in _late_mutations(parsed_ast(p), mutators, persisters)]
    assert not late, f"models / metadata mutated after the last save (the saved suite will not match the returned one): {late}"


def test_the_scan_sees_a_save_before_post_processing():
    """Canary: the shape that shipped INT-02 (save, then post-process, then return) is flagged; a re-save clears it."""
    src = "def tail(ctx):\n    finalize_suite(ctx)\n    run_composite_post_processing(ctx)\n"
    fixed = src + "    persist_after_composite_post(ctx)\n"
    mutators, persisters = {"run_composite_post_processing"}, set(_PERSIST)
    assert _late_mutations(ast.parse(src), mutators, persisters) == ["tail:3:run_composite_post_processing"]
    assert _late_mutations(ast.parse(fixed), mutators, persisters) == []
