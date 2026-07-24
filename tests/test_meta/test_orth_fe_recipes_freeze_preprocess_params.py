"""Meta-test: every orthogonal/Hermite-family FE module exposing a ``*_with_recipes`` recipe-emitting
function also references ``preprocess_params`` somewhere in the same module -- the mechanical proxy for
the B-17 freeze-and-replay contract (``EngineeredRecipe``/``preprocess_params`` must freeze fit-time basis
preprocess params so slice replay reproduces the fit-time value bit-for-bit, instead of silently refitting
from the slice's own local statistics).

mrmr_audit_2026-07-22 meta-test proposal #2 (mechanical half): the behavioral regression test
(``test_regression_orth_scorer_zoo_recipe_freezes_preprocess_params`` in
``tests/feature_selection/regression/test_regression_mrmr_audit_2026_07_22.py``) pins the B-17 fix across
the 13 modules commit f067e0d44 patched, by actually replaying a slice and comparing bit-for-bit -- the
strongest possible check, but it only runs against modules someone remembered to add to its
parametrize list. This static scan is a cheaper, automatically-comprehensive backstop: it can't prove
correctness the way the behavioral test does, but it catches the "a brand-new orth-family module was
added and nobody wired the freeze pattern in at all" case for free, the moment the module exists --
without waiting for someone to notice and hand-add it to the behavioral test's module list.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import orjson
import pytest

import mlframe

from tests.test_meta._shared_ast_cache import parsed_ast, source_text

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent
_FILTERS_DIR = MLFRAME_DIR / "feature_selection" / "filters"
_BASELINE_PATH = Path(__file__).resolve().parent / "_orth_fe_recipes_no_freeze_baseline.json"

_FAMILY_DIR_GLOBS = ("_orthogonal_*.py", "_orthogonal_univariate_fe/*.py", "hermite_fe/*.py")
_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "_benchmarks", "_vendored")


def _refresh_requested() -> bool:
    """True if ``--refresh-orth-fe-recipes-no-freeze-baseline`` was passed on the pytest command line."""
    return "--refresh-orth-fe-recipes-no-freeze-baseline" in sys.argv


def _defines_with_recipes_function(tree: ast.Module) -> bool:
    """True if the module defines a top-level function whose name ends in ``_with_recipes`` -- the
    codebase's own consistent naming convention for a recipe-emitting FE constructor."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.endswith("_with_recipes"):
            return True
    return False


def _candidate_files() -> list[Path]:
    """Every orth/Hermite-family source file under ``_FAMILY_DIR_GLOBS``, deduplicated and exempt-filtered."""
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in _FAMILY_DIR_GLOBS:
        for py in _FILTERS_DIR.glob(pattern):
            if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
                continue
            if py in seen:
                continue
            seen.add(py)
            out.append(py)
    return out


def _build_offending_set() -> set[str]:
    """Relpaths of every orth/Hermite-family module defining a ``*_with_recipes`` function but never
    referencing ``preprocess_params`` anywhere in the module."""
    out: set[str] = set()
    for py in _candidate_files():
        tree = parsed_ast(py)
        if tree is None:
            continue
        if not _defines_with_recipes_function(tree):
            continue
        text = source_text(py)
        if text is not None and "preprocess_params" in text:
            continue
        rel = py.relative_to(MLFRAME_DIR).as_posix()
        out.add(rel)
    return out


def test_no_new_orth_fe_recipe_module_missing_preprocess_params():
    """No new orthogonal/Hermite-family ``*_with_recipes`` module omits ``preprocess_params`` entirely,
    beyond the frozen baseline (each baseline entry is a module confirmed to legitimately delegate its
    recipe-emission to a sibling module rather than build recipes with a local preprocess step itself)."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"orth-fe-recipes-no-freeze baseline refreshed at {_BASELINE_PATH.name} ({len(current)} module(s))")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_orth_fe_recipe_module_missing_preprocess_params] {len(fixed)} module(s) "
            f"DRAINED:\n  " + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh: pytest ... --refresh-orth-fe-recipes-no-freeze-baseline\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} new orth/Hermite-family module(s) define a ``*_with_recipes`` function but "
            f"never reference ``preprocess_params`` -- either the module legitimately delegates its "
            f"recipe-emission to a sibling (add it to the baseline via --refresh-... after confirming "
            f"that), or it's missing the B-17 freeze-and-replay contract and needs a "
            f"``preprocess_params`` entry threaded into its emitted recipe's ``extra``:\n  "
            + "\n  ".join(new[:30]) + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )
