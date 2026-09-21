"""A shuffled KFold in discovery or ensemble code is built only by the shared splitter factory.

Hand-built ``KFold(shuffle=True)`` next to a CV that honours groups and time order is how the WAIC tie-break and the
auto-chain admission came to score specs on the per-group memorisation and look-ahead the tiny rerank was fixed to reject.
``make_discovery_splitter`` applies one precedence (groups, then time, then contiguous or shuffled) everywhere.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mlframe

from tests.test_meta._shared_ast_cache import parsed_ast

_PKG = Path(mlframe.__file__).resolve().parent
_SCOPE = ("training/composite/discovery", "training/composite/ensemble", "training/core/_phase_composite_post_xt_ensemble")
_FACTORY = "training/composite/discovery/_splitter.py"
_SHUFFLED = {"KFold", "StratifiedKFold", "RepeatedKFold", "ShuffleSplit"}


def _shuffled_constructions(tree: ast.Module) -> list[int]:
    """Line numbers of ``KFold(..., shuffle=True)``-style constructions (ShuffleSplit shuffles by definition)."""
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        if name not in _SHUFFLED:
            continue
        shuffles = name == "ShuffleSplit" or any(k.arg == "shuffle" and not (isinstance(k.value, ast.Constant) and k.value.value is False) for k in node.keywords)
        if shuffles:
            lines.append(node.lineno)
    return lines


def test_shuffled_kfold_is_built_only_by_the_splitter_factory():
    """Only ``_splitter.py`` may construct a shuffled KFold in discovery / ensemble / the cross-target builder."""
    scanned, offenders = 0, []
    for rel in _SCOPE:
        for path in sorted((_PKG / rel).rglob("*.py")):
            if "_benchmarks" in path.parts:
                continue
            tree = parsed_ast(path)
            if tree is None:
                continue
            scanned += 1
            relpath = path.relative_to(_PKG).as_posix()
            if relpath == _FACTORY:
                continue
            offenders += [f"{relpath}:{ln}" for ln in _shuffled_constructions(tree)]
    assert scanned >= 60, f"scanned only {scanned} modules; the scope paths no longer match the tree"
    assert not offenders, f"shuffled KFold built outside make_discovery_splitter: {offenders}"


def test_the_scan_sees_a_shuffled_kfold():
    """Canary: the detector fires on the shape it exists for, and not on an unshuffled or explicitly unshuffled one."""
    tree = ast.parse("KFold(n_splits=3, shuffle=True, random_state=0)\nKFold(3)\nKFold(3, shuffle=False)\nShuffleSplit(3)\n")
    assert _shuffled_constructions(tree) == [1, 4]
