"""Every emitted engineered-recipe kind resolves to a real origin label, and the unlabeled-kind self-audit finds recipes by simplified name."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import mlframe.feature_selection.filters as filters_pkg
from mlframe.feature_selection.filters._mrmr_fe_provenance import (
    _RECIPE_KIND_TO_ORIGIN,
    DELIBERATELY_UNLABELED_KINDS,
    FE_ORIGIN_LABELS,
    get_unlabeled_recipe_kinds,
)
from mlframe.feature_selection.filters.engineered_recipes._recipe_name_simplify import simplify_fe_name

# numpy sort kinds, and the basis-registry category label in bases.py; none of these are recipe kinds.
_NON_RECIPE_KINDS = {"stable", "mergesort", "quicksort", "heapsort", "non-polynomial"}


def _emitted_recipe_kinds() -> set[str]:
    """String literals passed as ``kind=`` anywhere under filters/, minus numpy sort kinds."""
    kinds: set[str] = set()
    root = Path(filters_pkg.__file__).parent
    for path in root.rglob("*.py"):
        if "_benchmarks" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "kind" and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                kinds.add(node.value.value)
    return kinds - _NON_RECIPE_KINDS


def test_every_emitted_recipe_kind_has_an_origin_label():
    """A kind missing from the map silently reports its surviving columns as engineered_unknown."""
    emitted = _emitted_recipe_kinds()
    assert "orth_univariate" in emitted, "scanner precondition: known kinds must be found"
    missing = sorted(emitted - set(_RECIPE_KIND_TO_ORIGIN))
    assert not missing, f"recipe kinds with no provenance origin: {missing}"


def test_origin_map_targets_are_public_labels_and_only_deliberate_kinds_are_unknown():
    """Every mapped origin is a public label, and the only kinds mapped to engineered_unknown are the deliberate ones."""
    assert set(_RECIPE_KIND_TO_ORIGIN.values()) <= set(FE_ORIGIN_LABELS)
    unknown = {k for k, v in _RECIPE_KIND_TO_ORIGIN.items() if v == "engineered_unknown"}
    assert unknown == set(DELIBERATELY_UNLABELED_KINDS)


def test_unlabeled_kinds_resolves_simplified_names():
    """The provenance frame holds simplified names; the self-audit must still find the recipe and report its kind."""
    raw_name = "abs(div(sqr(a),neg(b)))"
    simplified = simplify_fe_name(raw_name)
    assert simplified != raw_name, "fixture precondition: the name must simplify"
    recipe = SimpleNamespace(name=raw_name, kind="some_unregistered_kind")
    prov = pd.DataFrame({"feature_name": [simplified], "origin": ["engineered_unknown"], "support_rank": [0]})
    est = SimpleNamespace(fe_provenance_=prov, _produced_recipes_=[], _engineered_recipes_=[recipe])
    assert get_unlabeled_recipe_kinds(est) == {"some_unregistered_kind": 1}
