"""The empty-screen rescue excludes raw columns that are operands of surviving engineered recipes, read from the recipe's column name.

``_engineered_recipes_`` holds recipe objects; tokenising their ``repr`` also matched every name in ``kind``, ``extra`` and nested fields, so a raw
column that merely shared a token with that metadata was wrongly kept out of the rescue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mlframe.feature_selection.filters._mrmr_fit_impl._finalise import surviving_recipe_operands


@dataclass
class _Recipe:
    """A recipe-shaped object whose repr carries more names than its column name."""

    name: str
    kind: str = "unary_binary"
    src_names: tuple = ()
    extra: dict = field(default_factory=dict)


def test_rescue_operand_exclusion_uses_recipe_name_not_repr():
    """Only the operands in the column name are excluded; a raw name that appears only in the metadata stays eligible."""
    recipes = [_Recipe(name="log(a)", src_names=("a",), extra={"note": "b"})]
    assert surviving_recipe_operands(recipes, ["a", "b", "c"]) == {"a"}


def test_suffixed_leg_and_legacy_string_entries_resolve_to_their_raw_column():
    """A ``__``-suffixed leg maps to its raw prefix, and a bare-string legacy entry is read as the name."""
    recipes = [_Recipe(name="mul(a__relu_gt_0.5,c)"), "div(b,c)"]
    assert surviving_recipe_operands(recipes, ["a", "b", "c", "d"]) == {"a", "b", "c"}
