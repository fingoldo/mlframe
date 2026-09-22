"""The over-materialisation prunes read provenance, so they work on real column names.

Both prunes decided a candidate's raw coverage by matching ``(?<![A-Za-z0-9_])([a-z](?:[a-z]?\\d+)?)(?![A-Za-z0-9_])`` against the rendered
feature name. That matches the ``a`` / ``b`` / ``c`` / ``d`` of the fixtures every comment in the block cites, and matches nothing in
``mul(log(revenue),sin(customer_age))``. On any real frame the coverage sets came back empty and both prunes short-circuited, so the
documented over-materialisation control silently did not exist.

The step already knows each candidate's operands: ``prospective_additions`` is keyed by the operand indices, and engineered operands carry
their own ``src_names``. Resolving that chain gives the same answer for ``a`` and for ``revenue``.
"""

from __future__ import annotations

import pytest

from mlframe.feature_selection.filters._mrmr_fe_step._step_name_tokens import build_candidate_provenance


class _Recipe:
    """Minimal stand-in for an engineered recipe, carrying only the parent names the resolver follows."""

    def __init__(self, *src_names: str):
        self.src_names = tuple(src_names)


def _additions(pair_index, names):
    """One ``prospective_additions`` entry in the shape the step builds: keyed by operand indices."""
    return {pair_index: (object(), object(), list(names), [8] * len(names), [])}


@pytest.mark.parametrize(
    "cols",
    [
        ["a", "b", "c", "d"],
        ["revenue", "customer_age", "tenure_days", "region_code"],
    ],
    ids=["single_letter_fixture_names", "realistic_names"],
)
def test_raw_coverage_is_found_whatever_the_columns_are_called(cols):
    """The coverage of a candidate built from operands 0 and 2 is those two columns, under either naming."""
    raw_of, gates_of = build_candidate_provenance(_additions((0, 2), ["engineered_0"]), cols, {}, {})
    assert raw_of["engineered_0"] == frozenset({cols[0], cols[2]}), f"coverage came back {raw_of['engineered_0']}"
    assert gates_of["engineered_0"] == ()


def test_an_engineered_operand_resolves_to_its_own_raw_sources():
    """A candidate built on an engineered parent covers what that parent covers, not the parent's name."""
    cols = ["revenue", "prior_engineered", "tenure_days"]
    recipes = {"prior_engineered": _Recipe("customer_age", "region_code")}
    raw_of, _gates = build_candidate_provenance(_additions((1, 2), ["child"]), cols, recipes, {})
    assert raw_of["child"] == frozenset({"customer_age", "region_code", "tenure_days"})


def test_a_gate_operand_is_reported_as_a_gate_and_not_as_a_raw_variable():
    """Gate columns are expanded by the caller through the gate map, so they must not also count as raw coverage."""
    cols = ["revenue", "gate_mask__tenure__region"]
    gate_map = {"gate_mask__tenure__region": {"tenure_days", "region_code"}}
    recipes = {"gate_mask__tenure__region": _Recipe("tenure_days", "region_code")}
    raw_of, gates_of = build_candidate_provenance(_additions((0, 1), ["composite"]), cols, recipes, gate_map)
    assert gates_of["composite"] == ("gate_mask__tenure__region",)
    assert "gate_mask__tenure__region" not in raw_of["composite"], "the gate column itself must not read as a raw variable"
    assert "revenue" in raw_of["composite"]


def test_a_gate_is_matched_by_name_not_by_substring():
    """``gate_mask__b__d`` must not be taken to be present in the unrelated ``gate_mask__b__d2``."""
    cols = ["gate_mask__b__d2", "revenue"]
    gate_map = {"gate_mask__b__d": {"b", "d"}}
    _raw, gates_of = build_candidate_provenance(_additions((0, 1), ["composite"]), cols, gate_map and {}, gate_map)
    assert gates_of["composite"] == (), "a different gate column was matched by substring"


def test_a_cyclic_provenance_chain_terminates():
    """A recipe chain that refers back to itself must not hang the prune."""
    cols = ["looping", "revenue"]
    recipes = {"looping": _Recipe("looping")}
    raw_of, _gates = build_candidate_provenance(_additions((0, 1), ["composite"]), cols, recipes, {})
    assert raw_of["composite"] == frozenset({"revenue"})


def test_a_candidate_with_no_new_columns_contributes_nothing():
    """An entry the step filtered down to nothing must not appear in the provenance maps."""
    raw_of, gates_of = build_candidate_provenance({(0, 1): (None, None, [], [], [])}, ["a", "b"], {}, {})
    assert raw_of == {} and gates_of == {}
