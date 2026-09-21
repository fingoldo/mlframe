"""Reading raw-variable references out of an engineered feature name.

The FE step's coverage logic decides whether a candidate is already represented by a surviving feature by comparing which raw variables each
name mentions, so what counts as a mention is load-bearing: the lookarounds are what stop ``log(c)`` from reading as a variable ``l``.
"""

from __future__ import annotations

import pytest

from mlframe.feature_selection.filters._mrmr_fe_step._step_name_tokens import bare_tokens, gate_cols_in


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a", {"a"}),
        ("x12", {"x12"}),
        ("mul(a,b)", {"a", "b"}),
        ("div(sqr(a),abs(b))", {"a", "b"}),
        ("mul(log(c),sin(d))", {"c", "d"}),
        ("", set()),
    ],
)
def test_bare_tokens_reads_the_variables_a_name_mentions(name, expected):
    """Single-letter and letter-digit variables are found; the surrounding function names are not mistaken for them."""
    assert bare_tokens(name) == expected


@pytest.mark.parametrize("name", ["log(c)", "abs(b)", "sqr(a)", "sin(d)"])
def test_a_function_name_never_contributes_a_variable_of_its_own(name):
    """Only the operand is a variable: the function's own letters must not read as bare tokens."""
    assert bare_tokens(name) == {name[name.index("(") + 1]}


def test_an_underscore_suffixed_name_does_not_mention_its_own_stem():
    """``a__He2`` reads as no variable at all: the token must not be followed by an underscore.

    That is what keeps a basis-suffixed engineered column from counting as a mention of its own source, so coverage is decided on composite
    operands rather than on every engineered column's stem. Surprising enough to pin deliberately.
    """
    assert bare_tokens("a__He2") == set()
    assert bare_tokens("mul(a__He2,b)") == {"b"}


def test_gate_cols_in_finds_the_gate_composites_a_name_contains():
    """A gate composite counts when its own name appears inside the candidate's name."""
    gate_map = {"gate_ab": ("a", "b"), "gate_cd": ("c", "d")}
    assert gate_cols_in("mul(gate_ab,x1)", gate_map) == ["gate_ab"]
    assert sorted(gate_cols_in("add(gate_ab,gate_cd)", gate_map)) == ["gate_ab", "gate_cd"]
    assert gate_cols_in("mul(a,b)", gate_map) == []
    assert gate_cols_in("anything", {}) == []
