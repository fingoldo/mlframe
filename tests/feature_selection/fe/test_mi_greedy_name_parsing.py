"""The engineered-name parser cuts a binary name at the transform token of the outermost operation, not at a token inside a nested operand."""

from __future__ import annotations

import itertools

import pytest

from mlframe.feature_selection.filters._mi_greedy_fe import BINARY_TRANSFORMS, _parse_binary_name, engineered_name_binary

# The name the warm fit's final subsumption and residual rescue could not replay: the right operand holds an ``__add__`` of its own.
_NESTED = "(add(div(invsqrt(b),exp(a__LL2)),mul(log(c),sin(d)))__sub__add(sqrt((a__add__c)),sqr(opat__a_b_d)))"
_LEFT = "add(div(invsqrt(b),exp(a__LL2)),mul(log(c),sin(d)))"
_RIGHT = "add(sqrt((a__add__c)),sqr(opat__a_b_d))"


def _first_token_parse(name: str):
    """The parser as it was: the first occurrence of the first transform that occurs anywhere in the string (the witness that the old rule cut the name wrongly)."""
    inner = name[1:-1]
    for tname in BINARY_TRANSFORMS:
        token = f"__{tname}__"
        idx = inner.find(token)
        if idx >= 0:
            return tname, inner[:idx], inner[idx + len(token) :]
    return None


def test_the_name_that_broke_the_replay_is_cut_at_its_outermost_operation():
    """The nested ``__add__`` of the right operand is not the joint: the parse is (sub, left operand, right operand)."""
    assert _parse_binary_name(_NESTED) == ("sub", _LEFT, _RIGHT)
    assert _first_token_parse(_NESTED) != ("sub", _LEFT, _RIGHT), "the old rule cut this name at the nested token; the witness no longer shows the defect"


OPERANDS = [
    "a",
    "x_1",
    "sqrt(a)",
    "(a__add__b)",
    "(a__mul__(b__div__c))",
    "add(sqrt((a__add__c)),sqr(c))",
    "a__LL2",
    "log(a__LL2)",
    "(sqrt(a)__abs_diff__exp(b))",
]


@pytest.mark.parametrize("transform", list(BINARY_TRANSFORMS))
def test_every_transform_round_trips_over_nested_operands(transform):
    """name_binary(left, right, transform) parses back to (transform, left, right) for operands that hold tokens of other transforms."""
    for left, right in itertools.product(OPERANDS, repeat=2):
        name = engineered_name_binary(left, right, transform)
        assert _parse_binary_name(name) == (transform, left, right), name


def test_plain_names_and_non_binary_names_are_unchanged():
    """The simple forms parse as before; a unary name, a bare column and a name with no transform token are not binary."""
    assert _parse_binary_name("(a__ratio_log__b)") == ("ratio_log", "a", "b")
    assert _parse_binary_name("sqrt(a)") is None
    assert _parse_binary_name("a__add__b") is None
    assert _parse_binary_name("(a_b)") is None


def test_a_name_with_unbalanced_parentheses_falls_back_to_the_first_token():
    """A column name that itself holds an unclosed parenthesis cannot be depth-tracked; the parse keeps the earlier rule instead of failing."""
    assert _parse_binary_name("(a(b__add__c)") == ("add", "a(b", "c")
