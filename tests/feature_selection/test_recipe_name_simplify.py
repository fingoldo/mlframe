"""Value-preservation + canonical-form pins for ``simplify_fe_name``.

The simplifier must (1) produce the expected cleaner name and (2) NEVER change the value the
op-name denotes -- verified by evaluating BOTH the original and simplified names on random data
through an independent reference interpreter.
"""
import numpy as np
import pytest

from mlframe.feature_selection.filters.engineered_recipes._recipe_name_simplify import (
    simplify_fe_name,
    _parse,
    _render,
)

_UNARY = {
    "abs": np.abs,
    "neg": np.negative,
    "sqr": lambda v: v * v,
    "log": lambda v: np.log(np.abs(v) + 1e-9),
    "sin": np.sin,
    "cbrt": np.cbrt,
}
_BINARY = {
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / np.where(np.abs(b) < 1e-9, 1e-9, b),
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
}


def _eval(name, env):
    tree, end = _parse(name)
    assert end == len(name)

    def ev(node):
        if isinstance(node, str):
            return env[node]
        op, args = node
        if op in _UNARY:
            return _UNARY[op](ev(args[0]))
        if op in _BINARY:
            return _BINARY[op](ev(args[0]), ev(args[1]))
        raise KeyError(op)

    return ev(tree)


# (original, expected-canonical)
_CASES = [
    # the user's reported case: abs annihilates the neg through div
    ("abs(div(sqr(a),neg(b)))", "abs(div(sqr(a),b))"),
    ("add(abs(div(sqr(a),neg(b))),mul(log(c),sin(d)))", "add(abs(div(sqr(a),b)),mul(log(c),sin(d)))"),
    ("abs(neg(a))", "abs(a)"),
    ("sqr(neg(a))", "sqr(a)"),
    ("neg(neg(a))", "a"),
    ("abs(abs(a))", "abs(a)"),
    ("abs(sqr(a))", "sqr(a)"),
    ("sqr(abs(a))", "sqr(a)"),
    ("abs(mul(neg(a),b))", "abs(mul(a,b))"),
    ("abs(div(neg(a),neg(b)))", "abs(div(a,b))"),
    # NEGATIVES: neg under add/sub is NOT droppable (changes value) -> unchanged
    ("abs(add(neg(a),b))", "abs(add(neg(a),b))"),
    ("add(neg(a),b)", "add(neg(a),b)"),
    ("neg(a)", "neg(a)"),
    # plain leaves / non-op names unchanged
    ("a", "a"),
    ("cat_a__te_std", "cat_a__te_std"),
]


@pytest.mark.parametrize("original,expected", _CASES)
def test_canonical_form(original, expected):
    assert simplify_fe_name(original) == expected
    # idempotent
    assert simplify_fe_name(expected) == expected


@pytest.mark.parametrize("original,expected", _CASES)
def test_value_preserving(original, expected):
    if "(" not in original:
        return  # plain leaf / non-op name: nothing to evaluate (canonical-form test covers it)
    rng = np.random.default_rng(0)
    leaves = sorted({n for n in ("a", "b", "c", "d") if n in original})
    if not leaves:
        return
    env = {n: rng.standard_normal(500) for n in leaves}
    got = _eval(simplify_fe_name(original), env)
    ref = _eval(original, env)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9, equal_nan=True), (
        f"{original} -> {simplify_fe_name(original)} changed the value"
    )


def test_random_fuzz_value_preserving():
    """Random nested op-trees: the simplified name must denote the identical value."""
    rng = np.random.default_rng(7)
    uns = ["abs", "neg", "sqr", "log", "sin"]
    bins = ["mul", "div", "add", "sub"]
    leaves = ["a", "b", "c"]

    def gen(depth):
        if depth <= 0 or rng.random() < 0.3:
            return rng.choice(leaves)
        if rng.random() < 0.5:
            return f"{rng.choice(uns)}({gen(depth-1)})"
        return f"{rng.choice(bins)}({gen(depth-1)},{gen(depth-1)})"

    env = {n: rng.standard_normal(400) for n in leaves}
    for _ in range(300):
        name = gen(4)
        simp = simplify_fe_name(name)
        assert np.allclose(_eval(simp, env), _eval(name, env), rtol=1e-9, atol=1e-9, equal_nan=True), (
            f"value changed: {name} -> {simp}"
        )
        assert simplify_fe_name(simp) == simp, f"not idempotent: {simp}"
