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

# The reference interpreter is wired to the ACTUAL FE op implementations (guarded ``_safe_div``,
# data-dependent ``smart_log``, ``sqrt(|x|)``, non-negative ``abs_diff``/``hypot``, the
# non-symmetrical ``signed``) so every value-preservation assertion respects the real guard
# semantics rather than idealised math. An identity that holds in ideal algebra but breaks
# under these guards (e.g. ``exp(log x) -> x``) must therefore FAIL this harness if mis-added.
from mlframe.feature_selection.filters.feature_engineering import (
    create_unary_transformations,
    create_binary_transformations,
)

_REAL_UNARY = create_unary_transformations("medium")
_REAL_BINARY = create_binary_transformations("medium")

_UNARY = {
    "abs": _REAL_UNARY["abs"],
    "neg": _REAL_UNARY["neg"],
    "sqr": _REAL_UNARY["sqr"],
    "sqrt": _REAL_UNARY["sqrt"],
    "log": _REAL_UNARY["log"],
    "sin": _REAL_UNARY["sin"],
    "cbrt": _REAL_UNARY["cbrt"],
}
_BINARY = {
    "mul": _REAL_BINARY["mul"],
    "div": _REAL_BINARY["div"],
    "add": _REAL_BINARY["add"],
    "sub": _REAL_BINARY["sub"],
    "abs_diff": _REAL_BINARY["abs_diff"],
    "hypot": _REAL_BINARY["hypot"],
    "signed": _REAL_BINARY["signed"],
}


def _hard_domain(rng, n):
    """A column mixing negatives, near-zero, exact-zero and large magnitudes so the guarded
    FE ops (smart_log shift, _safe_div eps floor, sqrt(|x|), signed) are all exercised."""
    parts = [
        rng.standard_normal(n // 4),
        rng.standard_normal(n // 4) * 1e6,
        (rng.random(n // 4) - 0.5) * 1e-8,
        np.zeros(n - 3 * (n // 4)),
    ]
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


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
    # sqrt is a third sign-killer (FE sqrt == sqrt(|x|))
    ("sqrt(neg(a))", "sqrt(a)"),
    ("abs(sqrt(a))", "sqrt(a)"),
    ("sqrt(abs(a))", "sqrt(a)"),
    ("sqrt(neg(mul(a,neg(b))))", "sqrt(mul(a,b))"),
    # abs of an already-non-negative binary magnitude is dead
    ("abs(abs_diff(a,b))", "abs_diff(a,b)"),
    ("abs(hypot(a,b))", "hypot(a,b)"),
    # signed: 2nd arg enters as |b| -> neg there is always dead; 1st arg is sign-homogeneous
    ("signed(a,neg(b))", "signed(a,b)"),
    ("abs(signed(neg(a),neg(b)))", "abs(signed(a,b))"),
    ("signed(neg(a),b)", "signed(neg(a),b)"),  # 1st-arg neg matters when NOT under a sign-killer
    ("sqr(signed(a,neg(b)))", "sqr(signed(a,b))"),
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
    env = {n: _hard_domain(rng, 1500) for n in leaves}
    got = _eval(simplify_fe_name(original), env)
    ref = _eval(original, env)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9, equal_nan=True), (
        f"{original} -> {simplify_fe_name(original)} changed the value"
    )


def test_random_fuzz_value_preserving():
    """Random nested op-trees: the simplified name must denote the identical value."""
    rng = np.random.default_rng(7)
    uns = ["abs", "neg", "sqr", "sqrt", "log", "sin"]
    bins = ["mul", "div", "add", "sub", "abs_diff", "hypot", "signed"]
    leaves = ["a", "b", "c"]

    def gen(depth):
        if depth <= 0 or rng.random() < 0.3:
            return rng.choice(leaves)
        if rng.random() < 0.5:
            return f"{rng.choice(uns)}({gen(depth-1)})"
        return f"{rng.choice(bins)}({gen(depth-1)},{gen(depth-1)})"

    env = {n: _hard_domain(rng, 1200) for n in leaves}
    for _ in range(300):
        name = gen(4)
        simp = simplify_fe_name(name)
        assert np.allclose(_eval(simp, env), _eval(name, env), rtol=1e-9, atol=1e-9, equal_nan=True), (
            f"value changed: {name} -> {simp}"
        )
        assert simplify_fe_name(simp) == simp, f"not idempotent: {simp}"


def test_rejected_identities_not_applied():
    """Guard: identities that are value-preserving in ideal math but BROKEN under the real
    guarded ops (or only float-approximate) must NOT be silently introduced by the simplifier."""
    # exp(log x) != x because smart_log adds a data-dependent shift -> must stay verbatim.
    # (exp/log round-trip is not even attempted, but pin the structural no-op anyway.)
    for nm in ("exp(log(a))", "sqr(sqrt(a))", "sqrt(sqr(a))", "reciproc(reciproc(a))"):
        # whatever simplify returns, it must still denote the identical value under real ops;
        # the safest correct behaviour is to leave these cross-op pairs unchanged.
        assert simplify_fe_name(nm) == nm, f"unexpected rewrite of rejected pair: {nm}"
