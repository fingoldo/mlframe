"""Value-preserving simplification of engineered-feature op-NAMES.

The pair-FE / nested-composite namer (``feature_engineering.get_new_feature_name``)
emits literal op-trees like ``add(abs(div(sqr(a),neg(b))),mul(log(c),sin(d)))``. Such
names often carry algebraically DEAD operators -- e.g. the ``neg(b)`` inside
``abs(div(sqr(a),neg(b)))`` is annihilated by the enclosing ``abs`` (|p/(-q)| == |p/q|),
so the cleaner, IDENTICAL-VALUE name is ``abs(div(sqr(a),b))``.

This module rewrites the NAME STRING to its canonical form using ONLY value-preserving
identities, so ``transform()`` replay (which reads the recipe's stored op structure, not
the display name) is unaffected and the user-facing column label is simpler. It also makes
two recipes that differ only by a dead operator canonicalise to the SAME name (a dedup aid).

Correctness rule -- SIGN IRRELEVANCE. A ``neg`` is droppable iff every operator on the path
up to a SIGN-KILLER (``abs`` / even power ``sqr``) is SIGN-HOMOGENEOUS (``mul`` / ``div`` /
``neg`` itself); ``add`` / ``sub`` (and any non-sign-homogeneous unary) RESET sign-relevance,
because there ``neg`` changes the value. We thread a ``sign_irrelevant`` flag downward.

Identities applied (all numerically verified against the REAL guarded FE ops in
test_recipe_name_simplify -- abs, neg, sqr, sqrt, _safe_div, signed, abs_diff, hypot, ...):
  * under a sign-irrelevant context:  neg(x) -> x
  * abs(neg(x)) -> abs(x)   (via the flag)        * sqr(neg(x)) -> sqr(x)
  * sqrt(neg(x)) -> sqrt(x) (sqrt == sqrt(|x|))
  * abs(abs(x)) -> abs(x)                          * abs(sqr(x)) -> sqr(x)
  * neg(neg(x)) -> x                               * sqr(abs(x)) -> sqr(x)
  * abs(sqrt(x)) -> sqrt(x)                        * sqrt(abs(x)) -> sqrt(x)
  * abs(abs_diff(a,b)) -> abs_diff(a,b)            * abs(hypot(a,b)) -> hypot(a,b)
        (these binary ops already return a non-negative magnitude)
  * signed(a, neg(b)) -> signed(a, b)             (2nd arg of ``signed`` enters as |b|)
        signed is also sign-HOMOGENEOUS in its 1st arg (sign(neg a)*|b| == -(sign(a)*|b|)),
        so it propagates an enclosing sign-irrelevant context to BOTH children.

REJECTED candidates (numerically *would* hold in ideal math but were NOT adopted):
  * exp(log(x)) -> x        -- BROKEN: ``smart_log`` adds a data-dependent shift
                               (1e-5 - x_min), so exp(smart_log(x)) != x (fails on any
                               column with a negative minimum, e.g. standard-normal input).
  * sqr(sqrt(x)) -> abs(x), sqrt(sqr(x)) -> abs(x), reciproc(reciproc(x)) -> x
                            -- these DO hold to ~1e-16 on tested data, but each is a
                               cross-op float round-trip whose error is magnitude-dependent
                               and not bit-exact; rejected to keep every adopted rule a
                               purely structural / bit-identical rewrite (a wrong
                               simplification is far worse than a missed one).
"""
from __future__ import annotations

from typing import Sequence

# Operators whose OUTPUT sign does not depend on the sign of their argument(s):
# applying them makes the argument's sign irrelevant (a neg below can be dropped).
# ``sqrt`` qualifies because the FE impl is ``sqrt(|x|)`` (see feature_engineering.create_unary).
_SIGN_KILLERS = frozenset({"abs", "sqr", "sqrt"})
# Operators that PROPAGATE sign-irrelevance to their children (sign-homogeneous in each
# argument: flipping a child's sign only flips the result's sign, which a sign-killer above
# discards). ``neg`` is included (it is purely a sign flip). ``signed`` is sign-homogeneous
# in its 1st arg and its 2nd arg is already taken as |b|, so an enclosing sign-killer makes
# BOTH children's signs irrelevant.
_SIGN_HOMOGENEOUS = frozenset({"mul", "div", "neg", "signed"})
# Binary ops that ALREADY return a non-negative magnitude: an enclosing ``abs`` is dead.
_NONNEG_BINARY = frozenset({"abs_diff", "hypot"})
# Unary collapses (outer, inner) -> result-op, where the outer op is idempotent over the
# inner's already-constrained range. ``sqrt(sqrt)`` and ``sqr(sqr)`` are NOT here (genuine
# repeated application changes the value).
_ABS_LIKE = frozenset({"abs", "sqr", "sqrt"})


def _parse(name: str):
    """Parse an op-name into a nested (op, [args]) / leaf-string tree. Returns the tree and
    the index after the parsed expression. Leaves are any maximal run not containing ``(),``."""
    def parse_expr(i: int):
        """Recursive-descent parse of one expression starting at index ``i``; returns ``(node, next_index)`` where ``node`` is a leaf string or ``(op, [args])`` tuple."""
        # read an identifier / leaf token
        j = i
        while j < len(name) and name[j] not in "(),":
            j += 1
        token = name[i:j]
        if j < len(name) and name[j] == "(":
            args = []
            j += 1  # consume '('
            while True:
                arg, j = parse_expr(j)
                args.append(arg)
                if j < len(name) and name[j] == ",":
                    j += 1
                    continue
                break
            if j < len(name) and name[j] == ")":
                j += 1  # consume ')'
            return (token, args), j
        return token, j  # leaf

    tree, end = parse_expr(0)
    return tree, end


def _render(node) -> str:
    """Serialize a parsed (op, [args]) / leaf tree back into the ``op(arg1,arg2,...)`` name string."""
    if isinstance(node, str):
        return node
    op, args = node
    return f"{op}({','.join(_render(a) for a in args)})"


def _simplify(node, sign_irrelevant: bool):
    """Recursively simplify a parsed op tree: drops redundant/idempotent unary ops (e.g. a dead ``neg`` when the caller says the sign doesn't matter, or ``abs(abs(x))``-style collapses), returning an equivalent but shorter tree."""
    if isinstance(node, str):
        return node
    op, args = node

    if op == "neg" and len(args) == 1:
        if sign_irrelevant:
            return _simplify(args[0], True)  # drop the dead neg
        # neg(neg(x)) -> x  (two flips cancel regardless of context)
        inner = args[0]
        if isinstance(inner, tuple) and inner[0] == "neg" and len(inner[1]) == 1:
            return _simplify(inner[1][0], sign_irrelevant)
        return ("neg", [_simplify(args[0], False)])

    if op in _SIGN_KILLERS and len(args) == 1:
        child = _simplify(args[0], True)
        if isinstance(child, tuple):
            cop = child[0]
            # ``abs`` of a value already known non-negative is dead:
            #   abs(abs x)->abs x ; abs(sqr x)->sqr x ; abs(sqrt x)->sqrt x ;
            #   abs(abs_diff a,b)->abs_diff a,b ; abs(hypot a,b)->hypot a,b
            if op == "abs" and (cop in _ABS_LIKE or cop in _NONNEG_BINARY):
                return child
            # A sign-killer ignores its argument's sign, so a wrapping ``abs`` is dead:
            #   sqr(abs x)->sqr x ; sqrt(abs x)->sqrt x
            if cop == "abs" and len(child[1]) == 1:
                return (op, [child[1][0]])
        return (op, [child])

    if op == "signed" and len(args) == 2:
        # arg0 is sign-homogeneous (propagate context); arg1 enters as |b| -> its sign is
        # ALWAYS irrelevant, regardless of the enclosing context.
        return ("signed", [_simplify(args[0], sign_irrelevant), _simplify(args[1], True)])

    if op in _SIGN_HOMOGENEOUS:  # mul / div : propagate the (possibly irrelevant) context
        return (op, [_simplify(a, sign_irrelevant) for a in args])

    # add / sub / any other unary or binary op: sign of each child matters -> reset.
    return (op, [_simplify(a, False) for a in args])


def simplified_recipe_names(recipes: Sequence[object]) -> list[str]:
    """Order-preserving simplified DISPLAY names for a recipe list.

    Applies :func:`simplify_fe_name` to each ``recipe.name``, but ONLY adopts the simplified
    set if it stays exactly as UNIQUE as the originals -- otherwise returns the original names
    unchanged. This all-or-nothing guard prevents a (rare) canonicalisation COLLISION (two
    recipes differing only by a dead ``neg``/``abs``) from shrinking the column set, which would
    desync ``transform``'s output width from ``get_feature_names_out``. Both callers run the same
    deterministic function over the same recipe order, so their names stay identical.
    """
    orig = [getattr(r, "name", "") for r in recipes]
    simp = [simplify_fe_name(n) for n in orig]
    if len(set(simp)) == len(set(orig)):
        return simp
    return orig


def simplify_fe_name(name: str) -> str:
    """Return the value-preserving canonical form of an engineered op-NAME.

    Idempotent and safe on plain column names / unparseable strings (returns them unchanged).
    Never raises -- any parse hiccup falls back to the original name.
    """
    if not name or ("(" not in name):
        return name
    try:
        tree, end = _parse(name)
        if end != len(name):
            return name  # trailing garbage -> not a clean op-name, leave as-is
        return _render(_simplify(tree, False))
    except Exception:
        return name
