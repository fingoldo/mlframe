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

Identities applied (all numerically verified in test_recipe_name_simplify):
  * under a sign-irrelevant context:  neg(x) -> x
  * abs(neg(x)) -> abs(x)   (via the flag)        * sqr(neg(x)) -> sqr(x)
  * abs(abs(x)) -> abs(x)                          * abs(sqr(x)) -> sqr(x)
  * neg(neg(x)) -> x                               * sqr(abs(x)) -> sqr(x)
"""
from __future__ import annotations

# Operators whose OUTPUT sign does not depend on the sign of their argument(s):
# applying them makes the argument's sign irrelevant (a neg below can be dropped).
_SIGN_KILLERS = frozenset({"abs", "sqr"})
# Operators that PROPAGATE sign-irrelevance to their children (sign-homogeneous in each
# argument: flipping a child's sign only flips the result's sign, which a sign-killer above
# discards). ``neg`` is included (it is purely a sign flip).
_SIGN_HOMOGENEOUS = frozenset({"mul", "div", "neg"})


def _parse(name: str):
    """Parse an op-name into a nested (op, [args]) / leaf-string tree. Returns the tree and
    the index after the parsed expression. Leaves are any maximal run not containing ``(),``."""
    def parse_expr(i: int):
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
    if isinstance(node, str):
        return node
    op, args = node
    return f"{op}({','.join(_render(a) for a in args)})"


def _simplify(node, sign_irrelevant: bool):
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
        # idempotent collapses: abs(abs)->abs, abs(sqr)->sqr, sqr(abs)->sqr, sqr(sqr) stays
        if isinstance(child, tuple) and len(child[1]) == 1:
            cop = child[0]
            if op == "abs" and cop in ("abs", "sqr"):
                return child            # abs(abs x)->abs x ; abs(sqr x)->sqr x
            if op == "sqr" and cop == "abs":
                return ("sqr", [child[1][0]])  # sqr(abs x)->sqr x
        return (op, [child])

    if op in _SIGN_HOMOGENEOUS:  # mul / div : propagate the (possibly irrelevant) context
        return (op, [_simplify(a, sign_irrelevant) for a in args])

    # add / sub / any other unary or binary op: sign of each child matters -> reset.
    return (op, [_simplify(a, False) for a in args])


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
