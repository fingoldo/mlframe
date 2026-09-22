"""Which raw columns an engineered candidate is built from.

The FE step's over-materialisation prunes decide whether a candidate's coverage is already provided by the candidates that survived the CMI
gate, so they need the set of raw variables each candidate touches. That was read out of the rendered feature name with a regex matching one
lowercase letter plus optional digits, which matches the ``a`` / ``b`` / ``x12`` of the fixtures every comment in the block cites and matches
nothing at all in ``mul(log(revenue),sin(customer_age))``. On any real frame the coverage sets came back empty, the prunes short-circuited,
and the documented protection silently did not exist.

The step already knows the answer exactly: ``prospective_additions`` is keyed by the ``(i, j)`` operand indices the candidate was built from,
and engineered operands carry their own ``src_names``, so the provenance resolves by walking that chain to the raw columns rather than by
guessing at the name.
"""

from __future__ import annotations


def _resolve_raw(name, engineered_recipes, _seen=None) -> frozenset:
    """The raw column names behind ``name``, following engineered parents through their recipes' ``src_names``."""
    recipe = engineered_recipes.get(name) if engineered_recipes else None
    src_names = tuple(getattr(recipe, "src_names", ()) or ()) if recipe is not None else ()
    if not src_names:
        return frozenset((name,))  # no recipe, or a recipe with no parents: this is as raw as it gets
    seen = set() if _seen is None else _seen
    if name in seen:
        return frozenset()  # a cyclic provenance chain cannot be resolved, and must not hang the prune
    seen = seen | {name}
    out: set = set()
    for parent in src_names:
        out |= _resolve_raw(str(parent), engineered_recipes, seen)
    return frozenset(out)


def build_candidate_provenance(prospective_additions, cols, engineered_recipes, gate_map) -> tuple[dict, dict]:
    """``(raw_sources_of, gates_of)`` for every candidate name in ``prospective_additions``.

    ``raw_sources_of[name]`` is the set of raw column names the candidate ultimately reads, with engineered and gate operands resolved through
    to their own sources. ``gates_of[name]`` is the gate columns in its provenance, matched by exact name rather than by substring, so a gate
    ``gate_mask__b__d`` is no longer taken to be present in the unrelated ``gate_mask__b__d2``.
    """
    raw_sources_of: dict = {}
    gates_of: dict = {}
    n_cols = len(cols)
    for pair_key, entry in prospective_additions.items():
        new_cols = entry[2] if isinstance(entry, tuple) and len(entry) > 2 else None
        if not new_cols:
            continue
        indices = pair_key if isinstance(pair_key, (tuple, list)) else ()
        operands = [str(cols[i]) for i in indices if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n_cols]
        raws: set = set()
        gates: set = set()
        for operand in operands:
            if operand in gate_map:
                gates.add(operand)
            raws |= _resolve_raw(operand, engineered_recipes)
        # A gate column's own name is provenance, not a raw variable: the callers expand it via the gate map instead.
        raws -= set(gate_map)
        for name in new_cols:
            raw_sources_of[str(name)] = frozenset(raws)
            gates_of[str(name)] = tuple(sorted(gates))
    return raw_sources_of, gates_of
