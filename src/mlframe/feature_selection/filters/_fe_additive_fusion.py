"""C2 ADDITIVE-FUSION of two unfused, additively-separable engineered halves.

When the FE pair search constructs the two additively-separable halves of a compound
target ``y = f(group_1) + g(group_2) + noise`` -- e.g. ``y = a**2/b + log(c)*sin(d)``
materialised as the SEPARATE engineered features ``div(neg(b),a__p2sin1)`` (the {a,b}
half) and ``mul(log(c),sin(d))`` (the {c,d} half) -- but does NOT fuse them into the
single ``add(...)`` compound, the two fragments survive side by side. The conditional-MI
redundancy gates cannot collapse them (each fragment carries a PRIVATE additive term the
OTHER does not span, so both keep a large CMI) and the downstream model never sees the
fused feature. This is the FUSION-blocked failure mode of the distribution-robustness goal
(``test_f2_single_compound_across_distributions`` heavy_tailed / mixed).

The fix reuses the EXISTING ``unary_binary`` recipe machinery -- NO new recipe kind. A
fused candidate ``add(half_a, half_b)`` is built with ``binary_name='add'``,
``unary_names=('identity','identity')`` and ``nested_parent_a/b`` set to the two halves'
own ``EngineeredRecipe`` objects, so it replays byte-exactly by recursively replaying the
parents (``_recipe_unary_binary.py``). The candidate is proposed ONLY when two surviving
engineered features (or one engineered + one raw operand) have:
  * DISJOINT raw-token sets (no shared raw operand -- they cover different signal groups);
  * each half RELEVANT (its binned MI clears a marginal-permutation null floor);
  * GENUINE additive separability: the fused ``add`` MI strictly exceeds BOTH halves'
    MI by a margin (so an unrelated pair whose sum carries no joint uplift is never fused).
When admitted, the fused compound is materialised exactly like an escalation survivor and
the two now-subsumed fragment columns are dropped from ``selected_vars`` / the recipe dict
(the fused compound carries both their additive terms, so each fragment's conditional
excess given the fused feature collapses -- the redundancy logic the S5 gate would apply,
realised structurally here because the fragments pre-date the fusion candidate).

Default-ON (``fe_additive_fusion_enable``); self-gates to a no-op when fewer than two
relevant disjoint engineered halves are present (the common case), so it is byte-identical
on every target that does not exhibit the unfused-additive-halves pattern.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Split an engineered name into identifier tokens (``div(neg(b),a__p2sin1)`` ->
# {div, neg, b, a__p2sin1}); the raw operands are the tokens that are raw columns OR
# whose ``base__warp`` prefix is a raw column (the warped-token form the orth-FE /
# prewarp passes emit, e.g. ``a__p2sin1`` -> base ``a``). Mirrors the token recovery in
# ``_fe_raw_redundancy_drop._TOKEN_SPLIT`` so the fusion's raw-coverage agrees with the
# downstream raw-redundancy verdict.
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9_]+")


def _bare_tokens(name: str, raw_name_set: set) -> set:
    """Raw-column operands referenced by an engineered (or raw) name -- including warped
    ``base__suffix`` tokens mapped back to their raw base."""
    out: set = set()
    if name in raw_name_set:
        out.add(name)
    for t in _TOKEN_SPLIT.split(name or ""):
        if not t:
            continue
        if t in raw_name_set:
            out.add(t)
        elif "__" in t and t.split("__", 1)[0] in raw_name_set:
            out.add(t.split("__", 1)[0])
    return out


def propose_additive_fusions(
    self,
    *,
    engineered_recipes: dict,
    engineered_continuous: dict,
    newly_engineered_names: list,
    raw_name_set: set,
    cols: list,
    classes_y: np.ndarray,
    X,
    nbins: int,
    seed: int = 0,
    verbose: int = 0,
):
    """Propose ``add(half_a, half_b)`` fusions of disjoint, additively-separable engineered halves.

    Returns ``(admitted, subsumed_names)`` where ``admitted`` is a list of
    ``{"name", "values", "recipe"}`` dicts (each a fused compound to materialise exactly
    like an escalation survivor) and ``subsumed_names`` is the set of fragment engineered
    names the caller must drop from ``selected_vars`` / ``engineered_recipes`` (they are
    now carried by the fused compound). Pure / no live state captured (picklable fit).
    """
    from ._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin
    from ._fe_cmi_redundancy_gate import _conditional_perm_null
    from .engineered_recipes import build_unary_binary_recipe

    if not engineered_recipes or not newly_engineered_names:
        return [], set(), set()

    # Additive-separability margin multiple (getattr keeps the signature stable). The fused
    # ``add`` MI must exceed the STRONGER half by MORE than this multiple of that half's
    # marginal-permutation floor -- a chance-fluctuation scale, so an unrelated pair whose
    # sum carries no genuine joint uplift (fused MI ~= the stronger half +/- noise) is NOT
    # fused, while a genuine second additive term (the weak half DOES lift the joint MI
    # above the chance scale) is. This separates real additive separability from a spurious
    # sum far more robustly than a flat MI-ratio bar -- the weak half's contribution to the
    # COARSELY-BINNED joint MI is small in absolute terms (the dominant half saturates most
    # bins) yet still well above the per-half chance floor.
    _floor_margin = float(getattr(self, "fe_additive_fusion_floor_margin", 1.0))
    _max_fusions = int(getattr(self, "fe_additive_fusion_max", 4))

    # y codes (dense int) the MI primitives score against.
    _y = np.asarray(classes_y).ravel()
    if not np.issubdtype(_y.dtype, np.integer):
        _y = _y.astype(np.int64)
    _, y_dense = np.unique(_y, return_inverse=True)
    y_dense = y_dense.astype(np.int64)
    n_rows = y_dense.shape[0]

    # Candidate halves: engineered features just admitted that have BOTH a replayable
    # recipe AND continuous values on hand (the fused operand is built on continuous
    # values, exactly as the nested-parent replay reconstructs them at transform time).
    halves: list[dict] = []
    for nm in newly_engineered_names:
        rec = engineered_recipes.get(nm)
        vals = engineered_continuous.get(nm)
        if rec is None or vals is None:
            continue
        vals = np.asarray(vals, dtype=np.float64).ravel()
        if vals.shape[0] != n_rows:
            continue
        toks = _bare_tokens(nm, raw_name_set)
        if not toks:
            continue
        vb = _quantile_bin(vals, nbins=int(nbins))
        mi = float(_cmi_from_binned(vb, y_dense, None))
        floor, _ = _conditional_perm_null(vb, y_dense, None, seed=seed)
        if mi <= floor:
            continue  # not relevant -- never a fusion half
        halves.append({"name": nm, "recipe": rec, "vals": vals, "tokens": toks,
                       "mi": mi, "floor": float(floor), "binned": vb})

    if len(halves) < 2:
        return [], set(), set()

    # Strongest half first so the most-informative disjoint pair is fused before the cap.
    halves.sort(key=lambda h: h["mi"], reverse=True)

    admitted: list[dict] = []
    subsumed: set = set()
    subsumed_raws: set = set()
    used: set = set()  # half names already consumed by an admitted fusion
    existing_names = set(engineered_recipes) | {cols[i] for i in range(len(cols))}

    for ia in range(len(halves)):
        if len(admitted) >= _max_fusions:
            break
        ha = halves[ia]
        if ha["name"] in used:
            continue
        for ib in range(ia + 1, len(halves)):
            hb = halves[ib]
            if hb["name"] in used:
                continue
            # DISJOINT raw-token sets -- the two halves cover DIFFERENT signal groups.
            if ha["tokens"] & hb["tokens"]:
                continue
            fused_vals = ha["vals"] + hb["vals"]
            fused_vals = np.nan_to_num(fused_vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            fvb = _quantile_bin(fused_vals, nbins=int(nbins))
            fused_mi = float(_cmi_from_binned(fvb, y_dense, None))
            # GENUINE ADDITIVE SEPARABILITY: the sum must carry strictly MORE target
            # information than the STRONGER half by more than that half's marginal-perm
            # floor (a chance-fluctuation scale). Two unrelated features whose sum has no
            # genuine joint uplift (fused_mi ~= the stronger half +/- noise) are NOT fused.
            _strong = ha if ha["mi"] >= hb["mi"] else hb
            if fused_mi <= _strong["mi"] + _floor_margin * _strong["floor"]:
                continue
            # Build the fused recipe via the EXISTING unary_binary + nested-parent machinery.
            name = f"add({ha['name']},{hb['name']})"
            base = name
            k = 2
            while name in existing_names:
                name = f"{base}_{k}"
                k += 1
            recipe = build_unary_binary_recipe(
                name=name,
                src_a_name=ha["name"], src_b_name=hb["name"],
                unary_a_name="identity", unary_b_name="identity",
                binary_name="add",
                unary_preset=str(getattr(self, "fe_unary_preset", "medium")),
                binary_preset=str(getattr(self, "fe_binary_preset", "minimal")),
                quantization_nbins=self.quantization_nbins,
                quantization_method=self.quantization_method,
                quantization_dtype=self.quantization_dtype,
                fit_values_for_edges=fused_vals,
                nested_parent_a=ha["recipe"],
                nested_parent_b=hb["recipe"],
            )
            admitted.append({"name": name, "values": fused_vals, "recipe": recipe,
                             "mi": fused_mi})
            existing_names.add(name)
            # The two fragments are now subsumed by the fused compound (it carries BOTH
            # additive terms): drop them from selection / the recipe dict.
            subsumed.add(ha["name"])
            subsumed.add(hb["name"])
            used.add(ha["name"])
            used.add(hb["name"])
            # RAW-OPERAND SUBSUMPTION (2026-06-24). A raw operand of either half that the
            # fused compound now FULLY captures must drop too -- otherwise it lingers as a
            # redundant single-group fragment (raw ``a`` beside ``add(a**2/b, log(c)sin(d))``).
            # Reuse the production keep-probe (``raw_retains_signal_given_genuine_children``):
            # condition the raw on the fused compound's bin; if it retains NO significant
            # independent residual, it is subsumed -> drop. A raw that carries a genuine
            # PRIVATE term the compound does not span keeps its residual and is KEPT. This is
            # the same n-invariant verdict the post-fit raw-redundancy sweep applies, realised
            # here (the sweep is gated on ``redundancy_policy='drop'``, but a fused-compound's
            # operand is unconditionally redundant once the compound replaces the fragments).
            from ._fe_raw_redundancy_drop import raw_retains_signal_given_genuine_children
            for _rn in (ha["tokens"] | hb["tokens"]):
                if _rn in subsumed_raws:
                    continue
                _rv = None
                try:
                    if hasattr(X, "columns") and _rn in getattr(X, "columns", []):
                        _rv = np.asarray(X[_rn], dtype=np.float64).ravel()
                except Exception:
                    _rv = None
                if _rv is None or _rv.shape[0] != n_rows:
                    continue
                _rvb = _quantile_bin(np.nan_to_num(_rv, nan=0.0, posinf=0.0, neginf=0.0), nbins=int(nbins))
                _retains = raw_retains_signal_given_genuine_children(
                    raw_bin=_rvb, y_bin=y_dense, genuine_child_bins=[fvb], seed=seed,
                )
                if not _retains:
                    subsumed_raws.add(_rn)
            if verbose:
                logger.info(
                    "MRMR FE additive-fusion: fused %r (mi=%.4f) + %r (mi=%.4f) -> %r "
                    "(mi=%.4f, margin x%.2f); dropping the two subsumed fragments.",
                    ha["name"], ha["mi"], hb["name"], hb["mi"], name, fused_mi, _floor_margin,
                )
            break  # ha consumed; move to the next un-used half

    return admitted, subsumed, subsumed_raws
