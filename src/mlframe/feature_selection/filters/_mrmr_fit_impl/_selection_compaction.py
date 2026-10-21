"""Parsimony compaction of the final selection: replace several engineered features by one that carries at least as much information.

A forward selection takes the candidate with the largest gain at every step, and two representations of the same structure can differ by
estimation noise alone. On ``y ~ A + A*B*C`` the pair features ``f(A, B)`` and ``f(A, C)`` together with ``A`` determine ``A*B*C`` exactly, and
the composite ``A*B*C`` with ``A`` carries the same information in one feature fewer; which of the two the greedy keeps flips with the
backend that scored the gains. The comparison that does not flip is the bias-corrected joint information of the two SETS: the plug-in joint MI
minus the Miller-Madow term ``(cells - 1) * (y classes - 1) / (2 n)``, which charges every extra occupied cell. A set that is smaller and
scores at least as high, within one standard error of their difference, replaces the larger one (under equivalence the larger table's extra
information is a chi-square variable with ``(extra cells) * (y classes - 1)`` degrees of freedom, so the two corrected scores differ by noise of
size ``sqrt(2 * df) / (2 n)`` and a bare ``>=`` would be a coin flip). Only engineered candidates built from operands that the replaced features already use qualify,
a replacement must save at least one feature, and sets whose joint table is too sparse for the correction to mean anything are left alone.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable, Mapping
from typing import Any

import numpy as np

from .._parity_fe import _mi_codes_njit

logger = logging.getLogger(__name__)

_MIN_ROWS_PER_CELL = 20  # a joint table with more occupied cells than n / this is too sparse for the Miller-Madow correction


def corrected_joint_mi(columns: "list[np.ndarray]", y_codes: np.ndarray) -> "tuple[float, int]":
    """``(bias-corrected joint MI in nats, occupied joint cells)`` of the integer-coded ``columns`` with ``y_codes``."""
    n = int(y_codes.shape[0])
    joint = np.zeros(n, dtype=np.int64)
    for col in columns:
        c = np.asarray(col).astype(np.int64, copy=False)
        base = int(c.max()) + 1 if c.size else 1
        _, joint = np.unique(joint * base + c, return_inverse=True)
        joint = joint.astype(np.int64, copy=False)
    cells = int(joint.max()) + 1 if n else 1
    _, y_dense = np.unique(y_codes, return_inverse=True)
    n_y = int(y_dense.max()) + 1 if n else 1
    mi = float(_mi_codes_njit(joint, cells, y_dense.astype(np.int64).ravel(), n_y))
    return mi - (cells - 1.0) * (n_y - 1.0) / (2.0 * n), cells


def _equivalence_tolerance(extra_cells: int, n_y: int, n: int) -> float:
    """Largest deficit, in nats, of a replacement's corrected score against the set it replaces that still counts as the same information.

    ``extra_cells`` is the replacement's table size minus the set's. Cells the replacement adds that carry nothing cost exactly their Miller-Madow charge, ``extra * (y classes - 1) / (2 n)``,
    and the two plug-in estimates of equal information differ by noise of one standard error, ``sqrt(2 * |extra| * (y classes - 1)) / (2 n)``; a loss of real information is larger than both.
    """
    df = max(n_y - 1, 1)
    return float(max(extra_cells, 0) * df / (2.0 * n) + np.sqrt(2.0 * abs(extra_cells) * df) / (2.0 * n)) + 1e-12  # the floor absorbs summation-order rounding of equal partitions


def choose_compaction(
    columns: Mapping[str, np.ndarray],
    selected: Collection[str],
    engineered_selected: Collection[str],
    sources: Mapping[str, frozenset[str]],
    pool: Iterable[str],
    y_codes: np.ndarray,
    consumers: Mapping[str, set[str]] | None = None,
) -> tuple[str, list[str]] | None:
    """The single replacement that shrinks the selection without losing corrected information, as ``(candidate, replaced)``, or ``None``.

    ``columns`` maps a name to its integer codes; ``selected`` is every selected name (raw and engineered); ``engineered_selected`` the engineered part;
    ``sources`` maps a name to the frozenset of raw operands it is built from; ``pool`` the unselected engineered candidates; ``consumers`` maps a name
    to the names whose recipes read it (a replaced feature that a surviving recipe consumes cannot be dropped).
    """
    n = int(y_codes.shape[0])
    n_y = int(np.unique(y_codes).size)
    consumers = consumers or {}
    if len(engineered_selected) < 2 or any(nm not in columns for nm in selected):
        return None
    current, cells = corrected_joint_mi([columns[nm] for nm in selected], y_codes)
    if cells > n / _MIN_ROWS_PER_CELL:
        return None
    best = None
    for cand in sorted(pool):
        if cand in selected or cand not in columns:
            continue
        covered = [e for e in engineered_selected if sources.get(e) and sources[e] <= sources.get(cand, frozenset())]
        if len(covered) < 2:
            continue
        if any(consumers.get(e, set()) - set(covered) - {cand} for e in covered):
            continue
        kept = [nm for nm in selected if nm not in covered]
        score, new_cells = corrected_joint_mi([columns[nm] for nm in [*kept, cand]], y_codes)
        if new_cells > n / _MIN_ROWS_PER_CELL:
            continue
        if score < current - _equivalence_tolerance(new_cells - cells, n_y, n):
            continue
        key = (-score, len(kept) + 1, cand)
        if best is None or key < best[0]:
            best = (key, cand, covered)
    return None if best is None else (best[1], best[2])


_CELL_MAX_LEVELS = 8  # operands with at most this many distinct values are crossed as a joint cell table instead of a polynomial triplet


def _best_triple(sel_eng, sources) -> "frozenset | None":
    """The three raw columns that the most selected engineered features are built from (at least two of them), or ``None``."""
    import itertools

    spanned = sorted(frozenset().union(*(sources[nm] for nm in sel_eng))) if sel_eng else []
    if not 3 <= len(spanned) <= 8:
        return None
    best, best_cover = None, 1
    for trio in itertools.combinations(spanned, 3):
        t = frozenset(trio)
        cover = sum(1 for nm in sel_eng if sources[nm] and sources[nm] <= t)
        if cover > best_cover:
            best, best_cover = t, cover
    return best


def prune_redundant(
    columns: Mapping[str, np.ndarray],
    selected: Collection[str],
    engineered_selected: Collection[str],
    y_codes: np.ndarray,
    consumers: Mapping[str, Collection[str]] | None = None,
) -> list[str]:
    """Engineered features whose removal does not lower the bias-corrected joint information of the selection beyond its standard error, dropped one at a time (a feature that another selected
    recipe reads is kept). A copy of information the rest of the selection already holds, such as a fragment built on a proxy of a selected column, goes."""
    consumers = consumers or {}
    n = int(y_codes.shape[0])
    n_y = int(np.unique(y_codes).size)
    kept = list(selected)
    dropped = []
    for e in sorted(engineered_selected):
        if e not in kept or any(c in kept for c in consumers.get(e, ()) if c != e):
            continue
        rest = [nm for nm in kept if nm != e]
        if not rest:
            continue
        with_e, cells_with = corrected_joint_mi([columns[nm] for nm in kept], y_codes)
        if cells_with > n / _MIN_ROWS_PER_CELL:
            continue
        without, cells_without = corrected_joint_mi([columns[nm] for nm in rest], y_codes)
        if without >= with_e - _equivalence_tolerance(cells_without - cells_with, n_y, n):
            kept = rest
            dropped.append(e)
    return dropped


def _synthesized_cell_triple(X, operands):
    """``(name, recipe, codes)`` of the joint cell code of three few-level raw ``operands``, or ``None`` when an operand has too many levels.

    The cell table keeps every combination apart, so it holds whatever any function of the three operands (a conjunction included) can carry; the
    polynomial triplet of binary legs collapses to a parity and does not.
    """
    from .._cat_triple_fe import _encode_triple
    from .._target_encoding_fe import _column_to_str
    from ..engineered_recipes import build_cat_triple_cross_recipe
    from ..engineered_recipes.shared import apply_recipe

    triple = tuple(sorted(operands))
    if any(X[c].nunique(dropna=False) > _CELL_MAX_LEVELS for c in triple):
        return None
    cats = [np.asarray(_column_to_str(X[c])) for c in triple]
    _, mapping = _encode_triple(cats[0], cats[1], cats[2])
    name = "cell3(" + ",".join(triple) + ")"
    rec = build_cat_triple_cross_recipe(name=name, cat_a=triple[0], cat_b=triple[1], cat_c=triple[2], mapping=mapping)
    return name, rec, np.asarray(apply_recipe(rec, X)).astype(np.int64)


def _synthesized_triplet(X, y_codes, operands):
    """``(name, recipe, codes)`` of the cross-basis triplet feature of three raw ``operands``, or ``None`` when none is built.

    The triplet stage is seeded by univariate ranks, which never reach a conjunction whose operands carry no marginal signal; the compaction
    asks for exactly the one triplet its selected features already span.
    """
    from .._fe_usability_signal import _crit_np_dtype
    from .._orthogonal_triplet_fe import hybrid_orth_mi_triplet_fe_with_recipes
    from .._orthogonal_triplet_fe import _quantile_codes
    from ..engineered_recipes.shared import apply_recipe

    triple = tuple(sorted(operands))
    _, _, _, recipes = hybrid_orth_mi_triplet_fe_with_recipes(X[list(triple)], y_codes, cols=list(triple), top_k=0, explicit_triplets=[triple])
    built = [r for r in recipes if str(getattr(r, "name", "")).split("__", 1)[0].count("*") == 2]
    if not built:
        return None
    rec = built[0]
    values = np.asarray(apply_recipe(rec, X), dtype=_crit_np_dtype()).ravel()
    return str(rec.name), rec, _quantile_codes(values, 10).astype(np.int64)


def _replayed_codes(recipe, X) -> np.ndarray:
    """Integer codes (ten quantile bins) of the values a recipe produces on ``X``."""
    from .._fe_usability_signal import _crit_np_dtype
    from .._orthogonal_triplet_fe import _quantile_codes
    from ..engineered_recipes.shared import apply_recipe

    values = np.nan_to_num(np.asarray(apply_recipe(recipe, X), dtype=_crit_np_dtype()).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    return _quantile_codes(values, 10).astype(np.int64)


def _recipe_sources(name: str, recipes: dict, raw_set: frozenset, _depth: int = 0) -> frozenset:
    """Raw columns a feature is built from: a raw column is itself, an engineered one the union over the ``src_names`` of its recipe (recursively); a name with no recipe falls back to its
    spelled operands, and a ``dec<col>_<k>`` digit column to its ``<col>``."""
    import re

    from ._residual_rescue_stage import _raw_sources

    if name in raw_set:
        return frozenset({name})
    rec = recipes.get(name)
    if rec is not None and getattr(rec, "src_names", None) and _depth < 8:
        out: frozenset = frozenset()
        for src in rec.src_names:
            out |= _recipe_sources(str(src), recipes, raw_set, _depth + 1)
        if out:
            return out
    m = re.match(r"^dec(.+)_[0-9]+$", name)
    if m and m.group(1) in raw_set:
        return frozenset({m.group(1)})
    return _raw_sources(name, raw_set)


def run_selection_compaction(
    self, X: Any, selected_vars: list[int], cols: list[str], data: np.ndarray, classes_y: np.ndarray, engineered_recipes: dict[str, Any], verbose: bool
) -> tuple[bool, list[int]]:
    """Apply :func:`choose_compaction` to the finished selection; returns ``(changed, selected_vars)``."""
    self._selection_compaction_ = []
    try:
        recipes = list(self._engineered_recipes_ or [])
        if len(recipes) < 2 or not engineered_recipes:
            return False, selected_vars
        raw_names = [str(c) for c in self.feature_names_in_]
        raw_set = frozenset(raw_names)
        index = {str(c): i for i, c in enumerate(cols)}
        sel_raw = [raw_names[int(v)] for v in selected_vars if 0 <= int(v) < len(raw_names)]
        sel_eng = [str(getattr(r, "name", "")) for r in recipes]
        names = sel_raw + sel_eng
        by_recipe = {str(getattr(r, "name", "")): r for r in recipes}
        # A selected feature added after the discretisation (a fused compound, a rescue term) has no column in ``data``: it is replayed from its recipe and coded the same way.
        replayed = {nm: _replayed_codes(by_recipe[nm], X) for nm in sel_eng if nm not in index and nm in by_recipe}
        if any(nm not in index and nm not in replayed for nm in names):
            return False, selected_vars
        pool = [nm for nm in engineered_recipes if nm in index and nm not in names]
        wanted = set(names) | set(pool)
        columns = {nm: (replayed[nm] if nm in replayed else np.asarray(data[:, index[nm]])) for nm in wanted}
        all_recipes = {**{str(k): v for k, v in engineered_recipes.items()}, **by_recipe}
        sources = {nm: _recipe_sources(nm, all_recipes, raw_set) for nm in wanted}
        engineered_recipes = dict(engineered_recipes)
        spanned = _best_triple(sel_eng, sources)
        if spanned is not None and not any(nm.startswith("cell3(") and sources[nm] >= spanned for nm in pool):
            made = _synthesized_cell_triple(X[sorted(spanned)], spanned)
            if made is None:
                made = _synthesized_triplet(X, np.asarray(classes_y).ravel(), spanned)
            if made is not None:
                name, rec, codes = made
                pool.append(name)
                columns[name], sources[name], engineered_recipes[name] = codes, spanned, rec
        consumers: dict = {}
        for r in list(recipes) + [engineered_recipes[nm] for nm in pool]:
            for s in getattr(r, "src_names", ()) or ():
                consumers.setdefault(str(s), set()).add(str(getattr(r, "name", "")))
        y_codes = np.asarray(classes_y).ravel()
        changed = False
        # Features that add nothing to the rest go first: a proxy fragment would otherwise keep the replacement candidate from matching the set it replaces.
        # A hinge leg is admitted on its held-out linear usability, and a missingness indicator ties with the NaN bin of its source; binned joint information sees neither (a monotone leg of a selected source adds none by the
        # data-processing inequality), so the information-based compaction must not drop or replace it.
        protected = {c for attr in ("_hinge_features_", "missingness_indicator_features_", "missingness_pattern_features_") for c in (getattr(self, attr, None) or [])}
        dropped = prune_redundant(columns, sel_raw + sel_eng, [e for e in sel_eng if e not in protected], y_codes, consumers)
        if dropped:
            sel_eng = [e for e in sel_eng if e not in dropped]
            changed = True
            self._selection_compaction_ = [*self._selection_compaction_, *((None, (d,)) for d in dropped)]
        for _ in range(3):
            choice = choose_compaction(columns, sel_raw + sel_eng, [e for e in sel_eng if e not in protected], sources, [p for p in pool if p not in sel_eng], y_codes, consumers)
            if choice is None:
                break
            cand, replaced = choice
            sel_eng = [*(e for e in sel_eng if e not in replaced), cand]
            changed = True
            self._selection_compaction_ = [*self._selection_compaction_, (cand, tuple(replaced))]
            if verbose:
                logger.info("MRMR selection compaction: %s replaces %s (at least the same bias-corrected joint information, fewer features).", cand, replaced)
        if not changed:
            return False, selected_vars
        by_name = {str(getattr(r, "name", "")): r for r in recipes}
        by_name.update({nm: engineered_recipes[nm] for nm in sel_eng if nm in engineered_recipes})
        self._engineered_recipes_ = [by_name[nm] for nm in sel_eng if nm in by_name]
        self._engineered_features_ = [nm for nm in sel_eng if nm in by_name]
        return True, selected_vars
    except Exception:
        logger.warning("MRMR selection compaction failed; the selection is left as it was.", exc_info=True)
        return False, selected_vars
