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
    """One standard error, in nats, of the difference between the corrected scores of two sets that carry the same information."""
    return float(np.sqrt(2.0 * max(extra_cells, 0) * max(n_y - 1, 1)) / (2.0 * n))


def choose_compaction(columns, selected, engineered_selected, sources, pool, y_codes, consumers=None):
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
        if new_cells > cells or new_cells > n / _MIN_ROWS_PER_CELL:
            continue
        if score < current - _equivalence_tolerance(cells - new_cells, n_y, n):
            continue
        key = (-score, len(kept) + 1, cand)
        if best is None or key < best[0]:
            best = (key, cand, covered)
    return None if best is None else (best[1], best[2])


def _synthesized_triplet(X, y_codes, operands):
    """``(name, recipe, codes)`` of the cross-basis triplet feature of three raw ``operands``, or ``None`` when none is built.

    The triplet stage is seeded by univariate ranks, which never reach a conjunction whose operands carry no marginal signal; the compaction
    asks for exactly the one triplet its selected features already span.
    """
    from .._orthogonal_triplet_fe import hybrid_orth_mi_triplet_fe_with_recipes
    from .._orthogonal_triplet_fe import _quantile_codes
    from ..engineered_recipes.shared import apply_recipe

    triple = tuple(sorted(operands))
    _, _, _, recipes = hybrid_orth_mi_triplet_fe_with_recipes(X[list(triple)], y_codes, cols=list(triple), top_k=0, explicit_triplets=[triple])
    built = [r for r in recipes if str(getattr(r, "name", "")).split("__", 1)[0].count("*") == 2]
    if not built:
        return None
    rec = built[0]
    values = np.asarray(apply_recipe(rec, X), dtype=np.float64).ravel()
    return str(rec.name), rec, _quantile_codes(values, 10).astype(np.int64)


def run_selection_compaction(self, X, selected_vars, cols, data, classes_y, engineered_recipes, verbose):
    """Apply :func:`choose_compaction` to the finished selection; returns ``(changed, selected_vars)``."""
    try:
        from ._residual_rescue_stage import _raw_sources

        recipes = list(self._engineered_recipes_ or [])
        if len(recipes) < 2 or not engineered_recipes:
            return False, selected_vars
        raw_names = [str(c) for c in self.feature_names_in_]
        raw_set = frozenset(raw_names)
        index = {str(c): i for i, c in enumerate(cols)}
        sel_raw = [raw_names[int(v)] for v in selected_vars if 0 <= int(v) < len(raw_names)]
        sel_eng = [str(getattr(r, "name", "")) for r in recipes]
        names = sel_raw + sel_eng
        if any(nm not in index for nm in names):
            return False, selected_vars
        pool = [nm for nm in engineered_recipes if nm in index and nm not in names]
        wanted = set(names) | set(pool)
        columns = {nm: np.asarray(data[:, index[nm]]) for nm in wanted}
        sources = {nm: _raw_sources(nm, raw_set) for nm in wanted}
        engineered_recipes = dict(engineered_recipes)
        spanned = frozenset().union(*(sources[nm] for nm in sel_eng)) if sel_eng else frozenset()
        if len(spanned) == 3 and not any(sources[nm] >= spanned for nm in pool):
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
        for _ in range(3):
            choice = choose_compaction(columns, sel_raw + sel_eng, sel_eng, sources, [p for p in pool if p not in sel_eng], y_codes, consumers)
            if choice is None:
                break
            cand, replaced = choice
            sel_eng = [*(e for e in sel_eng if e not in replaced), cand]
            changed = True
            self._selection_compaction_ = [*(getattr(self, "_selection_compaction_", None) or []), (cand, tuple(replaced))]
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
