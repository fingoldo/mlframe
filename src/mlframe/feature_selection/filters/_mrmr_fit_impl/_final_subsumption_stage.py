"""MRMR glue of the end-of-fit subsumption pass (``_fe_final_subsumption``): runs after every selection stage has finished.

The anchors are the engineered compounds that combine two or more raw operands; every other selected column (raw or single-operand
engineered) is dropped when it carries no information about the target beyond those compounds. Columns without fit-time values are left
alone, since they cannot be judged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype

from ._residual_rescue_stage import _raw_sources

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from ..mrmr import MRMR

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)


def _column_values(X, name):
    """Continuous values of the raw column ``name`` (pandas or polars)."""
    col = X[name]
    return np.asarray(col.to_numpy() if hasattr(col, "to_numpy") else col, dtype=_crit_np_dtype()).ravel()


def _is_numeric_column(X, name):
    """True when the raw column ``name`` (pandas or polars) holds numbers or booleans."""
    import pandas as pd

    dtype = X[name].dtype
    if hasattr(dtype, "is_numeric"):
        if bool(dtype.is_numeric()):
            return True
        return str(dtype) == "Boolean"
    return bool(pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype))


def _replay_continuous(recipe, X):
    """Continuous values of a compound recipe on ``X`` (quantisation stripped), or ``None`` when the replay is not possible."""
    import dataclasses

    from ..engineered_recipes import apply_recipe

    try:
        values = apply_recipe(dataclasses.replace(recipe, quantization=None), X)
    except Exception as exc:  # a compound that cannot be replayed is left out of the pass, as before
        log_throttle(logger, "final_subsumption_not_replayed", logging.WARNING, "final subsumption: %s not replayed (%s: %s)", getattr(recipe, "name", "?"), type(exc).__name__, exc)
        return None
    return np.asarray(values, dtype=_crit_np_dtype()).ravel()


def _compound_halves(recipes, anchor_names, eng_values, X, raw_set):
    """Names of the nested halves of the surviving compounds that are not raw columns; their continuous values join ``eng_values``."""
    halves = []
    by_name = {str(getattr(r, "name", "")): r for r in recipes}
    for name in anchor_names:
        recipe = by_name.get(name)
        for side in ("nested_parent_a", "nested_parent_b"):
            parent = (getattr(recipe, "extra", None) or {}).get(side) if recipe is not None else None
            if parent is None:
                continue
            pname = str(getattr(parent, "name", ""))
            if not pname or pname in raw_set or pname in halves:
                continue
            if pname not in eng_values:
                values = _replay_continuous(parent, X)
                if values is None:
                    continue
                eng_values[pname] = values
            halves.append(pname)
    return halves


def _nesting(recipes, multi, raw_set):
    """``nested``: each multi-source column to the names of its nested parents; ``halves_of``: each to its non-raw halves."""
    by_name = {str(getattr(r, "name", "")): r for r in recipes}
    nested, halves_of = {}, {}
    for name in multi:
        recipe = by_name.get(name)
        extra = (getattr(recipe, "extra", None) or {}) if recipe is not None else {}
        parents = [str(getattr(extra.get(side), "name", "")) for side in ("nested_parent_a", "nested_parent_b") if extra.get(side) is not None]
        nested[name] = set(parents)
        halves_of[name] = [p for p in parents if p and p not in raw_set]
    return nested, halves_of


def _standing_compounds(multi, eng_values, classes_y, y_continuous, seed, nested, halves_of):
    """The multi-source columns that no other standing column subsumes.

    A column is judged against the others still standing, plus the halves of those others. A column never judges itself, and a
    column that is built from it (directly or through a half) is not a judge: a nested compound is a function of the compound it
    was built on, so letting it judge that compound would drop the compound on its own dependent.
    ``nested`` maps a compound to the names of its nested parents; ``halves_of`` maps a compound to its non-raw halves.
    """
    from .._fe_retention_subsumption import retention_form_is_subsumed

    y_binned = np.asarray(classes_y).ravel()
    standing = list(multi)
    for name in multi:
        judges = [o for o in standing if o != name and name not in nested.get(o, ())]
        refs = []
        for o in judges:
            for h in halves_of.get(o, ()):
                if h != name and h not in refs and h not in judges:
                    refs.append(h)
        others = judges + refs
        if not others:
            continue
        if retention_form_is_subsumed(
            cand_continuous=np.asarray(eng_values[name], dtype=_crit_np_dtype()).ravel(),
            incumbent_continuous=[np.asarray(eng_values[o], dtype=_crit_np_dtype()).ravel() for o in others],
            y_binned=y_binned,
            y_continuous=y_continuous,
            seed=seed,
        ):
            standing.remove(name)
    return standing


def _without_linearly_useful(self: "MRMR", dropped: list, candidates: dict, anchors: list, y: Any, standing: "list | tuple" = ()) -> list:
    """``dropped`` minus the columns that still give a linear model a significant held-out gain beyond the anchors.

    Information is not what a linear downstream uses: a target encoding of a category carries no information beyond a compound that recodes the same category, and a linear model still
    takes most of the target variance from it and nothing from the compound. A column is kept when its gain, judged with the bar of a raw operand (``t > z`` and a share of the target
    variance of at least ``fe_residual_rescue_operand_min_linear_gain``), is significant given the anchors, the other columns that stay selected (``standing``) and the columns kept before it; the strongest
    are judged first. Judging against the anchors alone resurrected a column the residual rescue had just pruned against the full design.
    """
    from scipy.stats import norm

    from .._pair_residual_screen import FAMILY_ALPHA
    from .._residual_rescue_fe import MIN_ROWS, linear_oos_gain

    y_arr = np.asarray(y).ravel()
    if not dropped or len(y_arr) < MIN_ROWS:
        return dropped
    z = float(norm.isf(FAMILY_ALPHA))
    floor = float(getattr(self, "fe_residual_rescue_operand_min_linear_gain", 0.0))
    rest = {a: candidates[a] for a in (*anchors, *standing) if a in candidates}
    gains = {}
    for name in dropped:
        if name in candidates and name not in rest:
            gains[name] = linear_oos_gain(rest, candidates[name], y_arr)
    kept: list = []
    for name in sorted(gains, key=lambda n: -gains[n][0]):
        t_stat, rel = linear_oos_gain(rest, candidates[name], y_arr) if kept else gains[name]
        if t_stat > z and rel >= floor:
            kept.append(name)
            rest[name] = candidates[name]
    return [n for n in dropped if n not in kept]


def run_final_subsumption_prune(
    self: "MRMR", X: "pd.DataFrame | pl.DataFrame", selected_vars: list[int], classes_y: np.ndarray, verbose: int | bool
) -> tuple[bool, list[int]]:
    """Drop the selected columns the compounds subsume; returns ``(changed, selected_vars)``."""
    try:
        from .._fe_final_subsumption import final_subsumption_prune

        raw_names = [str(c) for c in self.feature_names_in_]
        raw_set = frozenset(raw_names)
        store = getattr(self, "_engineered_continuous_", None) or {}
        recipes = list(self._engineered_recipes_ or [])
        eng_values = {}
        for recipe in recipes:
            name = str(getattr(recipe, "name", ""))
            if name in store:
                eng_values[name] = np.asarray(store[name], dtype=_crit_np_dtype()).ravel()
            else:
                # Not kept at fit time: replay the column on the raw columns, continuous (quantisation stripped), as its parents are replayed.
                replayed = _replay_continuous(recipe, X)
                if replayed is not None:
                    eng_values[name] = replayed
        raw_selected = [raw_names[int(v)] for v in selected_vars if 0 <= int(v) < len(raw_names)]
        candidates = {}
        for name in raw_selected:
            # a string / categorical column is not a continuous candidate: the compounds cannot subsume it, and converting it raised and cancelled the whole pass
            if _is_numeric_column(X, name):
                candidates[name] = _column_values(X, name)
        candidates.update(eng_values)
        multi = [name for name in eng_values if len(_raw_sources(name, raw_set)) >= 2]
        y_continuous = getattr(self, "_fe_prewarp_y_continuous_", None)
        seed = int(getattr(self, "random_seed", 0) or 0)
        # A multi-source column is a compound only if the other compounds do not already carry its information: a three-operand
        # product that the two-operand compound subsumes must not be kept as an anchor for that reason alone. Each one is judged
        # against the others still standing, so the verdict does not depend on the order of the list.
        # The halves a multi-source compound was built from carry its information too, even when the selection no longer holds them:
        # they are references for the subsumption test (never dropped, not columns of the output).
        # A compound is a function of its own halves, so the halves never judge a compound (they would always subsume it); they only
        # serve as anchors for the other columns.
        halves = _compound_halves(recipes, set(multi), eng_values, X, raw_set)
        nested, halves_of = _nesting(recipes, multi, raw_set)
        anchors = _standing_compounds(multi, eng_values, classes_y, y_continuous, seed, nested, halves_of) + halves
        candidates.update({name: eng_values[name] for name in anchors if name in eng_values})
        if not anchors:
            return False, selected_vars
        dropped = final_subsumption_prune(candidates, anchors, np.asarray(classes_y).ravel(), y_continuous=y_continuous, seed=seed)
        standing = [n for n in candidates if n not in dropped and n not in anchors and (n in raw_selected or n in set(str(f) for f in (self._engineered_features_ or [])))]
        dropped = _without_linearly_useful(self, dropped, candidates, anchors, y_continuous if y_continuous is not None else classes_y, standing)
        if not dropped:
            return False, selected_vars
        dropped_set = set(dropped)
        dropped_raw = dropped_set & set(raw_selected)
        dropped_eng = dropped_set - dropped_raw
        if dropped_raw:
            selected_vars = [v for v in selected_vars if not (0 <= int(v) < len(raw_names) and raw_names[int(v)] in dropped_raw)]
            self._raw_redundancy_dropped_ = set(getattr(self, "_raw_redundancy_dropped_", None) or set()) | dropped_raw
        if dropped_eng:
            self._engineered_recipes_ = [r for r in recipes if str(getattr(r, "name", "")) not in dropped_eng]
            self._engineered_features_ = [f for f in (self._engineered_features_ or []) if str(f) not in dropped_eng]
        from .._fe_rejection_ledger import record_fe_rejection

        for dropped_name in sorted(dropped):
            record_fe_rejection(self, gate="final_subsumption", candidate=dropped_name, operator="raw" if dropped_name in dropped_raw else "engineered", reason="no information beyond the compounds")
        if verbose:
            logger.info("MRMR final subsumption pass: dropped %d column(s) carrying no information beyond the compounds: %s", len(dropped), sorted(dropped))
        self._final_subsumption_dropped_ = sorted(dropped)
        return True, selected_vars
    except Exception:
        logger.warning("MRMR final subsumption pass failed; the selection is left as it was.", exc_info=True)
        return False, selected_vars
