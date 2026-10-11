"""MRMR glue of the residual rescue (``_residual_rescue_fe``): builds the selected design and the candidate pool from the fitted frame and attaches what the rescue admits.

A candidate built from operands that a single design column already holds is a refinement of a term the selection already has (a second view of the operands of a fused compound), not new
information, even when it lowers the residual a little; it is not eligible. A joint feature of operands the design holds only in separate columns is eligible.

Design = the selected raw columns plus the selected engineered features (replayed from their recipes). Pool = every other numeric column of the fit frame: the unselected raws and the engineered
columns the FE stages appended but the greedy did not keep. An admitted engineered column is attached through its own recipe (``_engineered_recipes_`` / ``_engineered_features_``), an admitted
raw through ``selected_vars`` / ``support_``. Regression targets only (the continuous target is ``_fe_prewarp_y_continuous_``).
"""

from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)
_TOKEN_SPLIT = re.compile(r"[(),|\s]+")

__all__ = ["run_residual_rescue"]


def _raw_sources(name: str, raw_cols: "frozenset[str]") -> "frozenset[str]":
    """Raw input columns a column is built from. Engineered names spell their operands (``div(sqr(a),abs(b))``, ``a__haar_j1k0``, ``binagg_mean(b|qbin(d))``), and a fused compound's recipe lists
    engineered operands that no longer exist as recipes, so the name is the one reliable record of the sources."""
    if name in raw_cols:
        return frozenset({name})
    tokens = _TOKEN_SPLIT.split(name)
    return frozenset(tok.split("__")[0] for tok in tokens if tok.split("__")[0] in raw_cols)


def _is_refinement(sources: "frozenset[str]", design_sources: "list[frozenset[str]]") -> bool:
    """True when a candidate is built from operands that one design column already holds: a second view of the same operands, beside a fused compound or a raw. A joint feature of operands that
    the design only holds in separate columns (a raw ``a`` and a raw ``b`` beside a candidate ``mul(a,b)``) is new information and not a refinement."""
    return any(sources <= held for held in design_sources)


def _cell_table_candidates(self, X, y, design, raw_cols, design_sources, pool) -> dict:
    """Out-of-fold 2-D cell tables of the design residual for the raw pairs that interact beyond the additive model (the pair residual screen) and that no single design column holds; added to ``pool`` as ``oofcell(a,b)``.

    FE builds a joint feature of a pair from its own unary / binary vocabulary and may build nothing for a pair whose interaction is weak in rank space; a table has no vocabulary to miss. Returns the
    recipe of each added candidate (attached only if the rescue admits it). Empty when the cell operator is off."""
    if not getattr(self, "fe_oof_cell2d_enable", False):
        return {}
    from .._oof_cell2d_fe import MAX_PAIRS, build_oof_cell2d_recipe, fit_oof_cell2d
    from .._pair_residual_screen import pair_interaction_screen
    from .._residual_rescue_fe import design_residual

    numeric = [c for c in X.columns if str(c) in raw_cols and np.issubdtype(X[c].dtype, np.number)]
    recipes: dict = {}
    current = dict(design)
    for _ in range(MAX_PAIRS):
        # one pair per round, each judged on the residual of the design plus the tables already built: a pair interaction is invisible to the screen while the stronger ones are still in the residual
        resid = design_residual(current, y)  # the tables are fitted on what the selection leaves, not on the target: the dominant terms would drown a weak pair interaction in cell noise
        screened = [
            p
            for p in pair_interaction_screen(X, resid, numeric, scan_rows=len(y))
            if not _is_refinement(frozenset({str(p[0]), str(p[1])}), design_sources) and f"oofcell({p[0]},{p[1]})" not in recipes
        ]
        if not screened:
            break
        ca, cb, _t = screened[0]
        a = np.asarray(X[ca].to_numpy(), dtype=np.float64)
        b = np.asarray(X[cb].to_numpy(), dtype=np.float64)
        fit = fit_oof_cell2d(a, b, resid)
        name = f"oofcell({ca},{cb})"
        pool[name] = fit["oof"]
        current[name] = fit["oof"]
        recipes[name] = build_oof_cell2d_recipe(
            name=name, src=(str(ca), str(cb)), ea=fit["ea"], eb=fit["eb"], tab=fit["tab"], fill=fit["fill"], lo=float(fit["oof"].min()), hi=float(fit["oof"].max())
        )
    return recipes


def _unsupported_columns(self, design: dict, y, raw_cols, residual_rescue) -> "list[str]":
    """Design columns that add nothing to the rest of the selection (``redundancy_policy='drop'`` only).

    Backward elimination on held-out predictive error, one criterion per kind of column. An engineered column, or a raw no engineered column is built from, is pruned when, given every other column
    of the design, it does not lower the held-out squared residual significantly (free-form effects): an engineered column fitted to noise (``sub(sqrt(c),sqrt(d))`` on two unrelated columns) has
    none. A raw OPERAND of a selected engineered column is pruned (``fe_drop_redundant_raw_operands``) unless it improves the held-out fit of a LINEAR model on the other columns significantly and by
    at least ``fe_residual_rescue_operand_min_linear_gain`` of the target variance: the engineered column already carries the operand's nonlinear effect, so what the operand can still add is a
    private linear term (``10*a`` beside ``add(exp(a), ...)``), which only the linear fit sees. Pruned columns leave the design before the next one is judged."""
    if getattr(self, "redundancy_policy", "emit_both") != "drop":
        return []
    operands: "frozenset[str]" = frozenset()  # raws held by a selected MULTI-source engineered column
    unary_only: "frozenset[str]" = frozenset()  # raws that only unary transforms of themselves are built from: ``a`` beside ``sin(a)`` is left alone
    for name in design:
        if name not in raw_cols:
            src = _raw_sources(name, raw_cols)
            if len(src) >= 2:
                operands |= src
            else:
                unary_only |= src
    from scipy.stats import norm

    from .._pair_residual_screen import FAMILY_ALPHA
    from .._residual_rescue_fe import linear_oos_gain

    z = float(norm.isf(FAMILY_ALPHA))
    floor = float(self.fe_residual_rescue_operand_min_linear_gain)
    prune_operands = bool(getattr(self, "fe_drop_redundant_raw_operands", True))
    pruned: "list[str]" = []
    for name in list(design):
        rest = {k: v for k, v in design.items() if k != name and k not in pruned}
        if not rest:
            continue
        if name in raw_cols and name in unary_only and name not in operands:
            continue
        if name in raw_cols and name in operands:
            if prune_operands:
                t_stat, rel = linear_oos_gain(rest, design[name], y)
                if not (t_stat > z and rel >= floor):
                    pruned.append(name)
        elif not residual_rescue(rest, {name: design[name]}, y, min_relative_gain=0.0, max_admit=1):
            pruned.append(name)
    return pruned


def run_residual_rescue(self, X, selected_vars, cols, engineered_recipes, retention_added_eng_names, verbose) -> bool:
    """Admit the pool columns that reduce the held-out residual of the selected design and prune the redundant raws (``redundancy_policy='drop'``); True when the selection changed."""
    if not getattr(self, "fe_residual_rescue_enable", False):
        return False
    y_cont = getattr(self, "_fe_prewarp_y_continuous_", None)
    if y_cont is None or not hasattr(X, "columns") or not hasattr(X, "iloc"):
        return False
    try:
        from .._residual_rescue_fe import DEFAULT_SCAN_ROWS, residual_rescue
        from ..engineered_recipes.shared import apply_recipe

        y = np.asarray(y_cont, dtype=np.float64).ravel()
        if len(y) != len(X):
            return False
        n = len(y)
        if n > DEFAULT_SCAN_ROWS:  # the scan rows are taken before any column is converted: no full-length float64 copy of the frame
            rows = np.linspace(0, n - 1, DEFAULT_SCAN_ROWS).astype(np.int64)
            X, y = X.iloc[rows], y[rows]
        sel_names = {str(cols[i]) for i in selected_vars}
        eng_names = [str(n) for n in (self._engineered_features_ or [])]
        design = {}
        for name in sel_names:
            if name in X.columns and name not in engineered_recipes and np.issubdtype(X[name].dtype, np.number):
                design[name] = np.asarray(X[name].to_numpy(), dtype=np.float64)
        for name, rec in zip(eng_names, self._engineered_recipes_ or []):
            design[name] = np.nan_to_num(np.asarray(apply_recipe(rec, X), dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        raw_cols = frozenset(str(c) for c in (getattr(self, "feature_names_in_", None) if getattr(self, "feature_names_in_", None) is not None else cols) if str(c) not in engineered_recipes)
        # prune first: a column without held-out gain must not shield the raw operands it holds from the pool (the sources of a pruned carrier are free again)
        prune_names = _unsupported_columns(self, design, y, raw_cols, residual_rescue)
        for name in prune_names:
            del design[name]
        design_sources = [_raw_sources(name, raw_cols) for name in design]
        pool = {
            str(c): np.asarray(X[c].to_numpy(), dtype=np.float64)
            for c in X.columns
            if str(c) not in design
            and str(c) not in sel_names
            and str(c) not in prune_names
            and np.issubdtype(X[c].dtype, np.number)
            and not _is_refinement(_raw_sources(str(c), raw_cols), design_sources)
        }
        cell_recipes = _cell_table_candidates(self, X, y, design, raw_cols, design_sources, pool)
        admitted = residual_rescue(
            design, pool, y, min_relative_gain=float(self.fe_residual_rescue_min_relative_gain), max_admit=int(self.fe_residual_rescue_max_admit)
        )
        eng_set = {str(n) for n in eng_names}
    except Exception:
        logger.warning("MRMR residual rescue failed; the selection is left as it was.", exc_info=True)
        return False
    names = []
    col_index = {str(c): i for i, c in enumerate(cols)}
    for name, t_stat, rel in admitted:
        rec = engineered_recipes.get(name) or cell_recipes.get(name)
        if rec is not None:
            self._engineered_recipes_.append(rec)
            self._engineered_features_.append(name)
            retention_added_eng_names.add(name)
        elif name in col_index and col_index[name] not in selected_vars:
            selected_vars.append(col_index[name])
        else:
            continue
        names.append(name)
        if verbose:
            logger.info("MRMR residual rescue: admitted %r (t=%.1f, %.1f%% of the target variance).", name, t_stat, 100.0 * rel)
    dropped = [col_index[n] for n in prune_names if n in col_index and col_index[n] in selected_vars and n not in eng_set]
    if dropped and not [i for i in selected_vars if i not in dropped] and not [f for f in self._engineered_features_ if str(f) not in prune_names]:
        # a selection never ends up with no column at all: the pruned raw that tracks the target best stays
        corr = {i: abs(float(np.corrcoef(np.asarray(X[cols[i]].to_numpy(), dtype=np.float64), y)[0, 1])) for i in dropped if cols[i] in X.columns}
        corr = {i: (c if np.isfinite(c) else 0.0) for i, c in corr.items()}
        if corr:
            dropped.remove(max(corr, key=corr.get))
    for idx in dropped:
        selected_vars.remove(idx)
        if verbose:
            logger.info("MRMR residual rescue: pruned raw column %r (no held-out gain given the rest of the selection).", cols[idx])
    pruned_eng = [n for n in prune_names if n in eng_set]
    if pruned_eng:
        keep = [(f, r) for f, r in zip(self._engineered_features_, self._engineered_recipes_) if str(f) not in pruned_eng]
        self._engineered_features_[:] = [f for f, _ in keep]
        self._engineered_recipes_[:] = [r for _, r in keep]
        if verbose:
            logger.info("MRMR residual rescue: pruned engineered column(s) %s (no held-out gain given the rest of the selection).", pruned_eng)
    if names or dropped:
        self.support_ = np.array(selected_vars, dtype=np.int64)
    return bool(names or dropped or pruned_eng)
