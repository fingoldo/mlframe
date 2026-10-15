"""MRMR glue of the residual rescue (``_residual_rescue_fe``): builds the selected design and the candidate pool from the fitted frame and attaches what the rescue admits.

A candidate built from operands that a single design column already holds is a refinement of a term the selection already has (a second view of the operands of a fused compound), not new
information, even when it lowers the residual a little; it is not eligible. A joint feature of operands the design holds only in separate columns is eligible.

Design = the selected raw columns plus the selected engineered features (replayed from their recipes). Pool = every other numeric column of the fit frame: the unselected raws and the engineered
columns the FE stages appended but the greedy did not keep. An admitted engineered column is attached through its own recipe (``_engineered_recipes_`` / ``_engineered_features_``), an admitted
raw through ``selected_vars`` / ``support_``. Regression targets only (the continuous target is ``_fe_prewarp_y_continuous_``).
"""

from __future__ import annotations

from typing import Any

import logging
import re

import numpy as np

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)
_TOKEN_SPLIT = re.compile(r"[(),|\s]+|__")

__all__ = ["run_residual_rescue"]


def _raw_sources(name: str, raw_cols: "frozenset[str]") -> "frozenset[str]":
    """Raw input columns a column is built from. Engineered names spell their operands (``div(sqr(a),abs(b))``, ``a__haar_j1k0``, ``binagg_mean(b|qbin(d))``), and a fused compound's recipe lists
    engineered operands that no longer exist as recipes, so the name is the one reliable record of the sources."""
    if name in raw_cols:
        return frozenset({name})
    return frozenset(tok for tok in _TOKEN_SPLIT.split(name) if tok in raw_cols)


def _is_refinement(sources: "frozenset[str]", design_sources: "list[frozenset[str]]") -> bool:
    """True when a candidate is built from operands that one design column already holds: a second view of the same operands, beside a fused compound or a raw. A joint feature of operands that
    the design only holds in separate columns (a raw ``a`` and a raw ``b`` beside a candidate ``mul(a,b)``) is new information and not a refinement."""
    return bool(sources) and any(sources <= held for held in design_sources)


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


def _target_mix_candidates(self, X, y, design, raw_cols, classify, pool, linear_only) -> dict:
    """Engineered candidates of the feature-base target mix (``_target_mix_fe``), added to ``pool``; returns their recipes (attached only if the rescue admits them). Empty when the mix is off."""
    self.fe_target_mix_provenance_ = []
    if not getattr(self, "fe_target_mix_enable", False):
        return {}
    from ..engineered_recipes.shared import apply_recipe
    from .._target_mix_fe import second_pass_recipes

    found, self.fe_target_mix_provenance_ = second_pass_recipes(
        self,
        X,
        y,
        design,
        max_bases=int(self.fe_target_mix_max_bases),
        scan_rows=int(self.fe_target_mix_scan_rows),
        min_relative_gain=float(self.fe_residual_rescue_min_relative_gain),
    )
    eng_recipe = {str(n): r for n, r in zip(self._engineered_features_ or [], self._engineered_recipes_ or [])}
    for entry in self.fe_target_mix_provenance_:
        base = entry["base"]
        entry["recipe"] = eng_recipe.get(base, {"raw": base} if base in X.columns else None)
    recipes: dict = {}
    for name, recipe in found.items():
        kind = classify(_raw_sources(name, raw_cols))
        if name in design or name in pool or name in X.columns or kind == "skip":
            continue
        try:
            pool[name] = np.nan_to_num(np.asarray(apply_recipe(recipe, X), dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            log_throttle(logger, "residual_rescue_stage.target_mix_replay_failed", logging.WARNING, "MRMR target-mix candidate %r failed to replay; skipped.", name, exc_info=True)
            continue
        recipes[name] = recipe
        if kind == "linear":
            linear_only.add(name)
    return recipes


def _unsupported_columns(self, design: dict, y, raw_cols, residual_rescue) -> "list[str]":
    """Design columns that add nothing to the rest of the selection (``redundancy_policy='drop'`` only).

    Backward elimination on held-out predictive error, one criterion per kind of column. An engineered column, or a raw no engineered column is built from, is pruned when, given every other column
    of the design, it does not lower the held-out squared residual at all (free-form effects, t <= 0): an engineered column fitted to noise (``sub(sqrt(c),sqrt(d))`` on two unrelated columns)
    does not. The bar is deliberately not significance: on a small sample or a weak signal no single column is significant, and removing the weak useful ones would lower the out-of-sample fit. A raw OPERAND of a selected engineered column is pruned (``fe_drop_redundant_raw_operands``) unless it improves the held-out fit of a LINEAR model on the other columns significantly and by
    at least ``fe_residual_rescue_operand_min_linear_gain`` of the target variance: the engineered column already carries the operand's nonlinear effect, so what the operand can still add is a
    private linear term (``10*a`` beside ``add(exp(a), ...)``), which only the linear fit sees. Pruned columns leave the design before the next one is judged."""
    from .._residual_rescue_fe import MIN_ROWS

    if getattr(self, "redundancy_policy", "emit_both") != "drop" or len(y) < MIN_ROWS:  # too few rows: no evidence is not evidence of no gain
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
        elif not residual_rescue(rest, {name: design[name]}, y, alpha=0.5, min_relative_gain=0.0, max_admit=1):
            pruned.append(name)  # alpha = 0.5 makes the bar t > 0: the column is removed only when its held-out gain is not even positive
    return pruned


def _is_group_leak_raw(X, name, y):
    """Lazy bridge to the retention-pass group-leak test (import deferred to keep this module's import order unchanged)."""
    from mlframe.feature_selection.filters._fe_pure_form_retention import _is_group_leak_raw as _impl

    return _impl(X, name, y)


def run_residual_rescue(self, X: Any, selected_vars: Any, cols: Any, engineered_recipes: Any, retention_added_eng_names: Any, verbose: Any) -> bool:
    """Admit the pool columns that reduce the held-out residual of the selected design and prune the redundant raws (``redundancy_policy='drop'``); True when the selection changed."""
    if not getattr(self, "fe_residual_rescue_enable", False):
        return False
    y_cont = getattr(self, "_fe_prewarp_y_continuous_", None)
    if y_cont is None or not hasattr(X, "columns") or not hasattr(X, "iloc"):
        return False
    try:
        from .._residual_rescue_fe import DEFAULT_SCAN_ROWS, residual_rescue
        from .._y_encoding import FEW_CLASSES_MAX
        from ..engineered_recipes.shared import apply_recipe

        y = np.asarray(y_cont, dtype=np.float64).ravel()
        if len(y) != len(X) or np.unique(y).size <= FEW_CLASSES_MAX:  # class labels are not a regression target
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
        _names = getattr(self, "feature_names_in_", None)
        _cols_src = _names if _names is not None else cols
        raw_cols = frozenset(str(c) for c in _cols_src if str(c) not in engineered_recipes)
        # prune first: a column without held-out gain must not shield the raw operands it holds from the pool (the sources of a pruned carrier are free again)
        prune_names = _unsupported_columns(self, design, y, raw_cols, residual_rescue)
        for name in prune_names:
            del design[name]
        design_sources = [_raw_sources(name, raw_cols) for name in design]
        engineered_held = [s for name, s in zip(design, design_sources) if name not in raw_cols and len(s) >= 2]  # compounds of several operands
        raw_held = [s for name, s in zip(design, design_sources) if name in raw_cols or len(s) < 2]  # a raw, or a one-column transform of a raw (hinge leg, warp)

        def classify(sources):
            """'skip': a view of the operands of one multi-source ENGINEERED design column (a refinement of a term the selection has); 'linear': a form of operands one RAW (or one-column
            engineered) design column holds: no free-form gain is possible, a linear model can still gain; 'free': anything else."""
            if _is_refinement(sources, engineered_held):
                # A view of the operands of a multi-source compound is scored by the linear channel only: a free-form gain is impossible there, a linear gain is not.
                # The rejected alternative (skip such views entirely) was never the production arm and is no longer reachable from configuration.
                return "linear"
            return "linear" if sources and _is_refinement(sources, raw_held) else "free"

        # A raw operand the redundancy sweep already dropped as subsumed by a surviving compound is not re-admitted here:
        # the rescue's linear gain for it is the same information the compound carries, not a new term.
        redundancy_dropped = set(getattr(self, "_raw_redundancy_dropped_", None) or set())
        pool = {
            str(c): np.asarray(X[c].to_numpy(), dtype=np.float64)
            for c in X.columns
            if str(c) not in design
            and str(c) not in sel_names
            and str(c) not in prune_names
            and str(c) not in redundancy_dropped
            and np.issubdtype(X[c].dtype, np.number)
            and classify(_raw_sources(str(c), raw_cols)) != "skip"
        }
        linear_only = {name for name in pool if classify(_raw_sources(name, raw_cols)) == "linear"}
        cell_recipes = _cell_table_candidates(self, X, y, design, raw_cols, design_sources, pool)
        cell_recipes.update(_target_mix_candidates(self, X, y, design, raw_cols, classify, pool, linear_only))
        admitted = residual_rescue(
            design,
            pool,
            y,
            min_relative_gain=float(self.fe_residual_rescue_min_relative_gain),
            max_admit=int(self.fe_residual_rescue_max_admit),
            linear_channel=True,
            linear_only=frozenset(linear_only),
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
            # A between-group leak raw column is not re-admitted by the global residual test under group_aware_mi.
            if _is_group_leak_raw(X, name, np.asarray(y).ravel()):
                continue
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
            dropped.remove(max(corr, key=lambda k: corr[k]))
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
