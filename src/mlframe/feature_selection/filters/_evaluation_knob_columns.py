"""Candidate, target and selected-set column materialisation shared by the research knobs of ``evaluation.evaluate_candidate`` (carved out of ``evaluation`` to keep it under 1000 lines)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _materialize_var(factors_data, var_idx, factors_nbins, dtype=np.int32):
    """Late-bound ``evaluation._materialize_var`` (imported at call time: ``evaluation`` imports this module at load)."""
    from .evaluation import _materialize_var as impl

    return impl(factors_data, var_idx, factors_nbins, dtype=dtype)


def _materialise_knob_columns(
    *,
    factors_data: np.ndarray,
    X: Any,
    y: Any,
    factors_nbins: np.ndarray,
    dtype: Any,
    selected_vars: Any,
    relax_y_col: Optional[np.ndarray],
    relax_k_y: Optional[int],
    relax_sel_cols: Optional[list],
    relax_sel_nbins: Optional[list],
) -> dict:
    """Materialise the candidate, target and selected-set columns the research knobs share.

    Carved out of ``evaluate_candidate`` to keep it under its length ceiling. RelaxMRMR, PID, the CMI-permutation stop and CPT all want
    the same three things, and each block used to materialise all of them for itself: with two knobs on, the same candidate was factorised
    twice in one call, and the target and selected set, fixed for the whole greedy round, were rebuilt per candidate. The round's hoist is
    used when the driver supplied it.

    Args:
        factors_data: the binned design the columns are materialised from.
        X: the candidate variable.
        y: the target variable.
        factors_nbins: per-column bin counts.
        dtype: the dtype the materialised columns are cast to.
        selected_vars: the round's already-selected variables.
        relax_y_col: the hoisted target column, or None.
        relax_k_y: the hoisted target bin count, or None.
        relax_sel_cols: the hoisted selected-set columns, or None.
        relax_sel_nbins: the hoisted selected-set bin counts, or None.

    Returns:
        A mapping with ``"x"``, ``"y"``, ``"sel"`` and ``"hoisted"``; ``hoisted`` says whether the round-level hoist was used, which the
        caller needs because the hoisted path has already range-checked the target and selected set for this round.
    """
    out: dict = {}
    out["x"] = _materialize_var(factors_data, X, factors_nbins, dtype=dtype)
    if relax_y_col is not None and relax_k_y is not None and relax_sel_cols is not None and relax_sel_nbins is not None:
        out["y"] = (relax_y_col, relax_k_y)
        out["sel"] = (relax_sel_cols, relax_sel_nbins)
        out["hoisted"] = True
        return out
    out["y"] = _materialize_var(factors_data, y, factors_nbins, dtype=dtype)
    cols, nbins = [], []
    for sv in selected_vars:
        svc, svk = _materialize_var(factors_data, sv, factors_nbins, dtype=dtype)
        cols.append(svc)
        nbins.append(svk)
    out["sel"] = (cols, nbins)
    out["hoisted"] = False
    return out
