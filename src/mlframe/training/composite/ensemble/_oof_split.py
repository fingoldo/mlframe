"""OOF inner-eval carve + split-row helpers for the cross-target ensemble.

Carved out of ``ensemble/__init__.py`` to keep that facade under the 1k-line
budget. Re-exported from the package ``__init__`` for back-compat (tests and
the ensemble body import these names from ``...composite.ensemble``).
"""
from __future__ import annotations

import numpy as np

try:
    import polars as pl  # type: ignore
    _HAS_POLARS = True
except Exception:  # pragma: no cover
    pl = None  # type: ignore
    _HAS_POLARS = False


def _slice_rows(X, mask: np.ndarray):
    """Index rows of X (pandas / polars / ndarray) by a boolean mask."""
    if hasattr(X, "iloc"):
        return X.iloc[mask].reset_index(drop=True) if hasattr(X.iloc[mask], "reset_index") else X.iloc[mask]
    if hasattr(X, "filter") and hasattr(X, "slice"):
        return X.filter(pl.Series(mask))
    return X[mask]


def _align_fit_sw(sample_weight, fit_mask, n_fit):
    """Slice ``sample_weight`` to match the fit rows a carve returned.

    The group-aware carve returns mask-SCATTERED fit rows, so the old
    ``sample_weight[:n_fit]`` prefix slice mis-aligned every weight with the
    wrong row (silent: the lengths matched). ``fit_mask`` is the boolean mask
    for the group path (apply it) and None for the prefix/no-split paths
    (a prefix slice is then exactly correct)."""
    if sample_weight is None:
        return None
    if fit_mask is not None:
        return np.asarray(sample_weight)[fit_mask]
    return np.asarray(sample_weight)[:n_fit]


def _carve_inner_eval_split(
    X, y, *, frac: float = 0.1, random_state: int | None = 0,
    group_ids: np.ndarray | None = None, return_fit_mask: bool = False,
):
    """Return ``(X_fit, y_fit, X_eval, y_eval)`` for OOF refits that need
    an eval_set to satisfy early-stopping callbacks on cloned boosters.

    When ``group_ids`` is supplied, carves whole groups into the eval
    slice (no group spans both fit and eval). Required for honest OOF on
    group-aware splits: rows from the same group/user/session in both fit
    and eval make early-stopping see same-group leakage, model
    under-stops, OOF RMSE artificially degrades (observed in prod:
    val_RMSE 10.64 from direct fit vs honest-OOF 13.34 from group-blind
    carve, +25% degradation that wrongly triggered the AR1 failsafe).

    Without ``group_ids`` falls back to the deterministic last-``frac``
    tail split (mirrors val_placement='forward' for temporal splits).
    For row counts below 1000 the split is skipped (returns
    ``X, y, None, None``) - early-stopping at that scale is noise.

    ``return_fit_mask=True`` appends a 5th element: the boolean fit-row mask
    for the group path (so the caller can align sample_weight via
    :func:`_align_fit_sw`) or None for the prefix / no-split paths. The
    4-tuple default keeps the legacy contract for existing callers."""
    def _ret(x_fit, y_fit, x_ev, y_ev, fit_mask):
        if return_fit_mask:
            return x_fit, y_fit, x_ev, y_ev, fit_mask
        return x_fit, y_fit, x_ev, y_ev
    try:
        n = len(y)
    except TypeError:
        return _ret(X, y, None, None, None)
    if n < 1000:
        return _ret(X, y, None, None, None)
    n_eval_target = max(100, int(frac * n))
    if n_eval_target >= n - 100:
        return _ret(X, y, None, None, None)
    if group_ids is not None:
        g = np.asarray(group_ids)
        if g.shape[0] == n:
            uniq, first_idx = np.unique(g, return_index=True)
            if uniq.size >= 4:
                order = np.argsort(first_idx)
                groups_in_order = uniq[order]
                rng = np.random.default_rng(random_state)
                shuffled = rng.permutation(groups_in_order)
                _, _, counts_orig = np.unique(g, return_index=True, return_counts=True)
                idx_for_group = {gid: i for i, gid in enumerate(uniq)}
                cumulative = 0
                eval_groups: list = []
                for gid in shuffled:
                    eval_groups.append(gid)
                    cumulative += int(counts_orig[idx_for_group[gid]])
                    if cumulative >= n_eval_target:
                        break
                if 0 < cumulative < n - 100 and len(eval_groups) < uniq.size:
                    eval_set = set(eval_groups.tolist() if hasattr(eval_groups, "tolist") else list(eval_groups))
                    eval_mask = np.isin(g, list(eval_set))
                    fit_mask = ~eval_mask
                    return _ret(_slice_rows(X, fit_mask), y[fit_mask], _slice_rows(X, eval_mask), y[eval_mask], fit_mask)
    cut = n - n_eval_target
    if hasattr(X, "iloc"):
        return _ret(X.iloc[:cut], y[:cut], X.iloc[cut:], y[cut:], None)
    if hasattr(X, "select") and hasattr(X, "slice"):
        return _ret(X.slice(0, cut), y[:cut], X.slice(cut, n_eval_target), y[cut:], None)
    return _ret(X[:cut], y[:cut], X[cut:], y[cut:], None)
