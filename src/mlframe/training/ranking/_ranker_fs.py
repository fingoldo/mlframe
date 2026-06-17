"""LTR-local feature selection for the ranker suite.

Kept deliberately SEPARATE from the core MRMR procedures (which are highly optimised and shared by every other
target type): this module only ORCHESTRATES a standard ``MRMR.fit`` on the (features, graded-relevance) pair and,
optionally, a small self-contained group-aware relevance pre-filter. Nothing here modifies the core MI / binning /
redundancy kernels. Used only on the LEARNING_TO_RANK path when ``LearningToRankConfig.feature_selection=True``.

Two relevance notions:
  * pointwise (default): MRMR's standard MI(feature, relevance) over all rows -- groups handled at the split level.
  * group-aware (``fs_group_aware_mi=True``): a per-query binned MI averaged across queries (computed HERE, not in
    the core), used as a relevance pre-filter; the surviving columns then go through the standard MRMR pass for
    redundancy removal. This gives a query-respecting relevance signal without touching the group-naive core MI.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_LTR_MRMR_KWARGS = {
    "verbose": 0,
    "use_simple_mode": True,   # raw-column selection (no engineered tail) so the ranker trains on real columns
    "max_runtime_mins": 1,
    "n_workers": 1,
}


def _to_numeric_columns(X) -> tuple[list[str], np.ndarray]:
    """Return (numeric_column_names, 2-D float array) for a pandas/polars frame; non-numeric columns are skipped."""
    try:
        import polars as pl
        _is_pl = isinstance(X, pl.DataFrame)
    except ImportError:
        _is_pl = False
    if _is_pl:
        num_cols = [c for c, dt in zip(X.columns, X.dtypes) if dt.is_numeric()]
        arr = X.select(num_cols).to_numpy().astype(np.float64) if num_cols else np.empty((X.height, 0))
        return num_cols, arr
    import pandas as pd
    if isinstance(X, pd.DataFrame):
        num = X.select_dtypes(include=[np.number])
        return list(num.columns), num.to_numpy(dtype=np.float64)
    arr = np.asarray(X, dtype=np.float64)
    return [str(i) for i in range(arr.shape[1])], arr


def _binned_mi(x: np.ndarray, y: np.ndarray, bins: int = 8) -> float:
    """Plug-in MI between two 1-D arrays via quantile binning + a joint histogram (nats). Self-contained."""
    n = x.shape[0]
    if n < 4:
        return 0.0
    xf = x[np.isfinite(x)]
    if xf.size < 2 or np.ptp(xf) == 0.0:
        return 0.0
    xe = np.unique(np.quantile(xf, np.linspace(0, 1, bins + 1)))
    ye_src = y[np.isfinite(y)]
    if ye_src.size < 2 or np.ptp(ye_src) == 0.0:
        return 0.0
    ye = np.unique(np.quantile(ye_src, np.linspace(0, 1, bins + 1)))
    if xe.size < 2 or ye.size < 2:
        return 0.0
    xb = np.clip(np.searchsorted(xe[1:-1], x, side="right"), 0, xe.size - 2)
    yb = np.clip(np.searchsorted(ye[1:-1], y, side="right"), 0, ye.size - 2)
    joint = np.zeros((xe.size - 1, ye.size - 1), dtype=np.float64)
    np.add.at(joint, (xb, yb), 1.0)
    joint /= n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / denom[mask])))


def group_aware_relevance(cols: list[str], arr: np.ndarray, y: np.ndarray, groups: np.ndarray, bins: int = 8) -> dict[str, float]:
    """Per-feature group-aware relevance: MI(feature, relevance) computed WITHIN each query group and averaged,
    weighted by group size. A ranking-appropriate relevance signal that respects query structure (vs pointwise MI
    pooled across all rows). Self-contained -- does not call the core MRMR MI estimator."""
    out: dict[str, float] = {}
    uniq = np.unique(groups)
    sizes = np.array([np.sum(groups == g) for g in uniq], dtype=np.float64)
    total = float(sizes.sum()) or 1.0
    for j, name in enumerate(cols):
        xj = arr[:, j]
        acc = 0.0
        for gi, g in enumerate(uniq):
            m = groups == g
            if int(m.sum()) < 4:
                continue
            acc += sizes[gi] * _binned_mi(xj[m], y[m], bins=bins)
        out[name] = acc / total
    return out


def select_ltr_features(
    X,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    *,
    mrmr_kwargs: Optional[dict] = None,
    group_aware_mi: bool = False,
    group_aware_keep_frac: float = 0.5,
    random_seed: int = 42,
    verbose: int = 0,
) -> list[str]:
    """Select raw feature columns for the LTR ranker from (X, graded relevance y).

    Default: a standard ``MRMR.fit(X, y)`` (relevance as the MI target, pointwise; groups are NOT passed so the
    core's group-naive MI never raises). With ``group_aware_mi=True``: first rank features by a per-query MI
    (``group_aware_relevance``), keep the top ``group_aware_keep_frac`` (>= 2 features), then run MRMR on that
    subset for redundancy removal. Returns the selected column names (a subset of ``X``'s columns); on any failure
    or empty selection it falls back to ALL numeric columns so the ranker still trains.
    """
    cols, arr = _to_numeric_columns(X)
    if len(cols) <= 1:
        return cols

    candidate_cols = cols
    if group_aware_mi and groups is not None:
        rel = group_aware_relevance(cols, arr, np.asarray(y, dtype=np.float64), np.asarray(groups))
        ranked = sorted(cols, key=lambda c: rel.get(c, 0.0), reverse=True)
        k = max(2, int(round(len(ranked) * float(group_aware_keep_frac))))
        candidate_cols = [c for c in ranked[:k] if rel.get(c, 0.0) > 0.0] or ranked[:k]
        if verbose:
            logger.info("LTR group-aware relevance pre-filter: %d -> %d candidate features", len(cols), len(candidate_cols))

    kw = dict(_DEFAULT_LTR_MRMR_KWARGS)
    if mrmr_kwargs:
        kw.update(mrmr_kwargs)
    kw.setdefault("random_seed", random_seed)
    kw["verbose"] = verbose
    # strict_groups is irrelevant here (we never pass groups to MRMR), but pin False so a user-supplied
    # mrmr_kwargs that flipped it on cannot turn the pointwise relevance fit into a NotImplementedError.
    kw["strict_groups"] = False

    try:
        from mlframe.feature_selection.filters.mrmr import MRMR
        import pandas as pd

        if hasattr(X, "select"):  # polars
            X_cand = X.select(candidate_cols)
            X_fit = X_cand.to_pandas()
        elif isinstance(X, pd.DataFrame):
            X_fit = X[candidate_cols]
        else:
            X_fit = pd.DataFrame(arr[:, [cols.index(c) for c in candidate_cols]], columns=candidate_cols)

        sel = MRMR(**kw)
        sel.fit(X_fit, pd.Series(np.asarray(y), name="relevance"))
        # ``support_`` is either a boolean mask over candidate_cols OR an array of integer column indices.
        support = getattr(sel, "support_", None)
        chosen: list[str] = []
        if support is not None:
            sup = np.asarray(support)
            if sup.dtype == bool:
                chosen = [candidate_cols[i] for i in np.where(sup)[0] if i < len(candidate_cols)]
            else:
                chosen = [candidate_cols[int(i)] for i in sup.tolist() if 0 <= int(i) < len(candidate_cols)]
        if chosen:
            return chosen
        logger.warning("LTR feature selection produced an empty support; falling back to all candidate features.")
        return candidate_cols
    except Exception as exc:  # never let FS break the ranker run
        logger.warning("LTR feature selection failed (%s: %s); training on all numeric features.", type(exc).__name__, exc)
        return candidate_cols
