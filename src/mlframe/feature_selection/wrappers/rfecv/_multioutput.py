"""Per-target multi-output feature selection for ``RFECV``.

sklearn's RFE/RFECV is single-target. When the caller passes a 2D ``y`` (multilabel classification or multi-target
regression) AND opts in via ``multioutput_strategy='union'|'intersect'``, ``RFECV.fit`` delegates here: one single-target
RFECV is cloned and fitted per output column, and the per-column ``support_`` masks are aggregated (OR / AND) into the
final selection. The default (``multioutput_strategy=None``) keeps the historical clear ``NotImplementedError``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone

logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")


def _y_columns(y):
    """Yield (label, y_column_1d) for each output column of a 2D y (DataFrame / ndarray)."""
    if isinstance(y, pd.DataFrame):
        for col in y.columns:
            yield str(col), y[col].to_numpy()
    else:
        arr = np.asarray(y)
        for k in range(arr.shape[1]):
            yield f"y{k}", arr[:, k]


def fit_multioutput(self, X, y, groups, sample_weight, fit_params, strategy: str):
    """Fit one single-target RFECV per output column of ``y`` and aggregate support_ via ``strategy`` ('union'/'intersect').

    Sets the standard fitted attributes (support_ as a bool mask over feature_names_in_, n_features_, n_features_in_,
    feature_names_in_, _selected_cols_cache) plus ``self.multioutput_supports_`` (per-column selected feature-name lists).
    Returns ``self``.
    """
    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else list(map(str, range(X.shape[1])))
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    n_features = len(feature_names)

    per_column_selected: dict[str, list] = {}
    skipped_columns: dict[str, str] = {}
    n_columns = 0
    for label, y_col in _y_columns(y):
        n_columns += 1
        sub = clone(self)
        # The sub-fit is single-target; clear the strategy so it takes the normal path, not infinite recursion.
        sub.multioutput_strategy = None
        # Per-column resilience: a single degenerate output (e.g. an all-constant target) raising inside its
        # sub-fit must not abort the OTHER, valid columns. Skip the failed column, aggregate over the rest.
        try:
            sub.fit(X, y_col, groups=groups, sample_weight=sample_weight, **(fit_params or {}))
        except Exception as exc:
            skipped_columns[label] = f"{type(exc).__name__}: {exc}"
            logger.warning("RFECV multioutput[%s]: sub-fit failed (%s); skipping this output column.", label, exc)
            continue
        per_column_selected[label] = list(sub.get_feature_names_out())
        if self.verbose:
            logger.info("RFECV multioutput[%s]: selected %d features.", label, len(per_column_selected[label]))

    if not per_column_selected:
        if skipped_columns:
            raise ValueError(f"RFECV multioutput: all {n_columns} output column(s) failed to fit: {skipped_columns}")
        raise ValueError("RFECV multioutput: y has no output columns to fit.")

    sets = [set(v) for v in per_column_selected.values()]
    if strategy == "union":
        aggregated = set().union(*sets)
    else:  # intersect
        aggregated = set.intersection(*sets)

    # Preserve original feature order so support_ / get_feature_names_out are deterministic regardless of set iteration order.
    selected_in_order = [f for f in feature_names if f in aggregated]

    support_mask = np.zeros(n_features, dtype=bool)
    for f in selected_in_order:
        support_mask[name_to_idx[f]] = True

    self.support_ = support_mask
    self.n_features_ = int(support_mask.sum())
    self.n_features_in_ = n_features
    self.feature_names_in_ = feature_names
    self._selected_cols_cache = selected_in_order
    self.multioutput_supports_ = per_column_selected
    self.multioutput_skipped_ = skipped_columns
    self.multioutput_strategy_ = strategy
    return self
