"""``SegmentRoutedEstimator``: reduced-feature specialist for a data-sparse entity segment, rank-spliced back.

Source: AMEX Default Prediction 2nd place -- a separate model trained on only 300 features for clients with
<=2 historical statements (too little history for the full feature set to be meaningfully populated); those
clients' final predictions come EXCLUSIVELY from that specialist model, then get re-spliced into the main
ranking by RE-SORTING WITHIN the low-history subgroup rather than blending the two models' raw scores.

Why re-ranking, not blending: AMEX's metric (like most ranking/AUC-style metrics) only cares about relative
order. The specialist and the main model are fit on different feature sets and can have arbitrarily different
score calibration/scale -- averaging their raw outputs for the sparse segment would inject an uncontrolled
distortion into the FULL population's ranking (the majority segment's carefully-calibrated relative order
would shift by however much the sparse segment's blended scores happen to land). Rank-splicing instead keeps
every sparse-segment row's exact numeric score VALUE from the set the main model already assigned to that
segment (so the segment's overall position/slot in the global distribution is untouched), and only permutes
WHICH row gets which of those values, using the specialist's own within-segment ranking to decide the order.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from mlframe.votenrank.rank_splice import segment_rank_splice

logger = logging.getLogger(__name__)


def _extract_column(X: Any, column: Any) -> np.ndarray:
    """Pull a single named/indexed column out of a pandas/polars/ndarray frame as a 1-D float array."""
    if isinstance(X, pd.DataFrame):
        return np.asarray(X[column], dtype=np.float64)
    try:
        import polars as pl
        if isinstance(X, pl.DataFrame):
            return np.asarray(X[column].to_numpy(), dtype=np.float64)
    except ImportError:
        pass
    return np.asarray(np.asarray(X)[:, column], dtype=np.float64)


def _select_columns(X: Any, mask: np.ndarray, columns: Optional[Sequence[Any]]) -> Any:
    """Select rows matching ``mask`` and, if given, restrict to ``columns``."""
    if isinstance(X, pd.DataFrame):
        X_seg = X.loc[mask]
        return X_seg[list(columns)] if columns is not None else X_seg
    try:
        import polars as pl
        if isinstance(X, pl.DataFrame):
            X_seg = X.filter(pl.Series(mask))
            return X_seg.select(list(columns)) if columns is not None else X_seg
    except ImportError:
        pass
    X_arr = np.asarray(X)[mask]
    if columns is not None:
        return X_arr[:, [int(c) for c in columns]]
    return X_arr


class SegmentRoutedEstimator(BaseEstimator, RegressorMixin):
    """Route data-sparse rows to a reduced-feature specialist model, splice its predictions back by rank.

    Parameters
    ----------
    main_estimator
        sklearn-compatible estimator prototype trained on ALL rows with the FULL feature set. Cloned at fit
        time. Also supplies each sparse-segment row's slot in the global score distribution (see module
        docstring) via its own prediction on those rows.
    specialist_estimator
        sklearn-compatible estimator prototype trained ONLY on the data-sparse segment's rows, optionally
        with a reduced feature subset (``specialist_features``). Cloned at fit time.
    segment_predicate
        ``callable(X) -> (n,) bool array``, True marks a row as belonging to the data-sparse segment (e.g.
        ``lambda X: X["n_statements"] <= 2``). Mutually exclusive with ``auto_segment_column`` -- exactly one
        of the two must be given.
    auto_segment_column
        Opt-in AUTO-segment-discovery: a column name (pandas/polars) or integer index (ndarray) to threshold
        instead of hand-writing ``segment_predicate``. The threshold is the ``auto_segment_quantile`` quantile
        of this column computed ONCE at fit time on the training data and reused unchanged at predict time
        (so train/test routing stays consistent even if the test column distribution drifts). Rows on the
        sparse side of the threshold (see ``auto_segment_direction``) are routed to the specialist -- e.g.
        ``auto_segment_column="n_statements", auto_segment_quantile=0.1, auto_segment_direction="low"``
        auto-discovers the bottom 10% of rows by history length without a manual cutoff.
    auto_segment_quantile
        Quantile (0-1) of ``auto_segment_column`` used as the auto-discovered threshold. Only used when
        ``auto_segment_column`` is set. Default 0.1 (bottom/top 10%).
    auto_segment_direction
        ``"low"`` (default) routes rows AT OR BELOW the quantile threshold to the specialist (the sparse-tail
        case, e.g. low history count); ``"high"`` routes rows AT OR ABOVE it. Only used when
        ``auto_segment_column`` is set.
    specialist_features
        Optional column subset (names for pandas/polars, integer indices for ndarray) the specialist model
        is fit/predicted on. None uses all columns.
    splice_by_rerank
        If True (default, matching the source technique), the specialist's predictions for the segment are
        NOT used directly -- only their RANK ORDER is used to permute the segment's own main-model score
        values (see :func:`mlframe.votenrank.rank_splice.segment_rank_splice`). If False, the specialist's raw predictions replace the segment's
        scores outright (a simpler, riskier combination when the two models' scales aren't comparable).

    Attributes
    ----------
    main_model_, specialist_model_
        The fitted clones.
    segment_rate_
        Fraction of training rows routed to the specialist (diagnostic).
    segment_threshold_
        The fit-time quantile threshold on ``auto_segment_column`` (``None`` unless auto-discovery is used).
    """

    segment_threshold_: Optional[float]

    def __init__(
        self,
        main_estimator: Any,
        specialist_estimator: Any,
        segment_predicate: Optional[Callable[[Any], np.ndarray]] = None,
        specialist_features: Optional[Sequence[Any]] = None,
        splice_by_rerank: bool = True,
        auto_segment_column: Optional[Any] = None,
        auto_segment_quantile: float = 0.1,
        auto_segment_direction: str = "low",
    ) -> None:
        self.main_estimator = main_estimator
        self.specialist_estimator = specialist_estimator
        self.segment_predicate = segment_predicate
        self.specialist_features = specialist_features
        self.splice_by_rerank = splice_by_rerank
        self.auto_segment_column = auto_segment_column
        self.auto_segment_quantile = auto_segment_quantile
        self.auto_segment_direction = auto_segment_direction

    def _resolve_segment_mask(self, X: Any, *, fitting: bool) -> np.ndarray:
        """Dispatch to the manual predicate or the auto-discovered threshold, whichever is configured."""
        if self.segment_predicate is not None and self.auto_segment_column is not None:
            raise ValueError("SegmentRoutedEstimator: pass exactly one of segment_predicate / auto_segment_column, not both")
        if self.segment_predicate is not None:
            return np.asarray(self.segment_predicate(X), dtype=bool)
        if self.auto_segment_column is None:
            raise ValueError("SegmentRoutedEstimator: pass exactly one of segment_predicate / auto_segment_column")
        if self.auto_segment_direction not in ("low", "high"):
            raise ValueError(f"auto_segment_direction must be 'low' or 'high', got {self.auto_segment_direction!r}")

        col_vals = _extract_column(X, self.auto_segment_column)
        if fitting:
            self.segment_threshold_ = float(np.quantile(col_vals, self.auto_segment_quantile))
        threshold = self.segment_threshold_
        return col_vals <= threshold if self.auto_segment_direction == "low" else col_vals >= threshold

    def fit(self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None) -> "SegmentRoutedEstimator":
        """Fit the main model on all rows and the specialist model on the segmented (data-sparse) rows."""
        y_arr = np.asarray(y, dtype=np.float64)
        if self.auto_segment_column is None:
            self.segment_threshold_ = None
        seg_mask = self._resolve_segment_mask(X, fitting=True)
        if seg_mask.shape[0] != y_arr.shape[0]:
            raise ValueError(f"segment_predicate returned {seg_mask.shape[0]} rows, expected {y_arr.shape[0]}")
        self.segment_rate_: float = float(seg_mask.mean()) if seg_mask.shape[0] else 0.0

        self.main_model_ = clone(self.main_estimator)
        main_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.main_model_.fit(X, y_arr, **main_kwargs)

        self.specialist_model_ = clone(self.specialist_estimator)
        self._specialist_fitted_ = bool(seg_mask.any())
        if self._specialist_fitted_:
            X_seg = _select_columns(X, seg_mask, self.specialist_features)
            spec_kwargs = {"sample_weight": sample_weight[seg_mask]} if sample_weight is not None else {}
            self.specialist_model_.fit(X_seg, y_arr[seg_mask], **spec_kwargs)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict with the main model, then splice in the specialist's predictions for the segmented rows."""
        seg_mask = self._resolve_segment_mask(X, fitting=False)
        main_pred = np.asarray(self.main_model_.predict(X), dtype=np.float64)
        if not seg_mask.any() or not self._specialist_fitted_:
            return main_pred

        X_seg = _select_columns(X, seg_mask, self.specialist_features)
        specialist_pred = np.asarray(self.specialist_model_.predict(X_seg), dtype=np.float64)

        if self.splice_by_rerank:
            return segment_rank_splice(main_pred, specialist_pred, seg_mask)
        out: np.ndarray = main_pred.copy()
        out[seg_mask] = specialist_pred
        return out


__all__ = ["SegmentRoutedEstimator"]
