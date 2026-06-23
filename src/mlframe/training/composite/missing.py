"""``MissingAwareComposite`` -- native NaN-in-base handling for composite targets.

A pure WRAPPER over :class:`CompositeTargetEstimator` (no transform-registry
change, no frame copy, train/OOF-only fit). It makes a composite robust to
missing values in the BASE column end-to-end:

- At **fit** it learns, on the TRAIN rows only:
  * a missing INDICATOR (which base rows are NaN),
  * an IMPUTATION value for the base (the train median of the finite base),
  * a per-missing OFFSET correction -- the gap between the mean ``y`` on the
    missing-base rows and the inner composite's mean prediction on those same
    rows (with their base imputed). When missingness is informative (MNAR --
    missingness correlated with ``y``), this offset captures the signal the
    plain composite throws away by imputing.
- At **predict** a NaN base row is imputed to the learned median, routed through
  the inner composite, then corrected by the learned offset. So a NaN base NEVER
  yields a NaN prediction.
- When TOO MANY base rows are missing at fit (``> max_missing_frac``), the
  offset is statistically unreliable, so the wrapper falls back to predicting
  the train ``y`` median for every missing-base row (a documented, NaN-free
  path) instead of trusting a noisy offset.

Why a wrapper, not a transform: imputation + the missing indicator + the
offset are OUT-OF-MODEL corrections layered around an already-fitted composite.
Keeping them in a wrapper means any of the 30+ registry transforms works
unchanged underneath, and the leakage surface is tiny + auditable (everything
learned here is fit on TRAIN rows only and frozen before predict).

Memory: the base column is pulled as a single ndarray via the parent's narrow
``_extract_base`` (one column, never a frame copy). The frame X is passed
straight through to the inner composite -- no ``.copy()`` / ``.to_pandas()`` on
a possibly-100GB carrier.

Biz value
---------
On a target where 20% of base values are MNAR-missing (missingness correlated
with ``y``), the missing-aware composite beats BOTH naive-impute-zero AND
drop-missing on OOS RMSE -- the indicator + offset capture the informative
missingness. Pinned by ``tests/training/test_biz_val_composite_missing.py``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from .estimator import CompositeTargetEstimator, _extract_base

logger = logging.getLogger(__name__)


class MissingAwareComposite(BaseEstimator, RegressorMixin):
    """Wrapper making a :class:`CompositeTargetEstimator` robust to NaN base.

    Parameters
    ----------
    composite
        An UNFITTED :class:`CompositeTargetEstimator` prototype. Cloned at
        fit time so the prototype stays clean. Its ``base_column`` /
        ``base_columns`` config drives which column the wrapper imputes.
    max_missing_frac
        If the fraction of NaN base rows at fit exceeds this, the offset is
        deemed unreliable: every missing-base row at predict falls back to the
        train ``y`` median instead. Default 0.5.
    impute_strategy
        ``"median"`` (default) or ``"mean"`` of the FINITE train base.

    Attributes set at fit
    ---------------------
    composite_
        The fitted inner :class:`CompositeTargetEstimator`.
    base_impute_value_
        The scalar imputed into NaN base rows (learned train-only).
    missing_offset_
        Additive y-scale correction applied to imputed (formerly-NaN) rows.
    y_train_median_
        Fallback constant for the too-many-missing path.
    missing_fraction_
        Fraction of base rows that were NaN at fit.
    use_offset_
        Whether the offset path (True) or the median-fallback path (False) is
        active for missing rows -- decided once at fit from ``missing_fraction_``.
    """

    def __init__(
        self,
        composite: Optional[CompositeTargetEstimator] = None,
        max_missing_frac: float = 0.5,
        impute_strategy: str = "median",
    ) -> None:
        self.composite = composite
        self.max_missing_frac = max_missing_frac
        self.impute_strategy = impute_strategy

    # ------------------------------------------------------------------
    def _base_column_name(self) -> str:
        """Resolve the single base column this wrapper imputes.

        Uses the inner composite's ``base_columns`` / ``base_column`` config.
        Multi-base composites are out of scope for the NaN-impute path (it would
        need a per-column imputer); raises so the misconfiguration is loud.
        """
        comp = self.composite
        cols = getattr(comp, "base_columns", None)
        if cols:
            cols = tuple(cols)
            if len(cols) != 1:
                raise ValueError(
                    "MissingAwareComposite: multi-base composites are not "
                    f"supported (got base_columns={cols!r}); wrap a single-base "
                    "composite."
                )
            return cols[0]
        bc = getattr(comp, "base_column", "")
        if not bc:
            raise ValueError(
                "MissingAwareComposite: the inner composite has no base_column / "
                "base_columns configured; nothing to impute."
            )
        return bc

    def _impute_inplace_safe(self, X: Any, col: str, value: float, mask: np.ndarray) -> Any:
        """Return X with NaN base rows in ``col`` replaced by ``value``.

        Flavour-preserving + no full-frame copy: only the single base column is
        rewritten. The original X is left untouched (we build a shallow new frame
        that SHARES every other column with X -- no row data of the other columns
        is materialised). When no row is missing, X is returned unchanged.
        """
        if not mask.any():
            return X
        base = _extract_base(X, col)
        filled = base.copy()
        filled[mask] = value
        return self._with_column(X, col, filled)

    @staticmethod
    def _with_column(X: Any, col: str, values: np.ndarray) -> Any:
        """Return X with ``col`` replaced by ``values``, sharing other columns.

        Polars ``with_columns`` / pandas ``assign`` both produce a new frame whose
        OTHER columns are shared (Arrow column projection / block reference), so a
        100GB frame is not row-copied -- only the one replaced column allocates.
        """
        try:
            import polars as pl

            if isinstance(X, pl.DataFrame):
                return X.with_columns(pl.Series(col, values))
        except ImportError:
            pass
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            return X.assign(**{col: values})
        if isinstance(X, np.ndarray):
            # ndarray base_column is not name-addressable; the wrapper needs a
            # named base column, so direct the caller to a frame carrier.
            raise TypeError(
                "MissingAwareComposite: ndarray X is unsupported (base column is "
                "name-addressed); pass a pandas or polars DataFrame."
            )
        raise TypeError(
            f"MissingAwareComposite: unsupported X type {type(X).__name__}."
        )

    # ------------------------------------------------------------------
    def fit(self, X: Any, y: Any, sample_weight: np.ndarray | None = None, **fit_kwargs: Any) -> "MissingAwareComposite":
        """Fit the inner composite on imputed base + learn the missing correction.

        All learned quantities (impute value, indicator, offset, fallback median)
        are computed on the TRAIN rows passed here ONLY -- no predict-time data
        touches them, so the wrapper is leakage-free.
        """
        if self.composite is None:
            raise ValueError("MissingAwareComposite: composite must not be None.")
        if not 0.0 < self.max_missing_frac <= 1.0:
            raise ValueError(
                f"MissingAwareComposite: max_missing_frac must be in (0, 1]; got {self.max_missing_frac}."
            )
        if self.impute_strategy not in ("median", "mean"):
            raise ValueError(
                f"MissingAwareComposite: unknown impute_strategy {self.impute_strategy!r}; "
                "choose 'median' or 'mean'."
            )

        col = self._base_column_name()
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        base = _extract_base(X, col)
        missing_mask = ~np.isfinite(base)
        n = base.shape[0]
        self.missing_fraction_ = float(missing_mask.mean()) if n else 0.0

        # Train-only imputation value from the FINITE base rows.
        finite_base = base[~missing_mask]
        if finite_base.size == 0:
            # All base missing: imputation is undefined; impute 0 and lean on the
            # y_train_median fallback for every row (offset path disabled below).
            self.base_impute_value_ = 0.0
        elif self.impute_strategy == "median":
            self.base_impute_value_ = float(np.median(finite_base))
        else:
            self.base_impute_value_ = float(np.mean(finite_base))

        # Train-only fallback median (NaN-safe).
        finite_y = y_arr[np.isfinite(y_arr)]
        self.y_train_median_ = float(np.median(finite_y)) if finite_y.size else 0.0

        # Fit the inner composite on the imputed frame (NaN base rows filled).
        X_imp = self._impute_inplace_safe(X, col, self.base_impute_value_, missing_mask)
        comp = clone(self.composite)
        if sample_weight is not None:
            comp.fit(X_imp, y_arr, sample_weight=sample_weight, **fit_kwargs)
        else:
            comp.fit(X_imp, y_arr, **fit_kwargs)
        self.composite_ = comp

        # Decide the routing for missing rows ONCE, at fit.
        self.use_offset_ = bool(
            missing_mask.any()
            and finite_base.size > 0
            and self.missing_fraction_ <= self.max_missing_frac
        )

        # Learn the per-missing offset: the mean gap between the true y on the
        # missing-base TRAIN rows and the inner composite's prediction there (with
        # base imputed). MNAR missingness shifts y on those rows; the offset
        # recovers that shift that the plain composite ignores. Computed on the
        # SAME imputed train frame the inner was fit on -- train-only, no leakage.
        self.missing_offset_ = 0.0
        if self.use_offset_:
            pred_train = np.asarray(self.composite_.predict(X_imp), dtype=np.float64).reshape(-1)
            miss_y = y_arr[missing_mask]
            miss_pred = pred_train[missing_mask]
            both = np.isfinite(miss_y) & np.isfinite(miss_pred)
            if both.any():
                self.missing_offset_ = float(np.mean(miss_y[both] - miss_pred[both]))

        self._base_column_ = col
        # Best-effort feature names passthrough (sklearn convention).
        try:
            self.feature_names_in_ = list(X.columns)
            self.n_features_in_ = len(X.columns)
        except AttributeError:
            n_feat = getattr(self.composite_, "n_features_in_", None)
            if n_feat is not None:
                self.n_features_in_ = n_feat
        return self

    # ------------------------------------------------------------------
    def predict(self, X: Any) -> np.ndarray:
        """y-scale prediction with NaN base rows imputed + corrected.

        Missing-base rows are imputed to the learned value and routed through the
        inner composite; then either the learned offset is added (informative-
        missingness path) or the prediction is overwritten with the train ``y``
        median (too-many-missing fallback). A NaN base never yields NaN.
        """
        if not hasattr(self, "composite_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("MissingAwareComposite: call fit before predict.")
        col = self._base_column_
        base = _extract_base(X, col)
        missing_mask = ~np.isfinite(base)
        X_imp = self._impute_inplace_safe(X, col, self.base_impute_value_, missing_mask)
        pred = np.asarray(self.composite_.predict(X_imp), dtype=np.float64).reshape(-1).copy()

        if missing_mask.any():
            if self.use_offset_:
                pred[missing_mask] = pred[missing_mask] + self.missing_offset_
            else:
                # Documented fallback: offset unreliable (too many / no finite
                # base at fit) -> predict the train y median on missing rows.
                pred[missing_mask] = self.y_train_median_
            # Final guard: any residual non-finite (e.g. inner extrapolated NaN
            # on an imputed row) is pinned to the y median so a NaN base can
            # never escape as a NaN prediction.
            bad = missing_mask & ~np.isfinite(pred)
            if bad.any():
                pred[bad] = self.y_train_median_
        return pred

    # ------------------------------------------------------------------
    # Inner-attribute delegation (feature_importances_, etc.).
    @property
    def missing_indicator_learned_(self) -> bool:
        """True when the fitted wrapper learned a non-trivial missing correction."""
        return bool(getattr(self, "use_offset_", False))
