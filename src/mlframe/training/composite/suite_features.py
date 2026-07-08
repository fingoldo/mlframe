"""CompositeFeatureGenerator: make a composite-target signal usable as ONE engineered FEATURE for the broader mlframe training suite.

This is the single-final-model alternative to the cross-target NNLS ensemble. Instead of compositing the target and inverting, the discovered composite prediction is attached as a column alongside the raw features; a downstream model corrects the composite's bias regions on the SAME row using every other feature.

Leakage contract (the cardinal sin to avoid):
- ``fit_transform`` / ``fit`` produce the feature on the training frame via OUT-OF-FOLD prediction (``composite_oof_predictions``): each row's value comes from a wrapper trained on the OTHER folds, so the downstream model never trains on an in-sample-optimistic column.
- After ``fit``, a single wrapper is (re)fitted on ALL train rows and stashed; ``transform`` on NEW data uses that fitted-predict path (``composite_predictions_as_feature``). New data is genuinely out-of-sample, so the fitted-predict value is honest.

No frame copy beyond the appended column on the polars path (``with_columns`` is zero-copy); the pandas path warns above the suite-wide 2 GB gate (pandas has no zero-copy column add). Reuses ``composite_oof_predictions`` / ``composite_predictions_as_feature`` -- no duplication.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, clone

from .estimator import CompositeTargetEstimator
from .ensemble.feature_stacking import (
    composite_oof_predictions,
    composite_predictions_as_feature,
)

logger = logging.getLogger(__name__)


def _spec_base_columns(spec: Any) -> tuple[str, ...]:
    """Full ordered base-column tuple for a CompositeSpec (single- or multi-base)."""
    base = getattr(spec, "base_column", None)
    extra = getattr(spec, "extra_base_columns", ()) or ()
    cols = tuple(c for c in (base, *extra) if c)
    if not cols:
        raise ValueError("CompositeFeatureGenerator: spec has no base_column.")
    return cols


def _default_inner() -> Any:
    """Cheap, dependency-light inner regressor for the composite wrapper.

    HistGradientBoostingRegressor is always available (sklearn core), handles NaNs natively, and is a reasonable default residual learner. Callers who want LightGBM / CatBoost pass ``base_estimator`` explicitly.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(random_state=0)


class CompositeFeatureGenerator(BaseEstimator, TransformerMixin):
    """Turn a discovered composite spec (or a prototype wrapper) into one OOF feature column.

    Parameters
    ----------
    spec
        A discovered ``CompositeSpec`` describing the transform + base column(s). Either ``spec`` or ``wrapper_factory`` must be given (``spec`` is the common path). When ``spec`` is supplied the generator builds a fresh ``CompositeTargetEstimator`` per fold from it.
    base_estimator
        Unfitted inner regressor cloned for every fold + the final all-train fit. Defaults to ``HistGradientBoostingRegressor`` (sklearn core, NaN-aware). Ignored when ``wrapper_factory`` is supplied.
    wrapper_factory
        Advanced escape hatch: a zero-arg callable returning a fresh unfitted predict-supporting wrapper. Use it to wire an already-configured ``CompositeTargetEstimator`` (or any compatible object). Mutually exclusive with ``spec`` for wrapper construction; when both are given ``wrapper_factory`` wins.
    column_name
        Name of the appended feature column. Default derives from the spec (``composite_pred__{transform}__{base}``).
    n_splits, random_state
        OOF CV settings forwarded to ``composite_oof_predictions``.
    time_aware, cv_splitter, groups
        Forwarded to ``composite_oof_predictions`` for time-series / group-aware OOF (no row from the same group lands in both train and val of a fold).
    fit_final_on_all
        When True (default) a single wrapper is fitted on ALL train rows at ``fit`` time and stored as ``estimator_`` so ``transform`` works on new data. Set False only when the generator is used purely for the OOF training column (then ``transform`` raises).

    Attributes set at fit
    ---------------------
    oof_feature_
        ``(n,)`` OOF prediction vector produced on the training frame (leakage-free).
    estimator_
        The single wrapper fitted on all train rows, used by ``transform`` on new data (present iff ``fit_final_on_all``).
    column_name_
        Resolved name of the appended column.
    """

    def __init__(
        self,
        spec: Any = None,
        *,
        base_estimator: Any = None,
        wrapper_factory: Optional[Callable[[], Any]] = None,
        column_name: Optional[str] = None,
        n_splits: int = 5,
        random_state: int = 42,
        time_aware: bool = False,
        cv_splitter: Any = None,
        groups: Optional[np.ndarray] = None,
        fit_final_on_all: bool = True,
    ) -> None:
        self.spec = spec
        self.base_estimator = base_estimator
        self.wrapper_factory = wrapper_factory
        self.column_name = column_name
        self.n_splits = n_splits
        self.random_state = random_state
        self.time_aware = time_aware
        self.cv_splitter = cv_splitter
        self.groups = groups
        self.fit_final_on_all = fit_final_on_all

    # ------------------------------------------------------------------
    def _resolve_column_name(self) -> str:
        if self.column_name:
            return self.column_name
        spec = self.spec
        t_name = getattr(spec, "transform_name", None)
        b_col = getattr(spec, "base_column", None)
        if t_name and b_col:
            return f"composite_pred__{t_name}__{b_col}"
        return "composite_pred"

    def _make_wrapper(self) -> Any:
        """Build a fresh unfitted wrapper from ``wrapper_factory`` or ``spec``."""
        if self.wrapper_factory is not None:
            return self.wrapper_factory()
        if self.spec is None:
            raise ValueError("CompositeFeatureGenerator: supply either `spec` or `wrapper_factory`.")
        inner = self.base_estimator if self.base_estimator is not None else _default_inner()
        base_columns = _spec_base_columns(self.spec)
        return CompositeTargetEstimator(
            base_estimator=clone(inner),
            transform_name=self.spec.transform_name,
            base_column=self.spec.base_column,
            base_columns=base_columns if len(base_columns) > 1 else None,
        )

    # ------------------------------------------------------------------
    def fit(self, X: Any, y: Any, **fit_kwargs: Any) -> "CompositeFeatureGenerator":
        """Compute the leakage-free OOF feature on ``X`` and (optionally) fit the final wrapper.

        ``y`` is required: OOF prediction needs the target to train per-fold wrappers. The OOF vector is stored in ``oof_feature_``; the appended-column frame is produced by ``transform`` / ``fit_transform``.
        """
        if y is None:
            raise ValueError("CompositeFeatureGenerator.fit requires y for OOF prediction.")
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        cols = getattr(X, "columns", None)
        if cols is not None:
            self.n_features_in_ = len(cols)
        elif getattr(X, "shape", None) is not None and len(X.shape) >= 2:
            self.n_features_in_ = int(X.shape[1])
        self.column_name_ = self._resolve_column_name()
        self.oof_feature_ = composite_oof_predictions(
            self._make_wrapper,
            X,
            y_arr,
            n_splits=int(self.n_splits),
            random_state=int(self.random_state),
            fit_kwargs=dict(fit_kwargs) or None,
            time_aware=bool(self.time_aware),
            cv_splitter=self.cv_splitter,
            groups=self.groups,
        )
        if self.fit_final_on_all:
            est = self._make_wrapper()
            est.fit(X, y_arr, **fit_kwargs)
            self.estimator_ = est
        else:
            self.estimator_ = None
        return self

    # ------------------------------------------------------------------
    def fit_transform(self, X: Any, y: Any = None, **fit_kwargs: Any) -> Any:
        """Fit then attach the OOF feature column to ``X`` (the leakage-free training column).

        Unlike a plain ``fit().transform()`` this attaches the OUT-OF-FOLD vector -- the training column a downstream model must train on. ``transform`` (new data) uses the all-train fitted wrapper instead.
        """
        self.fit(X, y, **fit_kwargs)
        return self._attach(X, self.oof_feature_)

    # ------------------------------------------------------------------
    def transform(self, X: Any) -> Any:
        """Attach the composite feature to NEW data via the all-train fitted wrapper (honest out-of-sample)."""
        if getattr(self, "estimator_", None) is None:
            from sklearn.exceptions import NotFittedError

            raise NotFittedError(
                "CompositeFeatureGenerator.transform: no fitted all-train estimator "
                "(fit_final_on_all=False or fit not called). Use fit_transform for the "
                "training column, or set fit_final_on_all=True to enable transform on new data."
            )
        return composite_predictions_as_feature(self.estimator_, X, column_name=self.column_name_)

    # ------------------------------------------------------------------
    def _attach(self, X: Any, values: np.ndarray) -> Any:
        """Append ``values`` to ``X`` as ``column_name_`` (polars zero-copy / pandas copy)."""
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        try:
            import polars as pl

            if isinstance(X, pl.DataFrame):
                return X.with_columns(pl.Series(name=self.column_name_, values=vals))
        except ImportError:
            pass
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            # Append one column WITHOUT deep-copying every existing column (X can be 100GB+). ``assign`` makes a
            # shallow BlockManager copy -- the existing column arrays are shared, only the new column is added --
            # and returns a NEW frame (the caller's X is not mutated), unlike ``X.copy()`` which duplicated all
            # columns just to add one.
            return X.assign(**{self.column_name_: vals})
        raise TypeError(f"CompositeFeatureGenerator: unsupported X type {type(X).__name__}; " "pass a pandas / polars DataFrame.")

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """sklearn feature-name surface: the appended composite column name."""
        name = getattr(self, "column_name_", None) or self._resolve_column_name()
        base = list(input_features) if input_features is not None else []
        return np.asarray([*base, name], dtype=object)
