"""``PerGroupCompositeRouter``: predict-time routing for opt-in per-group composite discovery.

Mirrors :class:`mlframe.training.composite.grouped_block_stacking.GroupedBlockStacker`'s
architecture -- one submodel per group, routed at predict time by a group key column --
applied to composite-target specs instead of feature blocks. Consumes
``CompositeTargetDiscovery.specs_by_group_`` (populated only when
``config.per_group_discovery_enabled=True``, see ``discovery/_per_group.py``): each
group gets its OWN :class:`CompositeTargetEstimator` fit with that group's own
discovered (base, transform) spec (the top-ranked one, i.e. ``specs_by_group_[g][0]``);
groups absent from ``specs_by_group_`` (below ``per_group_min_rows``, or unseen at
predict time) route to a single GLOBAL :class:`CompositeTargetEstimator` fit from
``discovery.specs_[0]`` -- the existing global spec, so the default single-global-spec
behaviour is exactly what a caller gets for every group when the opt-in flag is off
(this router is simply never constructed in that case).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from .estimator import CompositeTargetEstimator, _extract_groups

logger = logging.getLogger(__name__)


def _drop_group_column(X: Any, group_column: str) -> Any:
    """Drop the group-key column before handing X to an inner ``CompositeTargetEstimator``.

    The group key is routing metadata, not a real feature (constant within a
    per-group submodel, and typically non-numeric); the wrapper's own
    ``group_column`` drop only fires for ``requires_groups=True`` transforms, which
    this router does not use (each group already picks its own transform).
    """
    if isinstance(X, pd.DataFrame):
        return X.drop(columns=[group_column])
    return X.drop(group_column)  # polars DataFrame


class PerGroupCompositeRouter(BaseEstimator, RegressorMixin):
    """Routes each row to its group's own composite-target submodel.

    Parameters
    ----------
    discovery
        A fitted ``CompositeTargetDiscovery`` with ``specs_`` (global fallback) and,
        when ``config.per_group_discovery_enabled=True`` was used, ``specs_by_group_``.
    base_estimator
        Zero-arg-clonable sklearn-compatible regressor PROTOTYPE; cloned fresh for
        every per-group (and the global-fallback) ``CompositeTargetEstimator``.
    group_column
        Column carrying the group key, read from ``X`` at both fit and predict time.

    Attributes set at fit
    ----------------------
    group_estimators_
        ``{group_value: fitted CompositeTargetEstimator}`` for every group that had a
        per-group spec AND enough training rows present in the ``fit`` call's ``X``.
    global_estimator_
        Fallback ``CompositeTargetEstimator`` fit on ALL rows using ``discovery.specs_[0]``
        (the global spec) -- used for any row whose group has no per-group submodel,
        including groups unseen at predict time.
    """

    def __init__(
        self,
        discovery: Any = None,
        base_estimator: Any = None,
        group_column: str = "",
        min_group_fit_rows: int = 10,
    ) -> None:
        self.discovery = discovery
        self.base_estimator = base_estimator
        self.group_column = group_column
        self.min_group_fit_rows = min_group_fit_rows

    def fit(self, X: Any, y: Any) -> "PerGroupCompositeRouter":
        """Fit a global fallback submodel plus one per-group submodel per group with >= ``min_group_fit_rows`` rows."""
        if self.discovery is None or self.base_estimator is None:
            raise ValueError("discovery and base_estimator are required.")
        if not self.group_column:
            raise ValueError("group_column is required.")
        specs_ = list(getattr(self.discovery, "specs_", []) or [])
        if not specs_:
            raise ValueError("discovery.specs_ is empty; nothing to route.")
        specs_by_group = dict(getattr(self.discovery, "specs_by_group_", {}) or {})

        y_arr = np.asarray(y, dtype=np.float64)
        group_vals = _extract_groups(X, self.group_column)

        # Global fallback submodel: fit on ALL rows using the global (best) spec, exactly the
        # single-spec-forced-on-all-groups behaviour a caller gets without this router.
        _global_spec = specs_[0]
        self.global_estimator_ = CompositeTargetEstimator(
            base_estimator=clone(self.base_estimator),
            transform_name=_global_spec.transform_name,
            base_column=_global_spec.base_column,
        )
        self.global_estimator_.fit(_drop_group_column(X, self.group_column), y_arr)

        self.group_estimators_: dict[Any, CompositeTargetEstimator] = {}
        for group_val, group_specs in specs_by_group.items():
            if not group_specs:
                continue
            mask = group_vals == group_val
            n_rows = int(mask.sum())
            if n_rows < self.min_group_fit_rows:
                logger.info(
                    "[PerGroupCompositeRouter] group=%r has only %d fit rows (< %d); routing to the global fallback.",
                    group_val, n_rows, self.min_group_fit_rows,
                )
                continue
            spec = group_specs[0]
            X_group = X.loc[mask] if isinstance(X, pd.DataFrame) else X.filter(mask)
            X_group = _drop_group_column(X_group, self.group_column)
            est = CompositeTargetEstimator(
                base_estimator=clone(self.base_estimator),
                transform_name=spec.transform_name,
                base_column=spec.base_column,
            )
            try:
                est.fit(X_group, y_arr[mask])
            except Exception as exc:  # -- a per-group submodel failing to fit falls back to global, never aborts fit()
                logger.warning(
                    "[PerGroupCompositeRouter] group=%r submodel fit failed (%s); routing to the global fallback.",
                    group_val, exc,
                )
                continue
            self.group_estimators_[group_val] = est
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Route each row to its own group's fitted submodel, falling back to the global submodel elsewhere."""
        group_vals = _extract_groups(X, self.group_column)
        preds = np.asarray(self.global_estimator_.predict(_drop_group_column(X, self.group_column)), dtype=np.float64)
        for group_val, est in self.group_estimators_.items():
            mask = group_vals == group_val
            if not mask.any():
                continue
            X_group = X.loc[mask] if isinstance(X, pd.DataFrame) else X.filter(mask)
            X_group = _drop_group_column(X_group, self.group_column)
            preds[mask] = np.asarray(est.predict(X_group), dtype=np.float64)
        return preds


__all__ = ["PerGroupCompositeRouter"]
