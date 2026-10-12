"""Final regressor trained on the target the MRMR feature-base mix found: ``T = y - beta * u(X)``, predicted back as ``y_hat = f(X) + beta * u(X)``.

The selector (an ``MRMR`` with ``fe_target_mix_enable``) records the base of the mix as ``fe_target_mix_provenance_``: a replayable recipe of a design column ``u`` and its OLS slope ``beta`` on the
training rows. This wrapper fits the selector, then fits the regressor on the selected features against ``y - beta * u``, and adds ``beta * u`` back at predict time. ``u`` is replayed from ``X``
alone, so nothing of the training target reaches prediction time, and ``beta`` is fitted on the training rows only.

Only the first base of the provenance is used: each base's slope is fitted against the raw target, not against the residual of the previous base, so several bases do not compose into one
offset. A selector that found no base gives a zero offset, so the wrapper then equals the plain regressor on the selected features (the baseline of the comparison).
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

logger = logging.getLogger(__name__)

__all__ = ["TargetOffsetRegressor"]


def _replay_base(recipe: Any, X: pd.DataFrame) -> np.ndarray:
    """Values of the base column on ``X``: a raw column by name, or an engineered recipe replayed from ``X`` (no target)."""
    if isinstance(recipe, dict) and "raw" in recipe:
        return cast(np.ndarray, np.nan_to_num(np.asarray(X[recipe["raw"]].to_numpy()), nan=0.0, posinf=0.0, neginf=0.0))
    from .engineered_recipes.shared import apply_recipe

    return cast(np.ndarray, np.nan_to_num(np.asarray(apply_recipe(recipe, X)).ravel(), nan=0.0, posinf=0.0, neginf=0.0))


class TargetOffsetRegressor(RegressorMixin, BaseEstimator):
    """Regressor on the selected features of an MRMR selector, trained on the target the selector's feature-base mix found.

    Parameters
    ----------
    selector : estimator
        An ``MRMR`` (or any estimator with ``transform`` and ``fe_target_mix_provenance_`` after ``fit``); cloned at fit time.
    regressor : estimator
        The final model; cloned at fit time and fitted on ``selector.transform(X)``.
    offset_alpha : float, default 0.05
        One-sided significance of the offset's held-out gain. The offset is used only when a model trained on ``T`` beats the same model trained on ``y`` on the held-out odd rows, paired on squared error,
        at this level (a standard-error test, not a fixed margin); otherwise the final model is fitted on ``y`` and the offset is dropped. A mis-specified base (a linear term of a nonlinear one) lowers
        the gradient-boosting score, which this decision catches.
    use_offset : bool, default True
        False keeps the selector (its mix features stay in the selection) and fits the final model on ``y``: the arm that separates the features of the mix from its offset.
    """

    def __init__(self, selector, regressor, offset_alpha=0.05, use_offset=True):
        self.selector = selector
        self.regressor = regressor
        self.offset_alpha = offset_alpha
        self.use_offset = use_offset

    def _offset(self, X: pd.DataFrame) -> np.ndarray:
        """``beta * u(X)`` of the first recorded base, or zeros when the selector recorded none."""
        if self.offset_base_ is None:
            return np.zeros(len(X), dtype=self._y_dtype_)
        return self.offset_beta_ * _replay_base(self.offset_base_, X)

    def fit(self, X, y):
        """Fit the selector, then the regressor on the selected features against ``y - beta * u(X)``."""
        self.decide(X, y)
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = np.asarray(y).ravel()
        self._y_dtype_ = y.dtype
        self.regressor_ = clone(self.regressor).fit(self.selector_.transform(X), y - self._offset(X))
        return self

    def decide(self, X, y):
        """Select, fix beta and run the held-out gate, without fitting the final regressor; sets ``selector_``, ``offset_base_``, ``offset_beta_``.

        Used on its own where only the decision and the slope are needed (building the mix-offset target); ``fit`` is this plus the final fit."""
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = np.asarray(y).ravel()
        self._y_dtype_ = y.dtype
        # Selection and the slope beta see the even rows only; the held-out gate below reads the odd rows, which neither the selector nor beta has seen.
        even = np.arange(len(y)) % 2 == 0
        self.selector_ = clone(self.selector).fit(X.iloc[np.flatnonzero(even)].reset_index(drop=True), y[even])
        provenance = [e for e in (getattr(self.selector_, "fe_target_mix_provenance_", None) or []) if e.get("recipe") is not None] if self.use_offset else []
        if len(provenance) > 1:
            logger.info("TargetOffsetRegressor: %d bases recorded, using the first only (their slopes do not compose into one offset).", len(provenance))
        self.offset_base_ = provenance[0]["recipe"] if provenance else None
        self.offset_beta_ = float(provenance[0]["beta"]) if provenance else 0.0
        Z = self.selector_.transform(X)
        if self.offset_base_ is not None and not self._offset_helps_held_out(Z, y, X, even):
            self.offset_base_ = None
            self.offset_beta_ = 0.0
        return self

    def _offset_helps_held_out(self, Z, y, X, even) -> bool:
        """True when the regressor on ``y - offset`` beats the regressor on ``y`` on the odd rows, paired on squared error, at ``offset_alpha`` one-sided.

        The regressors are fitted on the even rows, the rows the selector and beta were fitted on; the odd rows are untouched by selection, so the decision is out-of-sample. Both fits use the same selected features."""
        from scipy.stats import norm

        off = self._offset(X)
        tr = np.asarray(even, dtype=bool)
        te = ~tr
        with_off = clone(self.regressor).fit(Z[tr], y[tr] - off[tr]).predict(Z[te]) + off[te]
        plain = clone(self.regressor).fit(Z[tr], y[tr]).predict(Z[te])
        d = (with_off - y[te]) ** 2 - (plain - y[te]) ** 2
        if d.size < 2:
            return False
        se = d.std(ddof=1) / np.sqrt(d.size)
        if se == 0.0:
            return bool(d.mean() < 0.0)
        return bool(d.mean() / se < -float(norm.isf(self.offset_alpha)))

    def predict(self, X):
        """``regressor(selector.transform(X)) + beta * u(X)``."""
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return np.asarray(self.regressor_.predict(self.selector_.transform(X))) + self._offset(X)
