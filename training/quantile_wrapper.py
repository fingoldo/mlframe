"""``_QuantileMultiOutputWrapper`` -- fan-out one base regressor to K
independent fits, one per alpha, and stack predictions into ``(N, K)``.

Why
---
Most regression backends (LightGBM, sklearn HistGradientBoostingRegressor,
sklearn QuantileRegressor) accept only a single scalar ``alpha`` /
``quantile`` per fit. Producing K conditional quantiles requires K
independent fits.

CatBoost (``loss_function="MultiQuantile:alpha=0.1,0.5,0.9"``) and
XGBoost >=2.0 (``objective="reg:quantileerror",
quantile_alpha=[0.1, 0.5, 0.9]``) DO single-fit multi-quantile
natively -- those strategies override ``supports_native_quantile=True``
and skip this wrapper.

Contract
--------
- ``fit(X, y, sample_weight=...)`` clones the base estimator K times,
  injects ``alpha`` (or ``quantile`` for HGB/QuantileRegressor) into
  each clone's params, fits all in parallel via joblib.
- ``predict(X)`` returns ``(N, K)`` ndarray, post-processed via
  ``fix_quantile_crossing(..., mode=crossing_fix)``.
- ``get_params`` / ``set_params`` -- sklearn-clone compatibility
  inherited from BaseEstimator.

The base estimator's parameter NAME for the quantile level varies by
library; we probe at fit-time in this order:
  1. ``"quantile_alpha"``  (XGB, takes a list -- only used by XGB
     native path; not relevant here since XGB has its own native path)
  2. ``"alpha"``           (LightGBM, NGBoost, sklearn QuantileRegressor)
  3. ``"quantile"``        (sklearn HistGradientBoostingRegressor)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from mlframe.training.quantile_postproc import fix_quantile_crossing

logger = logging.getLogger(__name__)


_ALPHA_PARAM_CANDIDATES = ("alpha", "quantile")
"""Parameter names to probe for the per-fit quantile level.

We try them in order on a clone's ``get_params()``; first match wins.
``"quantile_alpha"`` (XGB list-of-quantiles) is intentionally NOT in
this list -- XGB's native multi-quantile path skips this wrapper.
"""


def _probe_alpha_param_name(base_estimator: BaseEstimator) -> str:
    """Identify which kwarg name the base estimator uses for the quantile.

    Raises if neither standard name is in the estimator's signature.
    """
    params = base_estimator.get_params(deep=False)
    for name in _ALPHA_PARAM_CANDIDATES:
        if name in params:
            return name
    raise ValueError(
        f"_QuantileMultiOutputWrapper: base estimator "
        f"{type(base_estimator).__name__} does not accept any of the "
        f"standard quantile parameters {list(_ALPHA_PARAM_CANDIDATES)}. "
        f"Pass an estimator that takes ``alpha`` (LGB / sklearn "
        f"QuantileRegressor) or ``quantile`` (HGB)."
    )


def _resolve_n_jobs(requested: Union[int, str], k: int) -> int:
    """Resolve ``n_jobs="auto"`` to ``min(K, cpu_count // 2)``.

    Avoids nested-parallelism thrashing when the inner estimator already
    uses all cores via its own ``n_jobs=-1``.
    """
    if isinstance(requested, int):
        return requested
    if requested == "auto":
        cpu = os.cpu_count() or 2
        return max(1, min(k, cpu // 2))
    raise ValueError(
        f"_QuantileMultiOutputWrapper.n_jobs must be int or 'auto'; "
        f"got {requested!r}"
    )


class _QuantileMultiOutputWrapper(BaseEstimator, RegressorMixin):
    """Fits ``len(alphas)`` independent regressors and stacks their
    predictions into ``(N, K)``.

    Parameters
    ----------
    base_estimator : sklearn-compatible regressor
        The per-alpha base. MUST accept ``alpha`` or ``quantile`` as
        a constructor kwarg (probed at fit time).
    alphas : sequence of K floats
        Quantile levels to predict. Sorted ascending, each in (0, 1).
    crossing_fix : ``"sort"`` / ``"isotonic"`` / ``"none"``
        Post-prediction crossing fix. Default ``"sort"``.
    n_jobs : int or ``"auto"``
        Parallelism for the K fits. ``"auto"`` -> ``min(K, cpu//2)``.

    Notes
    -----
    The wrapper is sklearn-clone-friendly (BaseEstimator handles
    ``get_params`` / ``set_params``) but stores the per-alpha fitted
    estimators internally as ``self.estimators_`` (list of length K).
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        alphas: Sequence[float] = (0.1, 0.5, 0.9),
        crossing_fix: str = "sort",
        n_jobs: Union[int, str] = "auto",
    ):
        self.base_estimator = base_estimator
        self.alphas = tuple(alphas)
        self.crossing_fix = crossing_fix
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # sklearn estimator API
    # ------------------------------------------------------------------

    def fit(self, X, y, sample_weight=None):
        if y is None:
            raise ValueError("_QuantileMultiOutputWrapper requires non-None y.")
        y_arr = np.asarray(y).ravel()
        if y_arr.ndim != 1:
            raise ValueError(
                f"_QuantileMultiOutputWrapper expects 1-D y; got shape {y_arr.shape}"
            )
        alpha_param = _probe_alpha_param_name(self.base_estimator)
        n_jobs = _resolve_n_jobs(self.n_jobs, len(self.alphas))

        from joblib import Parallel, delayed

        def _fit_one(alpha):
            est = clone(self.base_estimator)
            est.set_params(**{alpha_param: float(alpha)})
            if sample_weight is not None:
                try:
                    est.fit(X, y_arr, sample_weight=sample_weight)
                except TypeError:
                    # Estimator doesn't accept sample_weight -> warn + fit unweighted.
                    logger.warning(
                        "_QuantileMultiOutputWrapper: %s does not accept "
                        "sample_weight; fitting alpha=%g unweighted.",
                        type(est).__name__, alpha,
                    )
                    est.fit(X, y_arr)
            else:
                est.fit(X, y_arr)
            return est

        if n_jobs == 1 or len(self.alphas) <= 1:
            self.estimators_ = [_fit_one(a) for a in self.alphas]
        else:
            self.estimators_ = Parallel(n_jobs=n_jobs)(
                delayed(_fit_one)(a) for a in self.alphas
            )
        # sklearn convention: expose feature info from the first fitted
        # estimator (all are fit on identical X).
        if self.estimators_ and hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "estimators_") or not self.estimators_:
            raise RuntimeError(
                "_QuantileMultiOutputWrapper.predict called before fit."
            )
        cols = [est.predict(X) for est in self.estimators_]
        # Each predict returns (N,); stack into (N, K).
        out = np.column_stack(cols)
        return fix_quantile_crossing(out, self.alphas, mode=self.crossing_fix)

    # ------------------------------------------------------------------
    # sklearn introspection
    # ------------------------------------------------------------------

    def __sklearn_tags__(self):
        # Inherit base estimator's tags but flag the multioutput shape.
        if hasattr(self.base_estimator, "__sklearn_tags__"):
            tags = self.base_estimator.__sklearn_tags__()
        else:
            from sklearn.utils._tags import default_tags
            tags = default_tags(self.base_estimator)
        tags.target_tags.multi_output = True
        return tags


__all__ = ["_QuantileMultiOutputWrapper"]
