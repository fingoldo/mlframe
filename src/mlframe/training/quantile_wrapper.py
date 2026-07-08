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
from typing import Sequence

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

    Probes ``get_params`` first (works for sklearn-pure estimators);
    falls back to ``set_params`` (succeeds when the estimator accepts
    the kwarg via __init__/**kwargs even when ``get_params`` omits it
    -- LightGBM's LGBMRegressor matches this pattern: ``alpha`` is a
    valid loss-function kwarg but not in get_params(deep=False)).
    Raises if neither candidate is accepted.
    """
    params = base_estimator.get_params(deep=False)
    for name in _ALPHA_PARAM_CANDIDATES:
        if name in params:
            return name
    # Fallback: try set_params with each candidate and see which sticks.
    # We use a clone so probing doesn't mutate the user's estimator.
    test_clone = clone(base_estimator)
    for name in _ALPHA_PARAM_CANDIDATES:
        try:
            test_clone.set_params(**{name: 0.5})
            return name
        except (ValueError, TypeError):
            continue
    raise ValueError(
        f"_QuantileMultiOutputWrapper: base estimator "
        f"{type(base_estimator).__name__} does not accept any of the "
        f"standard quantile parameters {list(_ALPHA_PARAM_CANDIDATES)}. "
        f"Pass an estimator that takes ``alpha`` (LGB / sklearn "
        f"QuantileRegressor) or ``quantile`` (HGB)."
    )


def _resolve_n_jobs(requested: int | str, k: int) -> int:
    """Resolve ``n_jobs="auto"`` to ``min(K, cpu_count // 2)``.

    Avoids nested-parallelism thrashing when the inner estimator already
    uses all cores via its own ``n_jobs=-1``.
    """
    if isinstance(requested, int):
        return requested
    if requested == "auto":
        cpu = os.cpu_count() or 2
        return max(1, min(k, cpu // 2))
    raise ValueError(f"_QuantileMultiOutputWrapper.n_jobs must be int or 'auto'; " f"got {requested!r}")


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
        n_jobs: int | str = "auto",
    ):
        self.base_estimator = base_estimator
        # sklearn contract: store params verbatim (no tuple() transform) so clone / get_params round-trip the value as passed.
        # alphas is materialised to a tuple where iterated (fit / predict) instead.
        self.alphas = alphas
        self.crossing_fix = crossing_fix
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # sklearn estimator API
    # ------------------------------------------------------------------

    def fit(self, X, y, sample_weight=None):
        """Fit one cloned ``base_estimator`` per alpha in parallel (or serially when ``n_jobs == 1`` or there's a single alpha), storing the fitted list in ``self.estimators_``."""
        if y is None:
            raise ValueError("_QuantileMultiOutputWrapper requires non-None y.")
        y_raw = np.asarray(y)
        if y_raw.ndim != 1:
            raise ValueError(f"_QuantileMultiOutputWrapper expects 1-D y; got shape {y_raw.shape}")
        y_arr = y_raw
        alphas = tuple(self.alphas)
        alpha_param = _probe_alpha_param_name(self.base_estimator)
        n_jobs = _resolve_n_jobs(self.n_jobs, len(alphas))

        from joblib import Parallel, delayed

        def _fit_one(alpha):
            """Clone the base estimator, set its alpha-quantile param, and fit it on ``(X, y_arr)``, falling back to unweighted fit if ``sample_weight`` is unsupported."""
            est = clone(self.base_estimator)
            est.set_params(**{alpha_param: float(alpha)})
            if sample_weight is not None:
                try:
                    est.fit(X, y_arr, sample_weight=sample_weight)
                except TypeError:
                    # Estimator doesn't accept sample_weight -> warn + fit unweighted.
                    logger.warning(
                        "_QuantileMultiOutputWrapper: %s does not accept " "sample_weight; fitting alpha=%g unweighted.",
                        type(est).__name__,
                        alpha,
                    )
                    est.fit(X, y_arr)
            else:
                est.fit(X, y_arr)
            return est

        if n_jobs == 1 or len(alphas) <= 1:
            self.estimators_ = [_fit_one(a) for a in alphas]
        else:
            # Use Parallel as a context manager so the worker pool is torn
            # down deterministically when fit returns. backend="threading"
            # mirrors the MRMR / screen flip (commits 0da27e0, 47923ab):
            # quantile-wrapper workers fit independent boosting estimators on
            # the SAME (X, y) tuple, so threading shares those arrays zero-copy
            # instead of paying loky's per-worker pickle copy (the iter-371
            # OOM pathway under Windows paging pressure). Tree backends release
            # the GIL inside their native fit; the sklearn pre/post hooks
            # (validation, attribute stamping) are short.
            with Parallel(n_jobs=n_jobs, backend="threading") as par:
                self.estimators_ = par(delayed(_fit_one)(a) for a in alphas)
        # sklearn convention: expose feature info from the first fitted
        # estimator (all are fit on identical X).
        if self.estimators_ and hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        return self

    def predict(self, X) -> np.ndarray:
        """Predict all fitted alpha-quantile estimators and stack their per-alpha predictions column-wise into ``(N, K)``."""
        if not hasattr(self, "estimators_") or not self.estimators_:
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("_QuantileMultiOutputWrapper.predict called before fit.")
        cols = [est.predict(X) for est in self.estimators_]
        # Each predict returns (N,); stack into (N, K).
        out = np.column_stack(cols)
        return fix_quantile_crossing(out, tuple(self.alphas), mode=self.crossing_fix)

    # ------------------------------------------------------------------
    # sklearn introspection
    # ------------------------------------------------------------------

    def __sklearn_tags__(self):
        # Inherit base estimator's tags but flag the multioutput shape.
        # Falls through to BaseEstimator's default __sklearn_tags__ when
        # the base estimator predates the tags-protocol (sklearn < 1.6).
        if hasattr(self.base_estimator, "__sklearn_tags__"):
            tags = self.base_estimator.__sklearn_tags__()
        else:
            tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags


__all__ = ["_QuantileMultiOutputWrapper"]
