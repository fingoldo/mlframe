"""Opt-in sklearn adapter that lets a shortlist standalone transformer be wired into ``train_mlframe_models_suite`` via ``custom_pre_pipelines``.

The 100+ ``compute_*`` transformer functions in this package are research-only standalone helpers -- they are NOT auto-wired into the suite (stacking subsumes most of them, so research-only is the deliberate default). They expose a ``(X_train, y_train, X_query, splitter, ...)`` Mode-A/Mode-B contract rather than the sklearn ``fit``/``transform`` protocol the suite's pre-pipeline slot expects.

``ShortlistTransformerAdapter`` bridges that gap leakage-safely:
  * ``fit(X, y)`` stashes the train fold (X_train, y_train) and any extra kwargs. NO query rows are seen at fit.
  * ``transform(X)`` calls the wrapped ``compute_*`` function in Mode B (``X_query=X``) -- the transformer fits its internal scaler / bandwidth / class banks on the train fold ONLY and applies them to ``X``. Train, val, test, and predict frames all route through the same train-fitted state, so the honest holdout never informs the fit.

This is OPT-IN: research-only remains the default. Users who want a shortlist transformer in the suite pass ``custom_pre_pipelines={"rff": ShortlistTransformerAdapter(compute_rff_features, ...)}``.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import polars as pl

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except Exception:  # pragma: no cover - sklearn always present in mlframe
    BaseEstimator = object  # type: ignore
    TransformerMixin = object  # type: ignore

logger = logging.getLogger(__name__)


def _to_2d_numeric(X) -> np.ndarray:
    """Coerce a polars / pandas / numpy frame to a 2-D float32 ndarray for the compute_* contract."""
    if isinstance(X, pl.DataFrame):
        arr = X.to_numpy()
    elif hasattr(X, "to_numpy"):
        arr = X.to_numpy()
    else:
        arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.ascontiguousarray(arr, dtype=np.float32)


class ShortlistTransformerAdapter(BaseEstimator, TransformerMixin):
    """Wrap a standalone ``compute_*`` shortlist transformer as a leakage-safe sklearn transformer for ``custom_pre_pipelines``.

    Parameters
    ----------
    compute_fn:
        A shortlist transformer with the ``(X_train, y_train, X_query, splitter=None, *, seed, ...)`` signature (e.g. ``compute_rff_features``, ``compute_class_distance_features``, ``compute_local_lift_features``).
    seed:
        Threaded through to ``compute_fn`` for reproducibility.
    needs_y:
        True when ``compute_fn`` requires ``y_train`` (class-distance / local-lift); False for unsupervised transformers (RFF). When False, ``y`` may be omitted at fit.
    passthrough:
        When True (default) the adapter returns the original columns concatenated with the new feature columns, so downstream models keep the raw features. When False, only the new feature columns are returned.
    compute_kwargs:
        Extra keyword arguments forwarded verbatim to ``compute_fn`` (e.g. ``n_features=128``, ``task="binary"``).
    """

    def __init__(
        self,
        compute_fn: Callable[..., pl.DataFrame],
        *,
        seed: int = 42,
        needs_y: bool = True,
        passthrough: bool = True,
        compute_kwargs: Optional[dict] = None,
    ):
        self.compute_fn = compute_fn
        self.seed = seed
        self.needs_y = needs_y
        self.passthrough = passthrough
        self.compute_kwargs = compute_kwargs

    def fit(self, X, y=None):
        # Stash the train fold ONLY. No query rows are observed here -- the wrapped transformer fits its scaler / bandwidth / class banks against this fold at transform time and applies to whatever frame transform() receives.
        self._X_train_ = _to_2d_numeric(X)
        if self.needs_y:
            if y is None:
                raise ValueError(f"{type(self).__name__}: compute_fn requires y_train but fit() got y=None.")
            self._y_train_ = np.asarray(y).ravel()
        else:
            self._y_train_ = None
        if hasattr(X, "columns"):
            self._input_columns_ = list(X.columns)
        else:
            self._input_columns_ = [f"f{i}" for i in range(self._X_train_.shape[1])]
        return self

    def transform(self, X):
        import inspect

        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "_X_train_")
        Xq = _to_2d_numeric(X)
        kwargs = dict(self.compute_kwargs or {})
        kwargs.setdefault("seed", self.seed)
        # Two shortlist calling conventions:
        #   * supervised kNN-style: ``f(X_train, y_train, X_query, *, seed, ...)`` (class_distance / local_lift).
        #   * RFF-style: ``f(X, *, seed, X_query=..., ...)`` -- single positional + X_query keyword.
        # Detect by inspecting the parameter names; route Mode B (fit on train fold, apply to query) either way.
        params = list(inspect.signature(self.compute_fn).parameters)
        if "y_train" in params or "X_train" in params:
            feats = self.compute_fn(self._X_train_, self._y_train_, Xq, **kwargs)
        else:
            feats = self.compute_fn(self._X_train_, X_query=Xq, **kwargs)
        if not self.passthrough:
            self._output_feature_names_ = [str(c) for c in feats.columns]
            return feats.to_pandas()
        # Concatenate original columns + new features. Return pandas so sklearn Pipeline keeps feature names.
        if isinstance(X, pl.DataFrame):
            base = X.to_pandas()
        elif hasattr(X, "reset_index"):
            base = X.reset_index(drop=True)
        else:
            import pandas as pd
            base = pd.DataFrame(np.asarray(X), columns=self._input_columns_[: np.asarray(X).shape[1]])
        feats_pd = feats.to_pandas()
        base = base.reset_index(drop=True)
        feats_pd.index = base.index
        import pandas as pd

        result = pd.concat([base, feats_pd], axis=1)
        self._output_feature_names_ = [str(c) for c in result.columns]
        return result

    def get_feature_names_out(self, input_features=None):
        # Output column count is only known after a transform (it depends on the wrapped function's feature count). Return the recorded output names once available so sklearn's set_output relabelling matches the transformed width; before any transform, fall back to the stashed input names.
        if getattr(self, "_output_feature_names_", None) is not None:
            return np.asarray(self._output_feature_names_, dtype=object)
        names = list(self._input_columns_) if hasattr(self, "_input_columns_") else []
        return np.asarray(names, dtype=object)
