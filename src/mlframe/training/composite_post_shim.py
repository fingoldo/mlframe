"""sklearn-compatible shim that routes predictions through a fitted pre_pipeline.

Used by the cross-target ensemble (``CompositeCrossTargetEnsemble``) to wrap per-entry models so OOF refit + NNLS-stack can clone components via ``sklearn.clone`` without losing the pre_pipeline (StandardScaler / SimpleImputer / etc.) that was fit during the main training pass.

Why a module-level class. Earlier the shim was defined as a local class inside ``run_composite_post_processing``. ``sklearn.clone`` walks ``get_params`` and refuses to clone any object that does not implement the BaseEstimator interface, so the OOF refit path raised ``Cannot clone object '_PrePipelinePredictShim(...)' : it does not seem to be a scikit-learn estimator as it does not implement a 'get_params' method`` for every component, leaving the NNLS stack with zero weighted members.

Semantics of the shim:
- ``fit(X, y, sample_weight=...)``: ``pre_pipeline.transform(X)`` -> ``inner.fit(...)``. The pre_pipeline is treated as already-fit and is NOT re-fit on the stack subset (refitting would shift the scaling distribution between train and predict).
- ``predict(X)``: ``pre_pipeline.transform(X)`` -> ``inner.predict(...)``.
- ``__sklearn_clone__``: returns a new shim with ``model=clone(self.model)`` and ``pre_pipeline=self.pre_pipeline`` (shared, not cloned). Sharing the fitted pipeline is correct because cloning a Pipeline drops its fitted state, which is exactly what we need to preserve for honest OOF.
- ``estimator_`` is forwarded from a ``CompositeTargetEstimator`` inner so the OOF composite-path (``clone(model.estimator_)``) keeps working when the composite wrapper is nested inside a shim.
"""
from __future__ import annotations

import logging
from typing import Any

from sklearn.base import BaseEstimator, clone

logger = logging.getLogger(__name__)


class PrePipelinePredictShim(BaseEstimator):
    """Route predictions through a fitted ``pre_pipeline`` before delegating to ``model``.

    Parameters
    ----------
    model : estimator with ``predict`` (and optionally ``fit``)
        The trained inner estimator. For the cross-target ensemble this is either a raw boosted model or a fitted ``CompositeTargetEstimator``.
    pre_pipeline : sklearn Pipeline or ``None``
        Already-fit pre-pipeline (typically ``SimpleImputer + StandardScaler``). ``None`` for tree-tier models that do not need numeric preprocessing.
    name : str
        Human-readable component name for log lines / ``__repr__``.
    """

    def __init__(self, model: Any = None, pre_pipeline: Any = None, name: str = "") -> None:
        self.model = model
        self.pre_pipeline = pre_pipeline
        self.name = name

    def _transform(self, X: Any) -> Any:
        if self.pre_pipeline is None:
            return X
        try:
            return self.pre_pipeline.transform(X)
        except Exception:
            # Let inner.predict / inner.fit raise the more descriptive error on pd/pl boundary mismatches (mirrors the previous local-class behaviour).
            return X

    def fit(self, X: Any, y: Any, sample_weight: Any = None, **kwargs: Any) -> "PrePipelinePredictShim":
        X_in = self._transform(X)
        if sample_weight is not None:
            try:
                self.model.fit(X_in, y, sample_weight=sample_weight, **kwargs)
                return self
            except TypeError:
                # Inner does not accept sample_weight; drop and retry.
                pass
        self.model.fit(X_in, y, **kwargs)
        return self

    def predict(self, X: Any) -> Any:
        return self.model.predict(self._transform(X))

    def __sklearn_clone__(self) -> "PrePipelinePredictShim":
        return type(self)(
            model=clone(self.model),
            pre_pipeline=self.pre_pipeline,
            name=self.name,
        )

    def __repr__(self) -> str:
        return f"PrePipelinePredictShim({self.name})"

    @property
    def estimator_(self) -> Any:
        # OOF composite-path does ``clone(model.estimator_)`` when ``isinstance(model, CompositeTargetEstimator)``. Forwarding ``estimator_`` from the inner keeps that path working when the composite wrapper is nested inside a shim.
        inner = self.model
        return getattr(inner, "estimator_", inner)
