"""sklearn-compatible shim that routes predictions through a fitted pre_pipeline.

Used by the cross-target ensemble (``CompositeCrossTargetEnsemble``) to wrap per-entry models so OOF refit + NNLS-stack can clone components via ``sklearn.clone`` without losing the pre_pipeline (StandardScaler / SimpleImputer / etc.) that was fit during the main training pass.

Why a module-level class. Earlier the shim was defined as a local class inside ``run_composite_post_processing``. ``sklearn.clone`` walks ``get_params`` and refuses to clone any object that does not implement the BaseEstimator interface, so the OOF refit path raised ``Cannot clone object '_PrePipelinePredictShim(...)' : it does not seem to be a scikit-learn estimator as it does not implement a 'get_params' method`` for every component, leaving the NNLS stack with zero weighted members.

Semantics of the shim:
- ``fit(X, y, sample_weight=...)``: ``pre_pipeline.transform(X)`` -> ``inner.fit(...)``. The pre_pipeline is treated as already-fit and is NOT re-fit on the stack subset (refitting would shift the scaling distribution between train and predict).
- ``predict(X)``: ``pre_pipeline.transform(X)`` -> ``inner.predict(...)``.
- ``__sklearn_clone__``: returns a new shim with ``model=clone(self.model)`` and ``pre_pipeline=self.pre_pipeline`` (shared, not cloned). Sharing the fitted pipeline is correct because cloning a Pipeline drops its fitted state, which is exactly what we need to preserve for honest OOF.
- ``estimator_`` is forwarded from a ``CompositeTargetEstimator`` inner so the OOF composite-path (``clone(model.estimator_)``) keeps working when the composite wrapper is nested inside a shim.
- ``__sklearn_is_fitted__``: reflects the REAL fit state (the inner model is fitted, whether via ``shim.fit`` or because a pre-fitted inner was passed in). Without it, ``sklearn.check_is_fitted(shim)`` falls back to scanning ``vars(shim)`` for a trailing-underscore attribute -- and the shim never set one in ``fit`` (the ``estimator_`` property is class-level, invisible to ``vars()``), so a *fitted* shim was wrongly reported as NOT fitted, breaking callers that gate on ``check_is_fitted`` (E19). ``fit`` now records ``_shim_fitted`` and the hook also probes the inner so externally-pre-fitted inners report correctly.
- ``predict_quantile(X, alpha=0.5)``: conditionally delegated (via ``available_if``) whenever the nested member exposes ``predict_quantile`` -- e.g. a quantile-capable ``CompositeTargetEstimator`` or a CatBoost/LightGBM/sklearn quantile regressor. ``X`` is routed through ``pre_pipeline.transform`` exactly like ``predict`` so quantile-stacking components keep their scaling. The method is hidden (``hasattr`` is False) when the inner cannot produce quantiles, so duck-typing probes stay honest (E19).
"""
from __future__ import annotations

import logging
from typing import Any

from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

try:
    # sklearn's canonical "does this estimator's fit accept <param>" check.
    from sklearn.utils.validation import has_fit_parameter as _sk_has_fit_parameter
except ImportError:  # pragma: no cover - sklearn always ships this
    _sk_has_fit_parameter = None

logger = logging.getLogger(__name__)


def _model_fit_accepts_sample_weight(model: Any) -> bool:
    """True when ``model.fit`` declares ``sample_weight`` (or accepts ``**kwargs``).

    Used to signature-GATE the optional ``sample_weight`` pass-through instead of
    the old catch-all ``except TypeError`` retry. The retry pattern was wrong: a
    ``TypeError`` raised DEEP inside a fit that *does* accept ``sample_weight`` (a
    bad dtype, a shape mismatch, a downstream library bug) was mis-attributed to
    "no sample_weight support" and the model was silently re-fit UNWEIGHTED --
    dropping the weighting the caller asked for AND hiding the real error. Gating
    on the declared signature swallows only the genuine "this fit has no
    sample_weight parameter" case and lets every real error propagate.
    """
    import inspect

    if _sk_has_fit_parameter is not None:
        try:
            return bool(_sk_has_fit_parameter(model, "sample_weight"))
        except Exception as e:  # pragma: no cover - defensive; fall through
            logger.debug("swallowed exception in post_shim.py: %s", e)
            pass
    fit_fn = getattr(model, "fit", None)
    if fit_fn is None:
        return False
    try:
        params = inspect.signature(fit_fn).parameters
    except (ValueError, TypeError):
        # Un-introspectable (builtin / C-extension): be permissive so we never
        # silently drop weighting; a real "no such param" TypeError then surfaces.
        return True
    if "sample_weight" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def subset_to_fit_columns(X: Any, pp: Any) -> Any:
    """Reorder/subset ``X`` to the exact column list ``pp`` was fit on, mirroring the predict-time guard in ``_predict_pre_pipeline._apply_extensions_pipeline``.

    ``pp.transform`` runs sklearn's strict feature-name check, which rejects ANY column not seen at fit time -- including a legitimate superset (e.g. row-wise extension columns added to the frame after ``pp`` was fit on an earlier snapshot). Without this, callers that pass the current (possibly wider) frame straight to ``pp.transform`` get "feature names should match those passed during fit" even though every fit-time column is present. Missing fit-time columns are left alone so ``pp.transform`` still raises loudly (never silently drop a column the model actually needs).
    """
    _fit_cols = getattr(pp, "feature_names_in_", None)
    if _fit_cols is None:
        return X
    _fit_list = [str(c) for c in _fit_cols]
    try:
        _cols = [str(c) for c in X.columns]
    except AttributeError:
        return X
    if _cols == _fit_list:
        return X
    if not (set(_fit_list) <= set(_cols)):
        return X
    if hasattr(X, "loc"):  # pandas
        return X.loc[:, _fit_list]
    if hasattr(X, "select"):  # polars
        return X.select(_fit_list)
    return X


def _inner_has_predict_quantile(shim: "PrePipelinePredictShim") -> bool:
    """``available_if`` predicate: expose ``shim.predict_quantile`` only when the
    nested member can actually produce quantiles.

    The nested member is the ``estimator_`` forwarded from a
    ``CompositeTargetEstimator`` (its inner quantile regressor) when one is
    present, otherwise ``self.model`` directly. Probing ``estimator_`` first
    matters because a quantile-capable ``CompositeTargetEstimator`` exposes
    ``predict_quantile`` on the wrapper, not on its raw inner.

    Keeping this conditional (rather than always defining the method) keeps
    ``hasattr(shim, "predict_quantile")`` honest: callers that duck-type the
    quantile capability (e.g. ``predict_quantile_ensemble``) get a truthful
    answer instead of a method that always raises.
    """
    model = shim.model
    if model is None:
        return False
    if hasattr(model, "predict_quantile"):
        return True
    inner = getattr(model, "estimator_", None)
    return inner is not None and hasattr(inner, "predict_quantile")


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
        """Apply the fitted ``pre_pipeline`` to ``X``, or pass it through unchanged when no pipeline is set."""
        if self.pre_pipeline is None:
            return X
        try:
            return self.pre_pipeline.transform(subset_to_fit_columns(X, self.pre_pipeline))
        except Exception as exc:
            # NEVER fall back to the untransformed X: a fitted pre_pipeline
            # (StandardScaler / SimpleImputer / ...) means the inner was
            # trained on SCALED features, so feeding it raw X silently
            # produces garbage whenever raw X is shape-compatible (the
            # common case for linear / MLP tiers inside the NNLS stack).
            # Raise loudly with context instead of masking the failure.
            raise RuntimeError(
                f"PrePipelinePredictShim({self.name}): pre_pipeline.transform "
                f"failed; refusing to feed untransformed X to the inner "
                f"(would predict on unscaled features). Original error: {exc!r}"
            ) from exc

    def fit(self, X: Any, y: Any, sample_weight: Any = None, **kwargs: Any) -> "PrePipelinePredictShim":
        """Fit ``model`` on ``pre_pipeline``-transformed ``X`` (the pipeline itself is treated as already-fit and never re-fit here)."""
        X_in = self._transform(X)
        # Signature-gate sample_weight rather than an ``except TypeError`` retry.
        # The retry mis-attributed a TypeError raised DEEP inside a weight-AWARE
        # inner fit (e.g. a downstream dtype/shape bug) to "no sample_weight
        # support" and silently re-fit UNWEIGHTED, dropping the weighting and
        # hiding the real error. ``has_fit_parameter`` is the canonical check and
        # also resolves metadata-routed / delegated inners; a genuine inner
        # TypeError now propagates instead of triggering an unweighted retry.
        if sample_weight is not None and _model_fit_accepts_sample_weight(self.model):
            self.model.fit(X_in, y, sample_weight=sample_weight, **kwargs)
        else:
            self.model.fit(X_in, y, **kwargs)
        self._shim_fitted = True
        return self

    def predict(self, X: Any) -> Any:
        """Predict on ``pre_pipeline``-transformed ``X``, mirroring the scaling applied during ``fit``."""
        # Tree-tier components carry no pre_pipeline (docstring above), so ``self.model`` -- not just
        # ``pre_pipeline`` -- can see a superset of its own fit-time columns (e.g. row-wise extension
        # columns added to the frame after this model was fit). Subset to the model's OWN
        # ``feature_names_in_`` too, same rationale as ``_transform``.
        return self.model.predict(subset_to_fit_columns(self._transform(X), self.model))

    @available_if(_inner_has_predict_quantile)
    def predict_quantile(self, X: Any, alpha: Any = 0.5) -> Any:
        """Quantile prediction routed through the fitted ``pre_pipeline``.

        Delegates to ``model.predict_quantile`` (the quantile-capable nested
        member -- typically a ``CompositeTargetEstimator`` or a CatBoost /
        LightGBM / sklearn quantile regressor) after transforming ``X`` so the
        quantile component sees the same SCALED features the inner was trained
        on -- identical routing to :meth:`predict`. ``alpha`` may be a scalar
        (returns ``(n_samples,)``) or array-like of K levels (returns
        ``(n_samples, K)``), matching the inner's contract.

        Only present when the nested member supports quantiles (gated by
        :func:`_inner_has_predict_quantile` via ``available_if``); otherwise the
        attribute does not exist so ``hasattr(shim, "predict_quantile")`` is
        ``False`` and duck-typing callers route around it.
        """
        return self.model.predict_quantile(subset_to_fit_columns(self._transform(X), self.model), alpha)

    def __sklearn_is_fitted__(self) -> bool:
        """Report the REAL fit state for ``sklearn.check_is_fitted``.

        The shim is fitted iff its inner ``model`` is fitted -- either because
        ``shim.fit`` ran (``_shim_fitted``) or because a pre-fitted inner was
        wrapped (the cross-target ensemble wraps already-trained components and
        calls ``predict`` without re-calling ``shim.fit``). Defining this hook
        stops ``check_is_fitted`` from falling back to scanning the always-present
        ``estimator_`` property, which made an UNFITTED shim look fitted (E19).
        """
        if getattr(self, "_shim_fitted", False):
            return True
        model = self.model
        if model is None:
            return False
        try:
            check_is_fitted(model)
            return True
        except NotFittedError:
            return False
        except (TypeError, ValueError):
            # Inner does not follow the sklearn fitted-attribute convention
            # (no trailing-underscore attrs, not a BaseEstimator). Fall back to
            # the explicit flag: if shim.fit never ran and the inner exposes no
            # detectable fitted state, treat the shim as unfitted.
            return False

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
        """Forward the inner ``estimator_`` (e.g. from a nested ``CompositeTargetEstimator``) so ``clone(model.estimator_)`` keeps working through the shim."""
        # OOF composite-path does ``clone(model.estimator_)`` when ``isinstance(model, CompositeTargetEstimator)``. Forwarding ``estimator_`` from the inner keeps that path working when the composite wrapper is nested inside a shim.
        inner = self.model
        return getattr(inner, "estimator_", inner)
