"""Native sklearn Pipeline / TransformedTargetRegressor integration for composite targets.

Two integration shapes are offered, both built on the existing
:class:`CompositeTargetEstimator` and the frozen transform registry:

1. :func:`make_composite_regressor` -- a thin convenience factory that
   returns a ready :class:`CompositeTargetEstimator`, usable directly as
   the FINAL step of a :class:`sklearn.pipeline.Pipeline`. The base column
   is read out of ``X`` by the wrapper at fit / predict, so the only thing
   the caller supplies is an inner regressor + the transform name + the
   base column name. This is the recommended path: the wrapper handles the
   forward / inverse loop, domain filtering, y-clip, and per-row base
   extraction natively, and survives ``sklearn.clone`` because the
   transform is looked up by NAME (never stored as a per-instance closure).

2. :class:`CompositeTargetTransformer` -- a :class:`sklearn.base.TransformerMixin`
   that exposes the target's forward / inverse transform as a
   ``func`` / ``inverse_func`` pair so a composite target can be plugged
   into :class:`sklearn.compose.TransformedTargetRegressor`. Use this when
   you specifically want sklearn's TTR plumbing (e.g. an existing pipeline
   already built around TTR) rather than the wrapper. The transformer
   captures the per-row ``base`` array at fit time (from a configured
   column of ``X``, or from an explicit ``base`` array) and replays it
   positionally on every forward / inverse, so it is exact ONLY when the
   row order / count is preserved between fit and the TTR forward (which
   TransformedTargetRegressor guarantees -- it transforms the SAME ``y``
   it was handed). For UNARY transforms (``requires_base=False``:
   ``cbrt_y`` / ``log_y`` / ``yeo_johnson_y`` / ``quantile_normal_y`` /
   ``winsor_y``) no base is needed and the transformer is row-order
   independent.

Why both. The wrapper (path 1) is strictly more capable -- it does per-row
domain checks, train-envelope clip, grouped / multi-base transforms, and
inner-model attribute delegation. The TTR transformer (path 2) is a
compatibility shim for codebases standardised on
``TransformedTargetRegressor``; it deliberately exposes ONLY the
target-side forward / inverse (no feature transform), and is a no-op on the
feature matrix (``transform(X) -> X`` with a :meth:`get_feature_names_out`
passthrough), so dropping it into a feature ``Pipeline`` is harmless.

Design notes
------------
- No frame copies on the hot path: the only read of ``X`` is the narrow
  single-column ``_extract_base`` pull (one ndarray, never ``to_pandas``
  on the whole frame), matching the wrapper's RAM discipline.
- ``func`` / ``inverse_func`` are exposed as BOUND METHODS (picklable;
  resolve the captured base + fitted params off ``self``) rather than
  closures, so a TTR holding the transformer round-trips through pickle.
- ``sklearn.clone`` is honoured: every constructor argument is stored
  verbatim as an attribute with no transformation, and the transform is a
  name (string), never a callable, so a cloned transformer re-derives the
  same forward / inverse from the registry.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .estimator import (
    CompositeTargetEstimator,
    _extract_base,
    _extract_base_matrix,
    _to_1d_numpy,
)
from .transforms import get_transform

logger = logging.getLogger(__name__)


__all__ = [
    "make_composite_regressor",
    "CompositeTargetTransformer",
]


# ----------------------------------------------------------------------
# Path 1: convenience factory -> a ready Pipeline final step.
# ----------------------------------------------------------------------

def make_composite_regressor(
    inner: Any,
    transform_name: str = "diff",
    base_column: str = "",
    *,
    base_columns: Optional[Sequence[str]] = None,
    group_column: Optional[str] = None,
    fallback_predict: str = "y_train_median",
    drop_invalid_rows: bool = True,
    **estimator_kwargs: Any,
) -> CompositeTargetEstimator:
    """Return a ready :class:`CompositeTargetEstimator` usable as a Pipeline final step.

    A thin, discoverable factory over the wrapper constructor: callers who
    only want "wrap this regressor in transform ``T`` over base column ``b``"
    should not have to remember the wrapper's full parameter list. The
    returned estimator is a fully-formed sklearn regressor -- it can be the
    final step of a :class:`sklearn.pipeline.Pipeline`, the ``estimator`` of
    a grid search, or wrapped again by an ensemble.

    Parameters
    ----------
    inner
        Any sklearn-compatible regressor (``fit(X, y)`` / ``predict(X)``).
        Cloned at the wrapper's ``fit`` time; the prototype stays unfitted.
    transform_name
        One of :func:`list_transforms`. The forward / inverse applied to ``y``.
    base_column
        Column of ``X`` carrying the base feature. Required for base
        transforms (``requires_base=True``); leave ``""`` for unary
        transforms (``cbrt_y`` etc.). When ``base_columns`` is given it wins.
    base_columns
        Multi-column base path (``linear_residual_multi``). Overrides
        ``base_column`` when set.
    group_column
        Column carrying group labels for grouped transforms
        (``linear_residual_grouped``). None for ungrouped transforms.
    fallback_predict
        Per-row inverse-failure strategy: ``"y_train_median"`` (default) or
        ``"nan"``.
    drop_invalid_rows
        Drop fit-time domain-violating rows (default True) vs raise.
    **estimator_kwargs
        Forwarded verbatim to :class:`CompositeTargetEstimator` (e.g.
        ``monotone_constraints=``, ``auto_variance_stabilise=``,
        ``recurrence_continuation=``). Lets advanced knobs through without
        re-listing them here.

    Returns
    -------
    CompositeTargetEstimator
        Unfitted, clone-safe, ready as a Pipeline final step.

    Raises
    ------
    UnknownTransformError
        If ``transform_name`` is not in the registry (validated eagerly so
        a typo fails at construction, not deep inside a Pipeline ``fit``).
    ValueError
        If a base transform is requested with no base column / columns.
    """
    # Eager registry lookup: surface a bad transform name at construction
    # time (clear traceback) rather than deferring to the wrapper's fit deep
    # inside a Pipeline.
    transform = get_transform(transform_name)
    has_base = bool(base_column) or bool(base_columns)
    if transform.requires_base and not has_base:
        raise ValueError(
            f"make_composite_regressor: transform {transform_name!r} requires a base "
            "column (requires_base=True) but neither base_column nor base_columns was "
            "given. Pass base_column='<col>' (or base_columns=[...] for the multi-base "
            "transforms)."
        )
    return CompositeTargetEstimator(
        base_estimator=inner,
        transform_name=transform_name,
        base_column=base_column,
        base_columns=base_columns,
        group_column=group_column,
        fallback_predict=fallback_predict,
        drop_invalid_rows=drop_invalid_rows,
        **estimator_kwargs,
    )


# ----------------------------------------------------------------------
# Path 2: TransformedTargetRegressor-compatible target transformer.
# ----------------------------------------------------------------------

class CompositeTargetTransformer(TransformerMixin, BaseEstimator):
    """Expose a composite target transform as a TTR-compatible transformer.

    Wraps one registry transform so it can be handed to
    :class:`sklearn.compose.TransformedTargetRegressor` via the
    ``transformer=`` argument, OR used standalone with its
    :attr:`func` / :attr:`inverse_func` bound methods via the
    ``func=`` / ``inverse_func=`` arguments of TTR.

    The transformer fits the transform's parameters on the target it is
    handed and captures the per-row ``base`` array (for base transforms)
    so the forward / inverse can be applied with ONLY ``y`` thereafter --
    exactly the calling convention TTR uses (it never passes ``X`` to the
    target transformer). Because the base is captured positionally, the
    forward / inverse are exact only while the row order / count match the
    fit; TTR honours this (it forwards the same ``y`` it received). For
    unary transforms (``requires_base=False``) no base is captured and the
    transformer is row-order independent.

    Parameters
    ----------
    transform_name
        One of :func:`list_transforms`.
    base_column
        Column of ``X`` to pull the per-row base from at fit (base
        transforms only). Mutually-usable with ``base`` (an explicit array);
        if both are absent for a base transform, :meth:`fit` raises.
    base_columns
        Multi-column base path (``linear_residual_multi``).
    feature_names_out
        Optional feature-name passthrough for :meth:`get_feature_names_out`
        when this transformer is (harmlessly) placed in a FEATURE pipeline.
        ``None`` echoes the input names (identity feature transform).

    Notes
    -----
    This is a TARGET transformer: :meth:`transform` / :meth:`inverse_transform`
    operate on ``y``-shaped arrays, not on the feature matrix. As a feature
    transformer it is the identity (``transform`` returns its input
    unchanged when handed a 2-D feature matrix is NOT supported -- use it
    only on the target side / via TTR).
    """

    def __init__(
        self,
        transform_name: str = "diff",
        base_column: str = "",
        *,
        base_columns: Optional[Sequence[str]] = None,
        base: Optional[np.ndarray] = None,
        feature_names_out: Optional[Sequence[str]] = None,
    ) -> None:
        self.transform_name = transform_name
        self.base_column = base_column
        self.base_columns = base_columns
        self.base = base
        self.feature_names_out = feature_names_out

    # ---- sklearn fit/transform contract (target side) ----------------

    def fit(self, y: Any, X: Any = None) -> "CompositeTargetTransformer":
        """Fit the transform params on ``y``; capture the base for replay.

        Signature is ``fit(y, X=None)`` rather than the feature-side
        ``fit(X, y)`` because this is a TARGET transformer: TTR calls
        ``transformer.fit(y)`` with the target as the first positional. To
        also support capturing a base column from a feature frame, pass the
        frame as the second positional (``fit(y, X)``) or set ``base=`` /
        ``base_column=`` ahead of time.
        """
        transform = get_transform(self.transform_name)
        y_arr = _to_1d_numpy(y)

        base_arr: Optional[np.ndarray]
        if not transform.requires_base:
            base_arr = None
        else:
            base_arr = self._resolve_base(X, n=y_arr.shape[0])

        # Fit transform-specific params on the TRAIN target only (the y handed here).
        if base_arr is None:
            params = transform.fit(y_arr, np.empty(0))
        else:
            params = transform.fit(y_arr, base_arr)

        self.transform_name_ = self.transform_name
        self.fitted_params_ = dict(params)
        self.base_ = base_arr
        self.n_features_in_seen_ = y_arr.shape[0]
        return self

    def transform(self, y: Any) -> np.ndarray:
        """Forward the target: ``y -> T`` (the scale the inner model learns)."""
        transform = get_transform(self._fitted_transform_name())
        y_arr = _to_1d_numpy(y)
        base_arr = self._replay_base(y_arr.shape[0])
        out = transform.forward(y_arr, base_arr, self.fitted_params_)
        return np.asarray(out, dtype=np.float64).reshape(-1)

    def inverse_transform(self, t: Any) -> np.ndarray:
        """Invert the target: ``T_hat -> y_hat`` (back to the original scale)."""
        transform = get_transform(self._fitted_transform_name())
        t_arr = _to_1d_numpy(t)
        base_arr = self._replay_base(t_arr.shape[0])
        out = transform.inverse(t_arr, base_arr, self.fitted_params_)
        return np.asarray(out, dtype=np.float64).reshape(-1)

    # ---- func / inverse_func pair for TransformedTargetRegressor -----
    # Exposed as bound-method PROPERTIES so a TTR built with
    # ``func=tr.func, inverse_func=tr.inverse_func`` is picklable (the bound
    # method resolves the captured base + params off ``self``; no closure).

    @property
    def func(self):
        """Bound :meth:`transform` -- pass as TTR ``func=``."""
        return self.transform

    @property
    def inverse_func(self):
        """Bound :meth:`inverse_transform` -- pass as TTR ``inverse_func=``."""
        return self.inverse_transform

    # ---- feature-name passthrough ------------------------------------

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Passthrough feature names (identity feature transform).

        Returns the configured ``feature_names_out`` if set, else echoes
        ``input_features``, else a single synthetic name. This lets the
        transformer sit (harmlessly, as a no-op on features) inside a
        ``ColumnTransformer`` / ``Pipeline`` that introspects feature names.
        """
        if self.feature_names_out is not None:
            return np.asarray(list(self.feature_names_out), dtype=object)
        if input_features is not None:
            return np.asarray(list(input_features), dtype=object)
        return np.asarray([f"{self.transform_name}_target"], dtype=object)

    # ---- internals ----------------------------------------------------

    def _fitted_transform_name(self) -> str:
        # Prefer the fit-frozen name (clone safety) but fall back to the
        # constructor arg so a not-yet-fitted introspection still resolves.
        return getattr(self, "transform_name_", self.transform_name)

    def _resolve_base(self, X: Any, n: int) -> np.ndarray:
        """Pull the per-row base array from explicit ``base`` or a column of X."""
        if self.base is not None:
            arr = _to_1d_numpy(self.base)
            if arr.shape[0] != n:
                raise ValueError(
                    f"CompositeTargetTransformer: base length {arr.shape[0]} != y length {n}."
                )
            return arr
        if self.base_columns:
            if X is None:
                raise ValueError(
                    "CompositeTargetTransformer: base_columns set but no X passed to fit; "
                    "call fit(y, X) or pass base=<array>."
                )
            mat = _extract_base_matrix(X, list(self.base_columns))
            # Multi-base fit stores its own K-col base; collapse to the first column
            # for the single-array replay contract (the multi-base transform reads
            # the full matrix via params, not via the replayed base, for inverse).
            return mat[:, 0]
        if self.base_column:
            if X is None:
                raise ValueError(
                    "CompositeTargetTransformer: base_column set but no X passed to fit; "
                    "call fit(y, X) or pass base=<array>."
                )
            return _extract_base(X, self.base_column)
        raise ValueError(
            f"CompositeTargetTransformer: transform {self.transform_name!r} requires a base "
            "but neither base, base_column, nor base_columns was provided. Set one of them, "
            "or use a unary transform (requires_base=False)."
        )

    def _replay_base(self, n: int) -> np.ndarray:
        """Return the captured base for a forward / inverse over ``n`` rows.

        Exact when ``n`` matches the fit-time row count (the TTR contract).
        A length mismatch means the caller broke the positional-replay
        invariant (reordered / subset the target); we raise rather than
        silently mis-align base to y.
        """
        base_arr = getattr(self, "base_", None)
        if base_arr is None:
            # Unary transform: forward / inverse ignore base. Hand a finite
            # placeholder of the right length so domain checks see no NaN.
            return np.zeros(n, dtype=np.float64)
        if base_arr.shape[0] != n:
            raise ValueError(
                f"CompositeTargetTransformer: captured base has {base_arr.shape[0]} rows but "
                f"the forward/inverse was asked for {n} rows. The base is replayed positionally "
                "(no X is passed to a target transform), so the row order / count must match fit. "
                "For variable-length inference use CompositeTargetEstimator (path 1) instead -- it "
                "extracts base from X per call."
            )
        return base_arr

    # sklearn tag: this transformer is stateless w.r.t. the FEATURE matrix
    # (it transforms the target), so it must not be subjected to the
    # feature-X validation sklearn runs on standard transformers.
    def _more_tags(self) -> dict[str, Any]:
        return {"stateless": True, "requires_y": True, "no_validation": True}
