"""Multi-output composite for a VECTOR target ``y`` of shape ``(n, K)``.

The single-output ``CompositeTargetEstimator`` learns one transformed target and
inverts it. Many real targets are vector-valued -- a basket of correlated series,
a multi-horizon forecast, a set of related rates -- where EACH output column has
its OWN dominant affine base (its own lag feature, its own anchor). Fitting a
shared transform / shared base across all columns wastes the structure: column
``k`` wants ``base_k``, not a global base.

``CompositeMultiOutputEstimator`` fits ONE independent
:class:`CompositeTargetEstimator` per output column. Each per-column wrapper gets
its own transform spec + its own base column(s), so column ``k`` is modelled as
``y_k = f_k^{-1}(inner_k(X))`` with ``inner_k`` trained on the ``k``-th
transformed target. ``predict`` stacks the per-column point predictions back into
an ``(n, K)`` matrix.

Why a dedicated estimator (vs sklearn ``MultiOutputRegressor``)
--------------------------------------------------------------
sklearn's ``MultiOutputRegressor`` clones ONE estimator prototype per column and
fits each on the SAME ``X`` against ``y[:, k]``. It cannot give column ``k`` a
different ``base_column`` / ``transform_name`` -- the whole point of the composite
target. This estimator carries a per-column spec map (or a shared default) so each
output gets the affine base that actually dominates it.

Public surface
--------------
- :class:`CompositeMultiOutputEstimator` -- sklearn ``MultiOutputMixin`` wrapper.
- :func:`make_per_column_specs` -- small helper to build the per-column spec list
  from a shared default + per-column overrides.

Design choices (mirroring the other composite wrappers)
-------------------------------------------------------
- Per-column wrappers are looked up / built from plain dict specs at ``fit`` time,
  never stored as captured closures, so ``sklearn.clone`` + pickle stay clean.
- Strict train-only fit: each per-column ``CompositeTargetEstimator.fit`` computes
  its transform params on the training rows only; no predict-time re-fit.
- Memory: never copies the (possibly 100+ GB) ``X`` frame. Each per-column wrapper
  receives the caller's ``X`` reference unchanged -- the per-column ``base`` is a
  narrow ``_extract_base``-style column pull inside the inner wrapper, not a frame
  copy. ``y`` is column-sliced to a 1-D ndarray per column (cheap, ``y`` is the
  target not the feature frame).
- NaN-safe: a fully-NaN output column (no finite training rows) is recorded as a
  failed column; ``predict`` returns the column's ``y_train_median`` (or NaN when
  no finite rows existed at all) for it instead of crashing the whole vector.

Out of scope
------------
- Cross-output coupling (chained / regressor-chain style where column ``k`` sees
  column ``k-1``'s prediction). This estimator fits each column INDEPENDENTLY; a
  chained variant is a separate future extension.
- Classification / GLM vector targets (use the per-family single-output wrappers
  column-wise if needed).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin, clone

from .estimator import CompositeTargetEstimator

logger = logging.getLogger(__name__)


# A per-column spec is a plain dict of CompositeTargetEstimator __init__ kwargs
# (transform_name / base_column / base_columns / group_column / fallback_predict /
# drop_invalid_rows / ...). Kept as a dict -- not a captured wrapper instance --
# so clone / pickle stay closure-free.
ColumnSpec = Dict[str, Any]


def make_per_column_specs(
    n_outputs: int,
    shared_spec: Optional[ColumnSpec] = None,
    per_column: Optional[Mapping[int, ColumnSpec]] = None,
    base_columns_map: Optional[Mapping[int, Union[str, Sequence[str]]]] = None,
) -> List[ColumnSpec]:
    """Build a length-``n_outputs`` list of per-column composite specs.

    Each output column ``k`` gets ``shared_spec`` (a dict of
    :class:`CompositeTargetEstimator` ``__init__`` kwargs) merged with the
    per-column override ``per_column[k]`` (override wins on key collisions). A
    ``base_columns_map[k]`` entry is a convenience shortcut that sets the
    column's base -- a single column name lands in ``base_column``, a sequence
    lands in ``base_columns`` (multi-base path).

    Parameters
    ----------
    n_outputs
        Number ``K`` of target columns. The returned list has this length.
    shared_spec
        Default kwargs applied to every column. ``None`` -> empty dict (each
        column then relies on its per-column / base-map override or the
        ``CompositeTargetEstimator`` defaults).
    per_column
        ``{k: spec}`` overrides for specific columns. Keys are column indices.
    base_columns_map
        ``{k: base}`` shortcut. ``base`` is a column name (-> ``base_column``)
        or a sequence of names (-> ``base_columns``). Takes priority over a
        ``base_column`` / ``base_columns`` already present in the merged spec
        for that column (it is the most specific declaration).

    Returns
    -------
    list of dict
        One spec dict per output column, ready to feed into a
        :class:`CompositeTargetEstimator`.
    """
    if n_outputs <= 0:
        raise ValueError(
            f"make_per_column_specs: n_outputs must be positive, got {n_outputs}."
        )
    shared = dict(shared_spec) if shared_spec else {}
    per_column = per_column or {}
    base_columns_map = base_columns_map or {}
    specs: List[ColumnSpec] = []
    for k in range(n_outputs):
        spec: ColumnSpec = dict(shared)
        if k in per_column:
            spec.update(per_column[k])
        if k in base_columns_map:
            base = base_columns_map[k]
            # A bare string is a single base column; a sequence is a multi-base
            # set. Clear the other slot so a shared default doesn't leak through.
            if isinstance(base, str):
                spec["base_column"] = base
                spec.pop("base_columns", None)
            else:
                spec["base_columns"] = tuple(base)
                spec.pop("base_column", None)
        specs.append(spec)
    return specs


class CompositeMultiOutputEstimator(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Fit one :class:`CompositeTargetEstimator` per output column of a vector target.

    Models a target ``y`` of shape ``(n, K)`` by fitting ``K`` INDEPENDENT
    composite wrappers -- one per column -- each with its own transform + base.
    ``predict`` returns an ``(n, K)`` matrix of the per-column inverse-transformed
    point predictions.

    Parameters
    ----------
    base_estimator
        The inner regressor prototype shared across columns (cloned once per
        column at fit). A per-column spec may override it via a
        ``base_estimator`` key. Must not be ``None`` unless every per-column spec
        supplies its own.
    column_specs
        Either:

        * a single dict applied to EVERY column (the shared spec), or
        * a list of length ``K`` of per-column dicts (built e.g. by
          :func:`make_per_column_specs`).

        Each dict is a set of :class:`CompositeTargetEstimator` ``__init__``
        kwargs (``transform_name``, ``base_column`` / ``base_columns``,
        ``group_column``, ``fallback_predict``, ``drop_invalid_rows``, ...). When
        ``None`` every column uses the ``CompositeTargetEstimator`` defaults
        (``transform="diff"``) on top of ``base_estimator``.
    base_columns_map
        Convenience ``{k: base}`` map (string -> ``base_column``, sequence ->
        ``base_columns``) applied on top of the shared / per-column spec. Lets a
        caller pass ONE shared transform spec plus a per-column base mapping
        without hand-building a full list. Ignored when ``column_specs`` is a
        list (the list is taken as already fully specified) UNLESS a column's
        spec lacks any base, in which case the map fills it in.
    skip_failed_columns
        When True (default), a per-column fit that raises (e.g. a fully-NaN /
        degenerate column) is recorded as a failed column and predicted as a
        constant fallback (its finite-``y`` median, or NaN) instead of aborting
        the whole vector fit. When False the first column failure re-raises.

    Attributes set at fit
    ---------------------
    estimators_ : list[CompositeTargetEstimator | None]
        The fitted per-column wrappers (``None`` for a failed column when
        ``skip_failed_columns``).
    n_outputs_ : int
        The number ``K`` of target columns.
    column_fallbacks_ : list[float]
        Per-column constant used by ``predict`` for a failed column.
    failed_columns_ : list[int]
        Indices of columns whose fit failed and fell back to a constant.
    feature_names_in_ : list | None
        Column names seen at fit (best effort: pandas / polars).
    """

    def __init__(
        self,
        base_estimator: Any = None,
        column_specs: Union[ColumnSpec, Sequence[ColumnSpec], None] = None,
        base_columns_map: Optional[Mapping[int, Union[str, Sequence[str]]]] = None,
        skip_failed_columns: bool = True,
    ) -> None:
        self.base_estimator = base_estimator
        self.column_specs = column_specs
        self.base_columns_map = base_columns_map
        self.skip_failed_columns = skip_failed_columns

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_2d_targets(y: Any) -> np.ndarray:
        """Coerce ``y`` to a float64 ``(n, K)`` ndarray without copying the
        feature frame. A 1-D ``y`` becomes ``(n, 1)`` -- a degenerate single
        output, still a valid vector target."""
        arr = np.asarray(y)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(
                f"CompositeMultiOutputEstimator: y must be 1-D or 2-D, got "
                f"ndim={arr.ndim}."
            )
        return arr.astype(np.float64, copy=False)

    def _resolve_specs(self, n_outputs: int) -> List[ColumnSpec]:
        """Normalise ``column_specs`` (+ ``base_columns_map``) to a length-K
        list of per-column kwarg dicts. A dict is broadcast to every column; a
        sequence is validated against ``K``; ``None`` yields empty specs."""
        cs = self.column_specs
        if cs is None:
            specs: List[ColumnSpec] = [dict() for _ in range(n_outputs)]
        elif isinstance(cs, Mapping):
            # A single shared spec dict -> broadcast to every column. Copy per
            # column so a later base-map fill on one column never mutates another.
            specs = [dict(cs) for _ in range(n_outputs)]
        else:
            specs = [dict(s) for s in cs]
            if len(specs) != n_outputs:
                raise ValueError(
                    f"CompositeMultiOutputEstimator: column_specs has "
                    f"{len(specs)} entries but y has {n_outputs} columns."
                )
        bcm = self.base_columns_map or {}
        for k in range(n_outputs):
            if k in bcm:
                base = bcm[k]
                # base-map is the most specific base declaration: it wins, except
                # when column_specs was a list AND already declares a base -- then
                # the list's explicit base stays (the map only FILLS missing bases
                # for list specs). For a shared/None spec the map always sets it.
                already = specs[k].get("base_column") or specs[k].get("base_columns")
                list_specified = not isinstance(cs, Mapping) and cs is not None
                if not (list_specified and already):
                    if isinstance(base, str):
                        specs[k]["base_column"] = base
                        specs[k].pop("base_columns", None)
                    else:
                        specs[k]["base_columns"] = tuple(base)
                        specs[k].pop("base_column", None)
        return specs

    def _build_column_estimator(self, spec: ColumnSpec) -> CompositeTargetEstimator:
        """Construct an unfitted :class:`CompositeTargetEstimator` from a spec
        dict, defaulting its inner ``base_estimator`` to this estimator's shared
        prototype (cloned) when the spec doesn't override it."""
        spec = dict(spec)
        inner = spec.pop("base_estimator", None)
        if inner is None:
            inner = self.base_estimator
        if inner is None:
            raise ValueError(
                "CompositeMultiOutputEstimator: no base_estimator -- supply one "
                "on the estimator or in every per-column spec."
            )
        return CompositeTargetEstimator(base_estimator=clone(inner), **spec)

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[np.ndarray] = None,
        **fit_kwargs: Any,
    ) -> "CompositeMultiOutputEstimator":
        """Fit one composite wrapper per output column of ``y``.

        ``X`` is passed by reference to every per-column wrapper (never copied);
        ``y`` is column-sliced per output. ``sample_weight`` / ``**fit_kwargs``
        forward to each per-column ``fit``. A column whose fit raises is recorded
        as a failed column when ``skip_failed_columns`` (else re-raises).
        """
        y2d = self._to_2d_targets(y)
        n_outputs = y2d.shape[1]
        specs = self._resolve_specs(n_outputs)

        estimators: List[Optional[CompositeTargetEstimator]] = []
        fallbacks: List[float] = []
        failed: List[int] = []
        for k in range(n_outputs):
            y_k = y2d[:, k]
            finite = np.isfinite(y_k)
            fallback = float(np.median(y_k[finite])) if bool(finite.any()) else float("nan")
            try:
                est = self._build_column_estimator(specs[k])
                est.fit(X, y_k, sample_weight=sample_weight, **fit_kwargs)
                estimators.append(est)
            except Exception as err:
                if not self.skip_failed_columns:
                    raise
                logger.warning(
                    "CompositeMultiOutputEstimator: column %d fit failed (%r); "
                    "predicting it as the constant fallback %.6g.",
                    k, err, fallback,
                )
                estimators.append(None)
                failed.append(k)
            fallbacks.append(fallback)

        self.estimators_ = estimators
        self.n_outputs_ = n_outputs
        self.column_fallbacks_ = fallbacks
        self.failed_columns_ = failed
        # Best-effort feature names (pandas / polars). An ndarray X lacks
        # ``.columns`` (AttributeError) -- not an error, just no names.
        try:
            self.feature_names_in_ = list(X.columns)
        except AttributeError:
            self.feature_names_in_ = None
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Return the ``(n, K)`` matrix of per-column inverse-transformed point
        predictions. A failed column is filled with its constant fallback."""
        if not hasattr(self, "estimators_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError(
                "CompositeMultiOutputEstimator: call fit before predict."
            )
        # Determine row count from the first successful per-column predict, or
        # fall back to len(X) when every column failed (all-constant output).
        cols: List[Optional[np.ndarray]] = []
        n_rows: Optional[int] = None
        for k, est in enumerate(self.estimators_):
            if est is None:
                cols.append(None)
                continue
            pred_k = np.asarray(est.predict(X), dtype=np.float64).reshape(-1)
            cols.append(pred_k)
            if n_rows is None:
                n_rows = pred_k.shape[0]
        if n_rows is None:
            n_rows = self._infer_n_rows(X)
        out = np.empty((n_rows, self.n_outputs_), dtype=np.float64)
        for k in range(self.n_outputs_):
            col = cols[k]
            if col is None:
                out[:, k] = self.column_fallbacks_[k]
            else:
                out[:, k] = col
        return out

    @staticmethod
    def _infer_n_rows(X: Any) -> int:
        """Row count of ``X`` without materialising it: ``len`` works for
        pandas / polars / ndarray; ``shape[0]`` is the ndarray fallback."""
        try:
            return len(X)
        except TypeError:
            return int(np.asarray(X).shape[0])

    @property
    def n_features_in_(self) -> Optional[int]:
        """Feature count seen at fit (from ``feature_names_in_`` when available,
        else the first fitted column's inner count)."""
        names = getattr(self, "feature_names_in_", None)
        if names is not None:
            return len(names)
        for est in getattr(self, "estimators_", []) or []:
            if est is not None:
                n = getattr(est, "n_features_in_", None)
                if n is not None:
                    return int(n)
        return None
