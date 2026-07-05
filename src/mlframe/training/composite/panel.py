"""Panel / longitudinal composite -- remove the per-entity fixed effect, model within-entity dynamics.

Repeated-measures data (``entity x time``: customers across months, sensors across cycles,
stores across days) carries two very different kinds of signal:

    * the BETWEEN-entity level -- the entity's own baseline (a high-spending customer, a
      hot sensor, a flagship store). A model that also receives the entity id can spend
      most of its capacity simply memorising these per-entity offsets.
    * the WITHIN-entity dynamics -- how the target moves around that entity's own baseline
      as the features move. This is usually the actually-interesting, transferable signal.

``CompositePanelEstimator`` is the panel analogue of the grouped residual composite. It
applies the econometric *within transformation* (a.k.a. fixed-effects / demeaning): it
subtracts each entity's own mean target, fits the inner composite on the demeaned signal
``y_within = y - entity_mean(y)``, and at predict adds the entity mean back. The inner
therefore never wastes capacity re-learning the entity level the id already explains; it
focuses entirely on within-entity dynamics.

Train-only / held-out discipline (CLAUDE.md): the per-entity means, the shrinkage toward
the global mean, and the global mean itself are ALL computed on the fit data only. At
predict the stored means are looked up -- no target is touched. An entity unseen at fit
falls back to the global mean, so predictions stay finite for new entities.

Shrinkage: a small entity (few fit rows) has a noisy sample mean. We shrink it toward the
global mean with the classic empirical-Bayes weight ``n / (n + alpha)`` -- an entity with
many rows keeps its own mean; an entity with one row is pulled most of the way to global.
This prevents the within transform from over-fitting per-entity noise into the offset.

sklearn-compatible (fit / predict / get_params / clone). The entity id is supplied either
as a column name inside ``X`` (``entity_column=``) or as a separate ``entity_id=`` array to
``fit`` / ``predict``. The inner estimator is passed by config (never a closure), so clone /
pickle stay clean. The entity column, when present in ``X``, is dropped from the matrix the
inner sees -- otherwise the inner could re-absorb the entity level we just demeaned out.

Out of scope: time-varying entity effects (random-slopes), cross-sectional dependence, and
serial-correlation-robust inference -- this estimator targets the standard one-way
fixed-effects decomposition of an additive continuous ``y``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)


def _is_polars_df(x: Any) -> bool:
    """Explicit isinstance check (no duck-typing on ``to_pandas``)."""
    try:
        import polars as pl

        return isinstance(x, pl.DataFrame)
    except Exception:
        return False


def _to_1d_numpy(y: Any) -> np.ndarray:
    return np.asarray(y, dtype=np.float64).ravel()


def _n_features(X: Any) -> int:
    """Robust column count for pandas / polars / 2-D ndarray.

    ``getattr(X, 'shape', (0, 0))[1]`` returns 0 for any carrier whose ``shape`` is 1-D or absent (e.g. a list of dicts);
    prefer the named-column length when present so ``n_features_in_`` reflects the real feature count for the sklearn contract.
    """
    cols = getattr(X, "columns", None)
    if cols is not None:
        try:
            return len(cols)
        except TypeError:
            pass
    shape = getattr(X, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1])
    return 0


def _extract_entity(X: Any, entity_column: str) -> np.ndarray:
    """Pull the entity-id column from X (pandas / polars) as a 1-D object ndarray.

    Narrow one-column read only -- never copies the whole frame. Ids are kept as objects so
    int / str / categorical entity labels all hash uniformly. Raises ``KeyError`` with an
    actionable message if the column was dropped (e.g. by feature selection).
    """
    if _is_polars_df(X):
        if entity_column not in X.columns:
            raise KeyError(
                f"CompositePanelEstimator: entity column '{entity_column}' missing from X. "
                "If feature selection (MRMR/RFECV) is dropping it, add entity_column to "
                "forced_keep_columns in the feature selection config."
            )
        return X.get_column(entity_column).to_numpy().astype(object)
    if hasattr(X, "columns"):
        if entity_column not in X.columns:
            raise KeyError(f"CompositePanelEstimator: entity column '{entity_column}' missing from X.")
        return np.asarray(X[entity_column], dtype=object).ravel()
    raise TypeError("CompositePanelEstimator requires a pandas/polars DataFrame X with named columns.")


def _drop_entity_column(X: Any, entity_column: str) -> Any:
    """Return X without the entity column (lightweight drop, no full-frame copy).

    The inner composite must NOT see the raw entity id as a feature -- otherwise it re-learns
    the entity level we demeaned out. polars / pandas ``drop`` share the underlying buffers.
    """
    if _is_polars_df(X):
        return X.drop(entity_column) if entity_column in X.columns else X
    if hasattr(X, "columns"):
        return X.drop(columns=[entity_column]) if entity_column in X.columns else X
    return X


def _resolve_entity_ids(X: Any, entity_column: Optional[str], entity_id: Optional[Any], n: int) -> np.ndarray:
    """Resolve entity ids from the explicit ``entity_id`` arg or the ``entity_column`` in X."""
    if entity_id is not None:
        ids = np.asarray(entity_id, dtype=object).ravel()
        if ids.shape[0] != n:
            raise ValueError(f"CompositePanelEstimator: entity_id length {ids.shape[0]} != n_rows {n}.")
        return ids
    if entity_column is not None:
        return _extract_entity(X, entity_column)
    raise ValueError("CompositePanelEstimator: supply the entity id either via entity_column (a column " "in X) or via the entity_id= argument to fit/predict.")


class CompositePanelEstimator(BaseEstimator, RegressorMixin):
    """Panel composite: demean the per-entity fixed effect, model within-entity dynamics.

    Parameters
    ----------
    inner_estimator : sklearn regressor
        Models the within-entity signal ``y - entity_mean(y)`` on the (entity-dropped) X.
        Passed by config; cloned at fit, never captured as a closure.
    entity_column : str, optional
        Name of the entity-id column inside ``X``. Pulled with a narrow one-column read and
        dropped from the inner's feature matrix. Mutually-usable with ``entity_id=``; if both
        are given the explicit ``entity_id=`` array wins.
    shrinkage_alpha : float, default 10.0
        Empirical-Bayes shrinkage strength. An entity with ``n`` fit rows uses offset
        ``w * entity_mean + (1 - w) * global_mean`` with ``w = n / (n + alpha)``. ``alpha=0``
        disables shrinkage (raw per-entity means); larger ``alpha`` pulls small entities
        harder toward the global mean. Must be >= 0.

    Attributes (post-fit)
    ---------------------
    global_mean_ : float
        Train-only global mean of y; the fallback offset for entities unseen at fit.
    entity_offsets_ : dict
        Map ``entity_id -> shrunken offset`` (train-only). Looked up at predict; never the
        raw target.
    entity_counts_ : dict
        Map ``entity_id -> fit row count`` (diagnostics; drives the shrinkage weight).
    inner_ : fitted inner estimator on the demeaned target.
    """

    def __init__(
        self,
        inner_estimator: Any,
        entity_column: Optional[str] = None,
        shrinkage_alpha: float = 10.0,
    ) -> None:
        self.inner_estimator = inner_estimator
        self.entity_column = entity_column
        self.shrinkage_alpha = shrinkage_alpha

    # -- internals ---------------------------------------------------------
    def _compute_offsets(self, ids: np.ndarray, y: np.ndarray) -> None:
        """Train-only per-entity shrunken offsets toward the global mean.

        ``offset(e) = w_e * mean_e + (1 - w_e) * global``, ``w_e = n_e / (n_e + alpha)``.
        Computed in one vectorised groupby pass over the fit data -- no target leakage at
        predict because only these reduced offsets are stored, never the rows themselves.
        """
        alpha = float(self.shrinkage_alpha)
        if alpha < 0:
            raise ValueError(f"CompositePanelEstimator: shrinkage_alpha must be >= 0; got {alpha}.")
        global_mean = float(y.mean())
        # Group sums + counts via a single argsort-free dict reduction (entity counts are
        # typically << n, so a Python dict reduction is cheaper than building a full index).
        sums: dict = {}
        counts: dict = {}
        for e, val in zip(ids.tolist(), y.tolist()):
            if e in sums:
                sums[e] += val
                counts[e] += 1
            else:
                sums[e] = val
                counts[e] = 1
        offsets: dict = {}
        for e, c in counts.items():
            mean_e = sums[e] / c
            w = c / (c + alpha) if (c + alpha) > 0 else 0.0
            offsets[e] = w * mean_e + (1.0 - w) * global_mean
        self.global_mean_ = global_mean
        self.entity_offsets_ = offsets
        self.entity_counts_ = counts

    def _lookup_offsets(self, ids: np.ndarray) -> np.ndarray:
        """Per-row offset: stored entity offset, global-mean fallback for unseen entities."""
        gm = self.global_mean_
        off = self.entity_offsets_
        return np.fromiter((off.get(e, gm) for e in ids.tolist()), dtype=np.float64, count=ids.shape[0])

    # -- sklearn API -------------------------------------------------------
    def fit(self, X: Any, y: Any, entity_id: Optional[Any] = None) -> "CompositePanelEstimator":
        y_arr = _to_1d_numpy(y)
        n = y_arr.shape[0]
        ids = _resolve_entity_ids(X, self.entity_column, entity_id, n)

        # Train-only offsets (the within-entity means + shrinkage + global mean).
        self._compute_offsets(ids, y_arr)
        offsets = self._lookup_offsets(ids)

        # Within transformation: the inner sees only the demeaned (within-entity) target.
        y_within = y_arr - offsets

        X_inner = _drop_entity_column(X, self.entity_column) if self.entity_column is not None else X
        self.inner_ = clone(self.inner_estimator)
        self.inner_.fit(X_inner, y_within)

        self.n_features_in_ = _n_features(X_inner)
        return self

    def predict(self, X: Any, entity_id: Optional[Any] = None) -> np.ndarray:
        if not hasattr(self, "inner_"):
            raise NotFittedError("CompositePanelEstimator: call fit before predict.")
        n = getattr(X, "shape", (0, 0))[0]
        ids = _resolve_entity_ids(X, self.entity_column, entity_id, n)
        offsets = self._lookup_offsets(ids)  # global-mean fallback for unseen entities

        X_inner = _drop_entity_column(X, self.entity_column) if self.entity_column is not None else X
        inner_pred = _to_1d_numpy(self.inner_.predict(X_inner))
        # Add the entity level back: within-entity prediction + the (shrunken) fixed effect.
        return inner_pred + offsets
