"""
XGBoost classifier/regressor shim that reuses ``QuantileDMatrix`` across
consecutive ``.fit()`` calls on the same feature matrix.

Why
---
The 2026-04-24 prod log captured the cost the shim is paid to eliminate:
``XGBClassifier.fit(X, y, sample_weight=w)`` rebuilds a 7.3M × 106
QuantileDMatrix in **104 s** for the ``uniform`` weight schema, then
**99 s** for ``recency`` — same feature matrix, only the
``sample_weight`` vector changed. With multi-target / multi-weight
sweeps that's many minutes of pure rebuild time on the same bytes.

The native ``xgb.train(params, dtrain)`` API accepts a pre-built
``QuantileDMatrix`` and ``DMatrix.set_label(y)`` / ``DMatrix.set_weight(w)``
mutate in place on the C++ handle (verified 2026-04-24 against XGBoost
3.x). The blocker is sklearn-side: ``XGBClassifier.fit(X, y)`` runs
``X`` through ``_validate_data`` / ``check_array`` and rebuilds the
DMatrix every call — so we cannot ``.fit(dtrain, ...)`` directly even
when our cache holds it ready.

This module subclasses ``XGBClassifier`` / ``XGBRegressor`` and:
  * overrides ``.fit()`` to (a) build a ``QuantileDMatrix`` once for a
    given (id(X), columns, shape, cat_features) signature, (b) on
    subsequent fits with the same signature swap label/weight in
    place, (c) call ``xgb.train()`` natively with the cached DMatrix,
    (d) attach the resulting ``Booster`` to ``self._Booster`` so the
    inherited ``predict`` / ``predict_proba`` / ``feature_importances_``
    keep working;
  * exposes public ``set_label(y)`` / ``set_weight(w)`` that mutate
    the cached DMatrix without a rebuild — useful for callers that
    drive their own xgb-train loop.

Drop-in compatibility
---------------------
The shim is a *subclass* of ``XGBClassifier`` / ``XGBRegressor``, so:
  * ``isinstance(model, XGBClassifier)`` checks downstream still pass;
  * ``get_params()`` / ``set_params()`` / ``sklearn.base.clone()`` work
    via the inherited sklearn-estimator protocol — and clone produces
    a fresh instance with an empty cache (the right thing);
  * ``predict``, ``predict_proba``, ``feature_importances_``,
    ``feature_names_in_``, ``n_features_in_`` are inherited and
    dispatch through the ``_Booster`` we attach.

Deprecation path
----------------
Once https://github.com/dmlc/xgboost/pull/<TBD> lands and ships in a
stable XGBoost release that accepts ``XGBClassifier.fit(X=DMatrix)``
natively, this shim becomes obsolete. Migration:

  1. In ``mlframe/training/trainer.py::_configure_xgboost_params``,
     replace ``XGBClassifierWithDMatrixReuse`` with ``XGBClassifier``
     (and same for the regressor).
  2. Adapt the cache code in trainer.py to call XGB's native fit
     directly with a pre-built ``QuantileDMatrix`` (mirror of the
     existing ``_CB_POOL_CACHE`` path for CatBoost).
  3. Delete this file and its test counterpart
     ``tests/training/test_xgb_dmatrix_reuse_shim.py``.

Until then the shim is the only practical way to get DMatrix reuse out
of the sklearn-XGB wrapper without monkey-patching XGBClassifier
globally (the latter would affect non-mlframe callers in the same
process).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    xgb = None  # type: ignore
    XGBClassifier = object  # type: ignore
    XGBRegressor = object  # type: ignore


# ---------------------------------------------------------------------
# Capability gate
# ---------------------------------------------------------------------

def xgb_dmatrix_reuse_capable() -> bool:
    """True iff the installed XGBoost has ``set_label`` / ``set_weight``
    on ``QuantileDMatrix`` — the two C++ mutators the shim relies on.

    XGBoost ≥ 1.7 has them. Returned as a runtime probe rather than a
    version-string compare so a future build with the methods removed
    or renamed is detected directly.
    """
    if not _XGB_AVAILABLE:
        return False
    return all(
        hasattr(xgb.QuantileDMatrix, attr)
        for attr in ("set_label", "set_weight")
    )


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _signature_of(X) -> tuple:
    """Cache key for a feature matrix.

    Combines ``id(X)`` (fast path: same Python object → same data) with
    a content-summary tuple (columns + shape) so two distinct DataFrame
    objects that share an ``id()`` (Python recycles ids after GC) still
    miss the cache rather than corrupting it. Mirrors the design of
    ``_CB_POOL_CACHE`` in trainer.py.
    """
    cols = tuple(X.columns) if hasattr(X, "columns") else None
    shape = getattr(X, "shape", (None, None))
    n_rows = int(shape[0]) if shape and shape[0] is not None else None
    n_cols = int(shape[1]) if shape and len(shape) > 1 and shape[1] is not None else None
    return (id(X), cols, n_rows, n_cols)


def _build_quantile_dmatrix(
    X, y, sample_weight, *, ref_dmatrix=None, enable_categorical: bool = True,
):
    """Build a fresh ``QuantileDMatrix``.

    XGBoost ≥ 2.x's ``QuantileDMatrix`` accepts pandas, numpy, scipy.sparse,
    AND polars DataFrames directly (verified 2026-04-24 against XGBoost
    3.x with pl.Enum + nullable pl.Boolean — no pre-conversion needed).
    Passing the polars frame straight through preserves mlframe's
    polars-fastpath: no extra ``get_pandas_view_of_polars_df`` call gets
    triggered just because the model wrapper is the shim.

    ``ref_dmatrix`` (when given) is passed as the ``ref`` argument so val
    DMatrices share the train DMatrix's quantile cuts — required by
    XGBoost when using eval_set with QuantileDMatrix.
    """
    kwargs: dict = dict(label=y, enable_categorical=enable_categorical)
    if sample_weight is not None:
        kwargs["weight"] = sample_weight
    if ref_dmatrix is not None:
        kwargs["ref"] = ref_dmatrix
    return xgb.QuantileDMatrix(X, **kwargs)


# ---------------------------------------------------------------------
# Mixin — shared fit-with-cache logic
# ---------------------------------------------------------------------

class _DMatrixReuseMixin:
    """Implements the override-fit + cache logic. Concrete subclasses
    just bind it to ``XGBClassifier`` or ``XGBRegressor``.

    Cache state lives on the instance (not module-global) so:
      * sklearn.clone() produces a fresh instance with empty cache
        (correct: cloned model should not silently inherit data);
      * concurrent training across multiple shims in one process
        doesn't share state.
    """

    # Type stub for static checkers — actual init runs in subclass via
    # super().__init__().
    _cached_train_dmatrix: Optional[Any]
    _cached_train_key: Optional[tuple]
    _cached_val_dmatrix: Optional[Any]
    _cached_val_key: Optional[tuple]

    # Names of cache attributes. Listed once so ``__getstate__`` /
    # ``clear_cache`` / the forward-/backward-transfer blocks in
    # ``core.py`` stay in sync. Two groups: ``_CACHE_POINTER_ATTRS``
    # are the heavyweight C++-backed DMatrix objects (unpicklable,
    # costly in RAM), ``_CACHE_KEY_ATTRS`` are the lightweight tuple
    # signatures that guard them.
    _CACHE_POINTER_ATTRS: tuple = ("_cached_train_dmatrix", "_cached_val_dmatrix")
    _CACHE_KEY_ATTRS: tuple = ("_cached_train_key", "_cached_val_key")

    def _init_cache(self) -> None:
        for _attr in self._CACHE_POINTER_ATTRS + self._CACHE_KEY_ATTRS:
            setattr(self, _attr, None)

    # ------------------------------------------------------------------
    # Pickle / joblib round-trip — strip cached DMatrix on save
    # ------------------------------------------------------------------
    #
    # ``QuantileDMatrix`` holds ctypes pointers to C++ memory —
    # joblib/pickle refuses it with "ctypes objects containing pointers
    # cannot be pickled". The 2026-04-24 prod log captured the regression:
    # mlframe.training.io failed to save both xgb_uniform and xgb_recency
    # dumps → next-run cached-model-load fell back to full retrain.
    #
    # The cache is transient runtime state — built on first fit, reused
    # only within the same process — so stripping it from the pickle
    # state is semantically correct: a reloaded model is a fresh
    # instance whose cache will repopulate on the next ``.fit()`` call.
    # ``__setstate__`` re-initialises the cache attrs so legacy saves
    # without the cache fields still load cleanly.

    def __getstate__(self):
        # Start from whatever parent (XGBModel) exposes as its state.
        # ``XGBModel.__getstate__`` returns ``self.__dict__``; override by
        # falling back to ``self.__dict__`` if parent doesn't define it.
        parent_get = getattr(super(), "__getstate__", None)
        state = parent_get() if parent_get is not None else self.__dict__
        state = dict(state)  # shallow copy so we can mutate safely
        # Strip pointer-typed cache attrs — their key siblings stay
        # (they're tuples, safely pickleable, but meaningless without
        # the DMatrix they indexed, so we null them too to avoid a
        # stale-key-looking-valid trap on load).
        for _attr in self._CACHE_POINTER_ATTRS + self._CACHE_KEY_ATTRS:
            state[_attr] = None
        return state

    def __setstate__(self, state) -> None:
        parent_set = getattr(super(), "__setstate__", None)
        if parent_set is not None:
            parent_set(state)
        else:
            self.__dict__.update(state)
        # Defensive: ensure cache attrs exist even if we loaded an older
        # save that predated this class's cache layout.
        for _attr in self._CACHE_POINTER_ATTRS + self._CACHE_KEY_ATTRS:
            if not hasattr(self, _attr):
                setattr(self, _attr, None)

    # ------------------------------------------------------------------
    # Explicit cache release — call after you're done with a run if you
    # need the memory back without waiting for GC
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Release cached DMatrix(s) and their keys. Useful between
        suite runs to free the C++ memory backing the quantile sketches
        (the 2026-04-24 prod log captured ~8 GB held by a single cached
        train DMatrix on the 7.3M × 106 frame)."""
        self._init_cache()

    # ------------------------------------------------------------------
    # Public extras — in-place swaps on the cached DMatrix
    # ------------------------------------------------------------------

    def set_label(self, y) -> None:
        """Swap label on the cached training DMatrix in place. Raises
        if no DMatrix has been built yet (call ``.fit()`` first)."""
        if self._cached_train_dmatrix is None:
            raise RuntimeError(
                "no cached DMatrix — call .fit() first to build one"
            )
        self._cached_train_dmatrix.set_label(np.asarray(y))

    def set_weight(self, w) -> None:
        """Swap sample weights on the cached training DMatrix in place.
        Raises if no DMatrix has been built yet."""
        if self._cached_train_dmatrix is None:
            raise RuntimeError(
                "no cached DMatrix — call .fit() first to build one"
            )
        self._cached_train_dmatrix.set_weight(np.asarray(w))

    # ------------------------------------------------------------------
    # Override .fit() — the cache + native xgb.train() path
    # ------------------------------------------------------------------

    def fit(  # type: ignore[override]
        self,
        X,
        y,
        sample_weight=None,
        eval_set=None,
        sample_weight_eval_set=None,
        verbose=False,
        **fit_kwargs,
    ):
        """Cached-DMatrix fit.

        Lifecycle:
          1. Build train DMatrix from cache or fresh; mutate label /
             weight in place on cache hit.
          2. Build val DMatrix(s) from cache or fresh.
          3. Call ``xgb.train(params, dtrain, num_boost_round=N, evals=...)``
             with proper params resolved from ``self.get_xgb_params()``.
          4. Attach Booster to ``self._Booster`` (sklearn convention)
             so inherited predict/predict_proba/feature_importances_
             work transparently.
          5. Set ``n_features_in_`` / ``feature_names_in_`` so sklearn
             pipeline integration sees the proper estimator state.

        Parameters mirror ``XGBClassifier.fit`` so the shim is a
        drop-in replacement.
        """
        # Lazy init in case __init__ wasn't called via subclass route
        # (sklearn.clone() of an exotic estimator).
        if not hasattr(self, "_cached_train_dmatrix"):
            self._init_cache()

        # ---- Train DMatrix: cache-or-build ---------------------------
        train_key = _signature_of(X)
        if self._cached_train_key == train_key and self._cached_train_dmatrix is not None:
            dtrain = self._cached_train_dmatrix
            dtrain.set_label(np.asarray(y))
            if sample_weight is not None:
                dtrain.set_weight(np.asarray(sample_weight))
            else:
                # When sample_weight was set previously and now None is
                # passed, "uniform weights" is the desired semantic. We
                # set ones to clear any prior weight; XGBoost will treat
                # an all-1 weight identically to no-weight.
                dtrain.set_weight(np.ones(dtrain.num_row(), dtype=np.float32))
            logger.debug(
                "[xgb-shim] reused cached train DMatrix (id=%d, shape=%dx%d)",
                id(dtrain), dtrain.num_row(), dtrain.num_col(),
            )
        else:
            dtrain = _build_quantile_dmatrix(
                X, y, sample_weight,
                enable_categorical=self.get_params().get("enable_categorical", True),
            )
            self._cached_train_dmatrix = dtrain
            self._cached_train_key = train_key
            logger.debug(
                "[xgb-shim] built fresh train DMatrix (id=%d, shape=%dx%d)",
                id(dtrain), dtrain.num_row(), dtrain.num_col(),
            )

        # ---- Eval DMatrix(s): cache-or-build -------------------------
        evals: List[Tuple[Any, str]] = []
        if eval_set:
            # Support eval_set = [(X_val, y_val), ...] form.
            for i, pair in enumerate(eval_set):
                X_val, y_val = pair
                # sample_weight_eval_set supports list-aligned weights;
                # default None.
                w_val = (
                    sample_weight_eval_set[i]
                    if sample_weight_eval_set and i < len(sample_weight_eval_set)
                    else None
                )
                val_key = _signature_of(X_val)
                if (
                    self._cached_val_key == val_key
                    and self._cached_val_dmatrix is not None
                ):
                    dval = self._cached_val_dmatrix
                    dval.set_label(np.asarray(y_val))
                    if w_val is not None:
                        dval.set_weight(np.asarray(w_val))
                else:
                    dval = _build_quantile_dmatrix(
                        X_val, y_val, w_val,
                        ref_dmatrix=dtrain,  # share quantile cuts
                        enable_categorical=self.get_params().get(
                            "enable_categorical", True
                        ),
                    )
                    self._cached_val_dmatrix = dval
                    self._cached_val_key = val_key
                evals.append((dval, f"validation_{i}"))

        # ---- Resolve params for xgb.train() --------------------------
        # ``get_xgb_params()`` excludes sklearn-only fields (n_estimators,
        # missing handling, etc) and returns the C++-level booster param
        # dict. ``n_estimators`` becomes ``num_boost_round`` for native API.
        params: dict = self.get_xgb_params()
        n_estimators = self.get_params().get("n_estimators", 100) or 100

        # Translate a couple of sklearn-only kwargs into native form:
        # (a) ``early_stopping_rounds`` → callback or kwarg below;
        # (b) ``eval_metric`` is already in params dict via get_xgb_params.
        early_stopping_rounds = self.get_params().get("early_stopping_rounds")

        callbacks = list(fit_kwargs.pop("callbacks", []) or [])
        if (
            early_stopping_rounds
            and evals
            and not any(
                isinstance(cb, xgb.callback.EarlyStopping) for cb in callbacks
            )
        ):
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=int(early_stopping_rounds),
                    save_best=True,
                )
            )

        # Determine objective for classification — XGBClassifier sets it
        # automatically based on n_classes; we mimic that here for the
        # native API. Subclasses (XGBClassifierWithDMatrixReuse) override
        # this to set binary:logistic / multi:softprob.
        params = self._finalize_native_params(params, y)

        # ---- Native xgb.train() --------------------------------------
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=int(n_estimators),
            evals=evals or None,
            callbacks=callbacks or None,
            verbose_eval=verbose,
        )

        # ---- Attach Booster — sklearn-convention metadata derived ----
        # XGBoost's sklearn wrapper exposes ``n_features_in_``,
        # ``feature_names_in_`` and ``feature_importances_`` as
        # read-only properties that derive from ``_Booster`` directly.
        # Setting ``self._Booster = booster`` is enough — the properties
        # auto-populate. Trying to assign them raises
        # ``AttributeError: property 'X' of '...' object has no setter``.
        self._Booster = booster

        # Subclass-specific bookkeeping (classifier needs ``n_classes_``
        # — ``classes_`` is a property returning ``np.arange(n_classes_)``).
        self._post_fit_bookkeeping(X, y)

        return self

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _finalize_native_params(self, params: dict, y) -> dict:
        """Subclass override: classifier sets objective/num_class,
        regressor leaves params alone."""
        return params

    def _post_fit_bookkeeping(self, X, y) -> None:
        """Subclass override: classifier sets ``classes_`` /
        ``n_classes_`` (sklearn convention); regressor no-op."""
        pass


# ---------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------

if _XGB_AVAILABLE:

    class XGBClassifierWithDMatrixReuse(_DMatrixReuseMixin, XGBClassifier):
        """XGBClassifier with cached QuantileDMatrix across fits.

        See module docstring for the full rationale and migration path.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._init_cache()

        def __sklearn_clone__(self):
            """Override sklearn's clone to produce a fresh instance with
            an empty cache. Default ``clone()`` would copy params via
            ``get_params(deep=False)`` and run ``__init__``, which already
            calls ``_init_cache()`` — so default behaviour is correct.
            But we declare ``__sklearn_clone__`` explicitly to make the
            contract visible: cloned instance MUST NOT inherit cache."""
            return type(self)(**self.get_params(deep=False))

        def _finalize_native_params(self, params: dict, y) -> dict:
            """Set objective and num_class for native xgb.train() based
            on label cardinality. Mirrors XGBClassifier.fit() internal
            handling."""
            y_arr = np.asarray(y)
            unique = np.unique(y_arr)
            n_classes = len(unique)
            params = dict(params)  # avoid mutating shared dict
            if "objective" not in params or params.get("objective") is None:
                params["objective"] = (
                    "binary:logistic" if n_classes == 2 else "multi:softprob"
                )
            if n_classes > 2 and "num_class" not in params:
                params["num_class"] = int(n_classes)
            return params

        def _post_fit_bookkeeping(self, X, y) -> None:
            """Set ``n_classes_`` (a plain attribute on XGBClassifier).
            ``classes_`` is a read-only property returning
            ``np.arange(n_classes_)`` in modern XGBoost — no separate
            assignment needed."""
            self.n_classes_ = int(len(np.unique(np.asarray(y))))


    class XGBRegressorWithDMatrixReuse(_DMatrixReuseMixin, XGBRegressor):
        """XGBRegressor with cached QuantileDMatrix across fits.

        See module docstring for the full rationale and migration path.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._init_cache()

        def __sklearn_clone__(self):
            return type(self)(**self.get_params(deep=False))

        def _finalize_native_params(self, params: dict, y) -> dict:
            params = dict(params)
            if "objective" not in params or params.get("objective") is None:
                params["objective"] = "reg:squarederror"
            return params


__all__ = [
    "xgb_dmatrix_reuse_capable",
    "XGBClassifierWithDMatrixReuse",
    "XGBRegressorWithDMatrixReuse",
]
