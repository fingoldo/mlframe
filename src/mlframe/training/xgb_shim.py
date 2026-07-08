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
from typing import Any

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


def _align_eval_categoricals(X_train, X_val):
    """Cast ``X_val`` categorical columns to the SAME categorical dtype as ``X_train``.

    XGBoost's ``enable_categorical`` path records the train DMatrix category
    universe; an eval / val frame whose categorical column carries a level (or a
    differently-ordered category list) absent from train trips
    ``cat_container.h: Found a category not in the training set`` during the
    early-stopping eval. Upstream prep usually shares the dtype, but a per-DF
    Enum / Categorical built independently per split (mixed-model polars tiers,
    pandas category columns) can diverge -- this is the last-line guarantee that
    the val frame's category universe matches train's before the DMatrix build.
    Returns ``X_val`` (possibly a new frame); no-op when dtypes already agree or
    the frame type is unsupported.
    """
    if X_train is None or X_val is None:
        return X_val
    # polars: cast each shared column to the train Enum/Categorical dtype; OOV
    # levels become null (strict=False), which XGBoost tolerates as missing.
    try:
        import polars as _pl
        if isinstance(X_train, _pl.DataFrame) and isinstance(X_val, _pl.DataFrame):
            _tr_schema = X_train.schema
            _exprs = []
            for _c, _dt in X_val.schema.items():
                _tdt = _tr_schema.get(_c)
                if _tdt is None or _dt == _tdt:
                    continue
                if isinstance(_tdt, _pl.Enum) or _tdt == _pl.Categorical:
                    _exprs.append(_pl.col(_c).cast(_pl.String).cast(_tdt, strict=False).alias(_c))
            if _exprs:
                X_val = X_val.with_columns(_exprs)
            return X_val
    except ImportError:
        pass
    # pandas: re-cast Categorical columns to train's categories list.
    try:
        import pandas as _pd
        if isinstance(X_train, _pd.DataFrame) and isinstance(X_val, _pd.DataFrame):
            for _c in X_val.columns:
                if _c not in X_train.columns:
                    continue
                _tdt = X_train[_c].dtype
                if isinstance(_tdt, _pd.CategoricalDtype) and isinstance(X_val[_c].dtype, _pd.CategoricalDtype):
                    if list(X_val[_c].cat.categories) != list(_tdt.categories):
                        X_val = X_val.assign(**{_c: X_val[_c].cat.set_categories(_tdt.categories)})
    except ImportError:
        pass
    return X_val


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
    return all(hasattr(xgb.QuantileDMatrix, attr) for attr in ("set_label", "set_weight"))


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _signature_of(X) -> tuple:
    """Cache key for a feature matrix; delegates to shared content fingerprint.

    See ``_dataset_cache_fingerprint.compute_signature`` for rationale --
    the same id(X) cache-key bug existed in xgb_shim, lgb_shim, CB train
    Pool cache, and CB val Pool cache; consolidating to one helper makes
    sklearn.clone() + .iloc-slicing safe across all four cache sites."""
    from ._dataset_cache_fingerprint import compute_signature
    return compute_signature(X)


# Module-level DMatrix cache keyed by content fingerprint. Survives
# ``sklearn.clone()`` (which creates a fresh shim instance with empty
# instance attrs) so OOF refits and per-target re-fits on identical
# data reuse the C++ DMatrix instead of rebuilding from scratch.
#
# Observed in prod: the CompositeCrossTargetEnsemble OOF refit
# helper calls ``clone(inner)`` per component, each clone has an empty
# instance cache, so QuantileDMatrix was rebuilt 4 times (20+s wasted
# per ensemble round). Cache is now module-level so the fingerprint
# actually delivers reuse across clones.
#
# Why an ``OrderedDict`` with LRU eviction (cap=8): each entry holds
# a QuantileDMatrix carrying ~200-500 MB of C++ memory on the 4M-row
# large-data scale. 8 entries is enough to keep train + val for 4 targets
# (raw + 3 composites) hot WITHOUT pinning the entire booster training
# memory budget. Eviction order is access-driven (``move_to_end`` on
# hit) so the most-recently-used train matrix never gets dropped.
#
# ``MLFRAME_XGB_CACHE_DISABLE=1`` env var forces bypass (testing only).
from collections import OrderedDict as _OrderedDict
import os as _os
import threading as _threading

_XGB_DMATRIX_CACHE: "_OrderedDict[tuple, Any]" = _OrderedDict()
_XGB_DMATRIX_CACHE_CAP: int = 8
_XGB_DMATRIX_CACHE_LOCK = _threading.Lock()


def _xgb_cache_disabled() -> bool:
    return bool(_os.environ.get("MLFRAME_XGB_CACHE_DISABLE"))


def _xgb_cache_get(key: tuple):
    if _xgb_cache_disabled() or key is None:
        return None
    with _XGB_DMATRIX_CACHE_LOCK:
        dm = _XGB_DMATRIX_CACHE.get(key)
        if dm is not None:
            _XGB_DMATRIX_CACHE.move_to_end(key)
        return dm


def _xgb_cache_put(key: tuple, dmatrix: Any) -> None:
    if _xgb_cache_disabled() or key is None or dmatrix is None:
        return
    with _XGB_DMATRIX_CACHE_LOCK:
        if key in _XGB_DMATRIX_CACHE:
            _XGB_DMATRIX_CACHE.move_to_end(key)
        _XGB_DMATRIX_CACHE[key] = dmatrix
        # LRU eviction.
        while len(_XGB_DMATRIX_CACHE) > _XGB_DMATRIX_CACHE_CAP:
            _XGB_DMATRIX_CACHE.popitem(last=False)


def _xgb_cache_clear() -> None:
    """Release all cached DMatrixes (call between long-running suite
    invocations to free C++ memory)."""
    with _XGB_DMATRIX_CACHE_LOCK:
        _XGB_DMATRIX_CACHE.clear()


def _build_quantile_dmatrix(
    X, y, sample_weight, *, ref_dmatrix=None, enable_categorical: bool = True,
    max_bin=None,
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

    ``max_bin`` MUST match the booster's ``max_bin`` param: XGBoost bakes
    the histogram bin count into the QuantileDMatrix's quantile cuts and
    raises ``Check failed: param.max_bin == init.max_bin (X vs. 256):
    Inconsistent max_bin`` at ``xgb.train`` time when the DMatrix was built
    with the default (256) but the model sets a different value. Threading
    the model's max_bin through here keeps the two consistent. ``None``
    leaves XGBoost at its default. Surfaced by fuzz (hgb_mlp_xgb combo with
    a non-default xgb max_bin)."""
    kwargs: dict = dict(label=y, enable_categorical=enable_categorical)
    if sample_weight is not None:
        kwargs["weight"] = sample_weight
    if ref_dmatrix is not None:
        kwargs["ref"] = ref_dmatrix
    if max_bin is not None:
        kwargs["max_bin"] = int(max_bin)
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
    _cached_train_dmatrix: Any | None
    _cached_train_key: tuple | None
    _cached_val_dmatrix: Any | None
    _cached_val_key: tuple | None

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
        # Wave 19 P1: stamp the xgboost version at save time so the load
        # side can detect skew (booster JSON in the unmodified __dict__
        # is library-version-sensitive across minor versions).
        try:
            import xgboost as _xgb
            state["_saved_xgb_version"] = str(getattr(_xgb, "__version__", "unknown"))
        except Exception:
            state["_saved_xgb_version"] = "unknown"
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
        # Wave 19 P1: compare saved xgboost version with the live one.
        _saved_ver = getattr(self, "_saved_xgb_version", None)
        if _saved_ver is not None and _saved_ver != "unknown":
            try:
                import xgboost as _xgb
                _live_ver = str(getattr(_xgb, "__version__", "unknown"))
            except Exception:
                _live_ver = "unknown"
            if _live_ver != "unknown" and _live_ver != _saved_ver:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "xgb_shim.__setstate__: xgboost version drift -- "
                    "artifact saved with xgboost==%s, loaded under "
                    "xgboost==%s. Booster internals may have changed "
                    "between versions; if predict() raises AttributeError "
                    "or returns suspicious values, retrain on the live "
                    "xgboost install.", _saved_ver, _live_ver,
                )

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
            raise RuntimeError("no cached DMatrix — call .fit() first to build one")
        self._cached_train_dmatrix.set_label(np.asarray(y))

    def set_weight(self, w) -> None:
        """Swap sample weights on the cached training DMatrix in place.
        Raises if no DMatrix has been built yet."""
        if self._cached_train_dmatrix is None:
            raise RuntimeError("no cached DMatrix — call .fit() first to build one")
        self._cached_train_dmatrix.set_weight(np.asarray(w))

    # ------------------------------------------------------------------
    # Override .fit() — the cache + native xgb.train() path
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        eval_set=None,
        sample_weight_eval_set=None,
        verbose=False,
        monotonic_decline_patience=7,
        capture_iteration_metrics=False,
        iteration_metrics_stride=1,
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
        # Lookup priority:
        #   1. Instance-level cache (same shim instance, same data) --
        #      fast path, no global-dict contention.
        #   2. Module-level cache (different shim, e.g. sklearn.clone() in
        #      composite-ensemble OOF refit, with identical content) --
        #      survives the clone-empties-cache problem.
        #   3. Fresh build, populate BOTH caches.
        # Fold max_bin into the cache key: a QuantileDMatrix bakes the bin
        # count into its quantile cuts, so two models with the same X but
        # different max_bin must NOT share a cached DMatrix (xgb.train would
        # raise "Inconsistent max_bin"). Without max_bin in the key the
        # module-level cross-clone cache hands a 256-bin matrix to a model
        # configured for a different bin count.
        _max_bin = self.get_params().get("max_bin")  # type: ignore[attr-defined]  # provided by the XGBModel sklearn base this mixin is combined with
        train_key = (_signature_of(X), _max_bin)
        dtrain = None
        _cache_source: str = "miss"
        if self._cached_train_key == train_key and self._cached_train_dmatrix is not None:
            dtrain = self._cached_train_dmatrix
            _cache_source = "instance"
        else:
            _global_hit = _xgb_cache_get(train_key)
            if _global_hit is not None:
                dtrain = _global_hit
                _cache_source = "module"
        if dtrain is not None:
            dtrain.set_label(np.asarray(y))
            if sample_weight is not None:
                dtrain.set_weight(np.asarray(sample_weight))
            else:
                # When sample_weight was set previously and now None is
                # passed, "uniform weights" is the desired semantic. We
                # set ones to clear any prior weight; XGBoost will treat
                # an all-1 weight identically to no-weight.
                dtrain.set_weight(np.ones(dtrain.num_row(), dtype=np.float32))
            # Promote module-hit to instance-level for the next call.
            self._cached_train_dmatrix = dtrain
            self._cached_train_key = train_key
            logger.debug(
                "[xgb-shim] reused %s cached train DMatrix (id=%d, shape=%dx%d)",
                _cache_source, id(dtrain), dtrain.num_row(), dtrain.num_col(),
            )
        else:
            dtrain = _build_quantile_dmatrix(
                X, y, sample_weight,
                enable_categorical=self.get_params().get("enable_categorical", True),  # type: ignore[attr-defined]  # provided by the XGBModel sklearn base this mixin is combined with
                max_bin=_max_bin,
            )
            self._cached_train_dmatrix = dtrain
            self._cached_train_key = train_key
            _xgb_cache_put(train_key, dtrain)
            logger.debug(
                "[xgb-shim] built fresh train DMatrix (id=%d, shape=%dx%d); "
                "stored in module cache for cross-clone reuse",
                id(dtrain), dtrain.num_row(), dtrain.num_col(),
            )

        # ---- Eval DMatrix(s): cache-or-build -------------------------
        evals: list[tuple[Any, str]] = []
        _iter_metrics_dval = None
        _iter_metrics_yval = None
        if eval_set:
            # Support eval_set = [(X_val, y_val), ...] form.
            for i, pair in enumerate(eval_set):
                X_val, y_val = pair
                # Align val categorical dtypes to train's BEFORE keying / building the
                # DMatrix so XGBoost's enable_categorical eval doesn't reject a val-only
                # category against the train category universe (cat_container.h).
                X_val = _align_eval_categoricals(X, X_val)
                # sample_weight_eval_set supports list-aligned weights;
                # default None.
                w_val = sample_weight_eval_set[i] if sample_weight_eval_set and i < len(sample_weight_eval_set) else None
                # Composite key (train_key, val_key) so we only reuse the
                # val DMatrix when BOTH the val content AND the originating
                # train content match. Different train content -> different
                # quantile cuts -> stale val bins would silently corrupt the
                # eval signal. Same lookup chain as train: instance -> module
                # -> fresh build. Without the module fallback (round-4 legacy)
                # sklearn.clone() in the composite-ensemble OOF refit produced
                # a fresh shim with empty instance cache and rebuilt the val
                # DMatrix every fit -- 5-10s wasted x4-5 refits per round.
                val_key = (_signature_of(X_val), train_key)
                dval = None
                _val_source: str = "miss"
                if self._cached_val_key == val_key and self._cached_val_dmatrix is not None:
                    dval = self._cached_val_dmatrix
                    _val_source = "instance"
                else:
                    _global_val_hit = _xgb_cache_get(val_key)
                    if _global_val_hit is not None:
                        dval = _global_val_hit
                        _val_source = "module"
                if dval is not None:
                    dval.set_label(np.asarray(y_val))
                    if w_val is not None:
                        dval.set_weight(np.asarray(w_val))
                    else:
                        dval.set_weight(np.ones(dval.num_row(), dtype=np.float32))
                    self._cached_val_dmatrix = dval
                    self._cached_val_key = val_key
                    logger.debug(
                        "[xgb-shim] reused %s cached val DMatrix (id=%d, shape=%dx%d)",
                        _val_source, id(dval), dval.num_row(), dval.num_col(),
                    )
                else:
                    dval = _build_quantile_dmatrix(
                        X_val, y_val, w_val,
                        ref_dmatrix=dtrain,
                        enable_categorical=self.get_params().get("enable_categorical", True),  # type: ignore[attr-defined]  # provided by the XGBModel sklearn base this mixin is combined with
                        max_bin=_max_bin,
                    )
                    self._cached_val_dmatrix = dval
                    self._cached_val_key = val_key
                    _xgb_cache_put(val_key, dval)
                    logger.debug(
                        "[xgb-shim] built fresh val DMatrix (id=%d, shape=%dx%d); "
                        "stored in module cache for cross-clone reuse",
                        id(dval), dval.num_row(), dval.num_col(),
                    )
                evals.append((dval, f"validation_{i}"))
                if i == 0:
                    _iter_metrics_dval = dval
                    _iter_metrics_yval = np.asarray(y_val)

        # ---- Resolve params for xgb.train() --------------------------
        # ``get_xgb_params()`` excludes sklearn-only fields (n_estimators,
        # missing handling, etc) and returns the C++-level booster param
        # dict. ``n_estimators`` becomes ``num_boost_round`` for native API.
        params: dict = self.get_xgb_params()  # type: ignore[attr-defined]  # provided by the XGBModel sklearn base this mixin is combined with
        # Wave 14 P1 (re-opened 2026-05-20): pre-fix `or 100` silently
        # rewrote n_estimators=0 (legitimate xgboost intent: "construct
        # an untrained booster; predict returns base score") to 100.
        # The shim ran 100 boost rounds when the user wanted 0.
        _n_est_raw = self.get_params().get("n_estimators", 100)  # type: ignore[attr-defined]  # provided by the XGBModel sklearn base this mixin is combined with
        n_estimators = 100 if _n_est_raw is None else _n_est_raw

        # Translate a couple of sklearn-only kwargs into native form:
        # (a) ``early_stopping_rounds`` → callback or kwarg below;
        # (b) ``eval_metric`` is already in params dict via get_xgb_params.
        early_stopping_rounds = self.get_params().get("early_stopping_rounds")  # type: ignore[attr-defined]  # provided by the XGBModel sklearn base this mixin is combined with

        callbacks = list(fit_kwargs.pop("callbacks", []) or [])
        if early_stopping_rounds and evals and not any(isinstance(cb, xgb.callback.EarlyStopping) for cb in callbacks):
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=int(early_stopping_rounds),
                    save_best=True,
                )
            )

        # Default-on monotonic strict-decline overfitting stop, COMPLEMENTARY to native
        # early_stopping_rounds: stop once the first eval set's metric strictly worsens for
        # ``monotonic_decline_patience`` consecutive rounds since the best. Only attached when
        # eval sets exist, the caller hasn't already supplied one, and it isn't disabled (None).
        if monotonic_decline_patience is not None and evals and not any(getattr(cb, "_is_mlframe_monotonic_decline", False) for cb in callbacks):
            from .callbacks.monotonic_decline import _make_xgb_monotonic_callback
            _mono_cb = _make_xgb_monotonic_callback(patience=monotonic_decline_patience)
            if _mono_cb is not None:
                callbacks.append(_mono_cb)

        # Per-iteration full-metric-suite capture (meta-learning / HPO-from-early-observation). The val DMatrix can be
        # re-predicted directly via iteration_range; capture every stride round + always the final round.
        _iter_metrics_cb = None
        if capture_iteration_metrics and _iter_metrics_dval is not None and _iter_metrics_yval is not None:
            from sklearn.base import is_classifier as _sk_is_classifier
            from .callbacks.iteration_metrics import make_xgb_iteration_metrics_callback
            _ncls = getattr(self, "n_classes_", None) or getattr(self, "_n_classes", None)
            if _ncls is None and _sk_is_classifier(self):
                _ncls = int(np.unique(np.asarray(y)).shape[0])
            if not _sk_is_classifier(self):
                _tt = "regression"
            elif _ncls is not None and int(_ncls) > 2:
                _tt = "multiclass_classification"
            else:
                _tt = "binary_classification"
            _iter_metrics_cb = make_xgb_iteration_metrics_callback(
                _iter_metrics_dval, _iter_metrics_yval, _tt,
                stride=int(iteration_metrics_stride), n_classes=int(_ncls) if _ncls else None,
            )
            if _iter_metrics_cb is not None:
                callbacks.append(_iter_metrics_cb)

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

        if _iter_metrics_cb is not None:
            self.iteration_metrics_ = _iter_metrics_cb.iteration_metrics_

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
            # 2-D y is multilabel/multioutput; ``np.unique`` would flatten
            # across all label columns and report a bogus class count. The
            # XGBClassifier multilabel path uses a different objective
            # (binary per-output) and reaches a different code path - if a
            # 2-D y arrives here, it's a routing bug upstream.
            if y_arr.ndim > 1:
                raise ValueError(
                    f"_finalize_native_params expects 1-D y for "
                    f"single-output classification; got y of shape "
                    f"{y_arr.shape}. Multilabel/multi-output must use the "
                    f"per-output classifier path, not this code path."
                )
            unique = np.unique(y_arr)
            n_classes = len(unique)
            params = dict(params)  # avoid mutating shared dict
            if "objective" not in params or params.get("objective") is None:
                params["objective"] = "binary:logistic" if n_classes == 2 else "multi:softprob"
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
