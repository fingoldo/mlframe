"""
LightGBM classifier/regressor shim that reuses ``lightgbm.Dataset`` across
consecutive ``.fit()`` calls on the same feature matrix.

Why
---
Mirror of ``mlframe.training.xgb_shim``. The 2026-04-24 prod log captured
the same multi-build cost on the LightGBM side: ``LGBMClassifier.fit(X, y,
sample_weight=w)`` constructs a fresh ``lightgbm.Dataset`` for every
weight-schema swap on the same feature matrix, and the ``Dataset.construct()``
call (binning + histogram setup) is the expensive part. With multi-target
/ multi-weight sweeps that's repeated work on identical bytes.

The native ``lightgbm.train(params, train_set)`` API accepts a pre-built
``Dataset`` and ``Dataset.set_label(y)`` / ``Dataset.set_weight(w)`` mutate
the binned dataset in place (verified 2026-05-08 against LightGBM 4.x).
The blocker is sklearn-side: ``LGBMClassifier.fit(X, y)`` validates ``X``
through ``_validate_data`` and rebuilds the Dataset every call -- so we
cannot ``.fit(train_set, ...)`` directly even when our cache holds it
ready. Our upstream PR for ``Dataset`` support in the sklearn interface
is pending; until it lands we route through this shim.

This module subclasses ``LGBMClassifier`` / ``LGBMRegressor`` and:
  * overrides ``.fit()`` to (a) build a ``Dataset`` once for a given
    (id(X), columns, shape, categorical_feature) signature, (b) on
    subsequent fits with the same signature swap label/weight in place,
    (c) call ``lightgbm.train()`` natively with the cached Dataset,
    (d) attach the resulting ``Booster`` to ``self._Booster`` so the
    inherited ``predict`` / ``predict_proba`` / ``feature_importances_``
    keep working;
  * exposes public ``set_label(y)`` / ``set_weight(w)`` that mutate
    the cached Dataset without a rebuild -- useful for callers that
    drive their own lgb-train loop.

Drop-in compatibility
---------------------
The shim is a *subclass* of ``LGBMClassifier`` / ``LGBMRegressor``, so:
  * ``isinstance(model, LGBMClassifier)`` checks downstream still pass;
  * ``get_params()`` / ``set_params()`` / ``sklearn.base.clone()`` work
    via the inherited sklearn-estimator protocol -- and clone produces
    a fresh instance with an empty cache (the right thing);
  * ``predict``, ``predict_proba``, ``feature_importances_``,
    ``feature_names_in_``, ``n_features_in_`` are inherited and
    dispatch through the ``_Booster`` we attach.

Deprecation path
----------------
Once https://github.com/microsoft/LightGBM/pull/<TBD> lands and ships in
a stable LightGBM release that accepts ``LGBMClassifier.fit(X=Dataset)``
natively, this shim becomes obsolete. Migration:

  1. In ``mlframe/training/trainer.py::_configure_lightgbm_params``,
     replace ``LGBMClassifierWithDatasetReuse`` with ``LGBMClassifier``
     (and same for the regressor).
  2. Adapt the cache code in trainer.py to call LGB's native fit
     directly with a pre-built ``Dataset`` (mirror of the existing
     ``_CB_POOL_CACHE`` path for CatBoost).
  3. Delete this file and its test counterpart
     ``tests/training/test_lgb_dataset_reuse_shim.py``.

Until then the shim is the only practical way to get Dataset reuse out
of the sklearn-LGBM wrapper without monkey-patching LGBMClassifier
globally (the latter would affect non-mlframe callers in the same
process).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
    from lightgbm.sklearn import _LGBMLabelEncoder, _EvalFunctionWrapper

    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    lgb = None  # type: ignore
    LGBMClassifier = object  # type: ignore
    LGBMRegressor = object  # type: ignore
    _LGBMLabelEncoder = None  # type: ignore
    _EvalFunctionWrapper = None  # type: ignore


try:
    import polars as pl
    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False
    pl = None  # type: ignore


def _maybe_bridge_polars_to_pandas(X):
    """Route a polars frame through the Arrow split-blocks bridge so LightGBM sees a proper pandas frame with ``pd.Categorical`` preserved.

    The default ``lgb.Dataset(data=polars_df)`` path falls through ``__array__`` and materialises X to a numpy object/float matrix, losing the Categorical
    codes that LightGBM needs to dispatch the native categorical split path. ``get_pandas_view_of_polars_df`` is the project's Arrow split-blocks bridge --
    zero-copy for numeric / boolean / string columns and ~32x faster than bare ``.to_pandas()`` on Categorical-heavy frames (benchmarked in
    ``profiling/bench_polars_to_pandas.py``). Non-polars inputs pass through untouched.
    """
    if not _PL_AVAILABLE or not isinstance(X, pl.DataFrame):
        return X
    try:
        from .utils import get_pandas_view_of_polars_df
        return get_pandas_view_of_polars_df(X)
    except ImportError:
        # Fallback: pyarrow extension arrays preserve Categorical dtype but are slower than the split-blocks bridge.
        return X.to_pandas(use_pyarrow_extension_array=True)


# ---------------------------------------------------------------------
# Capability gate
# ---------------------------------------------------------------------

def lgb_dataset_reuse_capable() -> bool:
    """True iff the installed LightGBM has ``set_label`` / ``set_weight``
    on ``Dataset`` -- the two C++ mutators the shim relies on.

    LightGBM >= 3.x has them. Returned as a runtime probe rather than a
    version-string compare so a future build with the methods removed
    or renamed is detected directly.
    """
    if not _LGB_AVAILABLE:
        return False
    return all(
        hasattr(lgb.Dataset, attr)
        for attr in ("set_label", "set_weight")
    )


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _signature_of(X, categorical_feature=None) -> tuple:
    """Cache key for a feature matrix; delegates to shared content fingerprint.

    Combines the cross-shim content fingerprint (cols + shape + 3-row
    sample hash) with the LGB-specific ``categorical_feature`` so cat-list
    changes invalidate the cached Dataset (LightGBM bakes cat-feature
    binning at construct time; reusing the cached dataset with a
    different cat-list would silently produce wrong splits).

    Pre-2026-05-23 the key included ``id(X)`` -- defeated by
    ``sklearn.clone()`` + ``.iloc`` slicing in composite-ensemble OOF
    refit, same as the XGB shim and CB Pool caches. Now content-based.
    """
    from ._dataset_cache_fingerprint import compute_signature
    if isinstance(categorical_feature, list):
        cat_key = tuple(categorical_feature)
    else:
        cat_key = categorical_feature  # 'auto' or None passes through
    return compute_signature(X, extra=(cat_key,))


def _is_pair_item(obj: Any) -> bool:
    """True when ``obj`` looks like an X/y array (DataFrame / ndarray / polars / Series), i.e. one element of a bare (X, y[, w]) bundle."""
    if isinstance(obj, (str, bytes)) or isinstance(obj, (list, tuple)):
        return False
    return bool(
        hasattr(obj, "shape") or hasattr(obj, "columns")
        or hasattr(obj, "iloc") or hasattr(obj, "dtypes")
    )


def normalize_eval_set(eval_set: Any) -> Optional[List[tuple]]:
    """Canonicalize an LGBM ``eval_set`` to a list-of-tuples once at the fit boundary.

    Accepts and returns:
      * ``None`` -> ``None``
      * a bare ``(X, y)`` / ``(X, y, w)`` tuple -> ``[(X, y[, w])]``
      * a bare ``[X, y]`` / ``[X, y, w]`` list (array-like items) -> ``[(X, y[, w])]``
      * a proper list of ``(X, y[, w])`` pairs -> the same list (items coerced to tuples)

    The bare 2/3-element forms are ambiguous with a genuine list of feature matrices;
    the disambiguator is ``_is_pair_item`` (first element is array-like) plus a guard
    that a real (X, y) bundle has y strictly lower-rank than X. Downstream code can
    then assume a clean list-of-tuples and assert that invariant.
    """
    if eval_set is None:
        return None

    # Bare tuple form: (X, y) or (X, y, w) where the first element is array-like.
    if isinstance(eval_set, tuple):
        if len(eval_set) in (2, 3) and _is_pair_item(eval_set[0]):
            return [tuple(eval_set)]
        # Otherwise treat as an iterable of pairs.
        return [tuple(p) for p in eval_set]

    if isinstance(eval_set, list):
        # Bare list form [X, y] / [X, y, w]: first element is array-like, not a pair.
        if len(eval_set) in (2, 3) and _is_pair_item(eval_set[0]) and not isinstance(eval_set[1], (list, tuple)):
            _first, _second = eval_set[0], eval_set[1]
            _first_shape = getattr(_first, "shape", None)
            _second_shape = getattr(_second, "shape", None)
            # A genuine list of feature matrices has both elements 2-D with matching ncols;
            # a real (X, y) bundle has y of rank 1 (or fewer cols). Only wrap the latter.
            _is_list_of_matrices = (
                _second_shape is not None and len(_second_shape) >= 2
                and _first_shape is not None and len(_first_shape) >= 2
                and _second_shape[1] == _first_shape[1]
            )
            if not _is_list_of_matrices:
                return [tuple(eval_set)]
        # Proper list of pairs.
        return [tuple(p) for p in eval_set]

    raise TypeError(f"lgb_shim: unsupported eval_set type {type(eval_set).__name__}; expected None, tuple, or list.")


def _build_dataset(
    X, y, sample_weight,
    *,
    reference=None,
    categorical_feature="auto",
    feature_name="auto",
    init_score=None,
    params=None,
):
    """Build a fresh ``lightgbm.Dataset``.

    LightGBM's ``Dataset`` accepts pandas, numpy, scipy.sparse, pyarrow
    Tables, and lists of sequences. Polars DataFrames are NOT accepted
    natively: they fall through ``__array__`` and lose Categorical codes,
    so the shim converts them up front via ``_maybe_bridge_polars_to_pandas``
    (Arrow split-blocks bridge) before calling this helper.

    ``reference`` (when given) is passed so val Datasets share the train
    Dataset's bin mapping -- required by LightGBM for any non-train
    Dataset to score consistently.

    ``free_raw_data=False`` keeps the source data referenced by the
    Dataset, so:
      (a) ``set_label`` / ``set_weight`` continue to work after the
          first fit (LightGBM keeps the binned representation, but
          some metadata paths still touch the raw data);
      (b) val datasets that ``reference=`` this train dataset don't
          lose their binning context if we rebuild train.
    """
    return lgb.Dataset(
        data=X,
        label=y,
        weight=sample_weight,
        reference=reference,
        init_score=init_score,
        categorical_feature=categorical_feature,
        feature_name=feature_name,
        params=params,
        free_raw_data=False,
    )


# ---------------------------------------------------------------------
# Mixin -- shared fit-with-cache logic
# ---------------------------------------------------------------------

class _DatasetReuseMixin:
    """Implements the override-fit + cache logic. Concrete subclasses
    just bind it to ``LGBMClassifier`` or ``LGBMRegressor``.

    Cache state lives on the instance (not module-global) so:
      * sklearn.clone() produces a fresh instance with empty cache
        (correct: cloned model should not silently inherit data);
      * concurrent training across multiple shims in one process
        doesn't share state.
    """

    # Type stubs for static checkers -- actual init runs in subclass via
    # super().__init__().
    _cached_train_dataset: Any | None
    _cached_train_key: tuple | None
    _cached_val_dataset: Any | None
    _cached_val_key: tuple | None

    # Names of cache attributes. Listed once so ``__getstate__`` /
    # ``clear_cache`` / the forward-/backward-transfer blocks in
    # ``core.py`` stay in sync. Two groups: ``_CACHE_POINTER_ATTRS``
    # are the heavyweight C++-backed Dataset objects (unpicklable when
    # constructed, costly in RAM), ``_CACHE_KEY_ATTRS`` are the
    # lightweight tuple signatures that guard them.
    _CACHE_POINTER_ATTRS: tuple = ("_cached_train_dataset", "_cached_val_dataset")
    _CACHE_KEY_ATTRS: tuple = ("_cached_train_key", "_cached_val_key")

    def _init_cache(self) -> None:
        for _attr in self._CACHE_POINTER_ATTRS + self._CACHE_KEY_ATTRS:
            setattr(self, _attr, None)

    # ------------------------------------------------------------------
    # Pickle / joblib round-trip -- strip cached Dataset on save
    # ------------------------------------------------------------------
    #
    # A constructed ``lightgbm.Dataset`` holds a ctypes pointer to the
    # C++ Dataset handle (``self.handle``) -- joblib/pickle refuses it
    # with "ctypes objects containing pointers cannot be pickled" once
    # the dataset is materialised. Same prod regression mode as XGB.
    #
    # The cache is transient runtime state -- built on first fit, reused
    # only within the same process -- so stripping it from the pickle
    # state is semantically correct: a reloaded model is a fresh
    # instance whose cache will repopulate on the next ``.fit()`` call.
    # ``__setstate__`` re-initialises the cache attrs so legacy saves
    # without the cache fields still load cleanly.

    def __getstate__(self):
        parent_get = getattr(super(), "__getstate__", None)
        state = parent_get() if parent_get is not None else self.__dict__
        state = dict(state)  # shallow copy so we can mutate safely
        # Strip pointer-typed cache attrs -- their key siblings stay
        # pickleable as tuples, but they're meaningless without the
        # Dataset they indexed, so we null them too to avoid a
        # stale-key-looking-valid trap on load.
        for _attr in self._CACHE_POINTER_ATTRS + self._CACHE_KEY_ATTRS:
            state[_attr] = None
        # Wave 19 P1: stamp the lightgbm version at save time. The booster
        # JSON inside the unmodified __dict__ is library-version-sensitive;
        # without this stamp the load side has no way to detect a minor
        # upgrade silently changing booster internals.
        try:
            import lightgbm as _lgb
            state["_saved_lgb_version"] = str(getattr(_lgb, "__version__", "unknown"))
        except Exception:
            # lightgbm should always be importable in this code path (the
            # class wraps it) but be defensive.
            state["_saved_lgb_version"] = "unknown"
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
        # Wave 19 P1: compare the saved lightgbm version against the live
        # one. WARN-only (booster libs are typically forward-compatible
        # for minor versions) so loads of older artifacts don't fail; the
        # operator just sees the skew before chasing weird predict crashes.
        _saved_ver = getattr(self, "_saved_lgb_version", None)
        if _saved_ver is not None and _saved_ver != "unknown":
            try:
                import lightgbm as _lgb
                _live_ver = str(getattr(_lgb, "__version__", "unknown"))
            except Exception:
                _live_ver = "unknown"
            if _live_ver != "unknown" and _live_ver != _saved_ver:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "lgb_shim.__setstate__: lightgbm version drift -- "
                    "artifact saved with lightgbm==%s, loaded under "
                    "lightgbm==%s. Booster internals may have changed "
                    "between versions; if predict() raises AttributeError "
                    "or returns suspicious values, retrain on the live "
                    "lightgbm install.", _saved_ver, _live_ver,
                )

    # ------------------------------------------------------------------
    # Explicit cache release
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Release cached Dataset(s) and their keys. Useful between
        suite runs to free the C++ memory backing the binned storage."""
        self._init_cache()

    # ------------------------------------------------------------------
    # Public extras -- in-place swaps on the cached Dataset
    # ------------------------------------------------------------------

    def set_label(self, y) -> None:
        """Swap label on the cached training Dataset in place. Raises
        if no Dataset has been built yet (call ``.fit()`` first)."""
        if self._cached_train_dataset is None:
            raise RuntimeError(
                "no cached Dataset -- call .fit() first to build one"
            )
        self._cached_train_dataset.set_label(np.asarray(y))

    def set_weight(self, w) -> None:
        """Swap sample weights on the cached training Dataset in place.
        Raises if no Dataset has been built yet."""
        if self._cached_train_dataset is None:
            raise RuntimeError(
                "no cached Dataset -- call .fit() first to build one"
            )
        self._cached_train_dataset.set_weight(np.asarray(w))

    # ------------------------------------------------------------------
    # Override .fit() -- the cache + native lgb.train() path
    # ------------------------------------------------------------------

    def fit(  # type: ignore[override]
        self,
        X,
        y,
        sample_weight=None,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_class_weight=None,
        eval_init_score=None,
        eval_metric=None,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
        monotonic_decline_patience=3,
    ):
        """Cached-Dataset fit.

        Lifecycle:
          1. Set classifier-only state (_le, _classes, _n_classes) so
             ``self._process_params("fit")`` picks the right objective.
          2. Build train Dataset from cache or fresh; mutate label /
             weight in place on cache hit.
          3. Build val Dataset(s) from cache or fresh.
          4. Resolve native params via ``self._process_params("fit")``.
          5. Call ``lightgbm.train(params, train_set, num_boost_round=N,
             valid_sets=...)``.
          6. Attach Booster to ``self._Booster`` plus the bookkeeping
             attrs LGBM sklearn predict expects (``_n_features``,
             ``_evals_result``, ``_best_iteration``, ``_best_score``).

        Parameters mirror ``LGBMClassifier.fit`` so the shim is a
        drop-in replacement.
        """
        # Lazy init in case __init__ wasn't called via subclass route
        # (sklearn.clone() of an exotic estimator).
        if not hasattr(self, "_cached_train_dataset"):
            self._init_cache()

        # ---- Subclass-specific pre-fit bookkeeping -------------------
        # Classifier needs _le / _classes / _n_classes set BEFORE
        # _process_params("fit") so it can pick binary vs multiclass
        # objective. Regressor is a no-op.
        y_for_fit = self._pre_fit_bookkeeping(y)

        # ---- Polars -> pandas (Arrow split-blocks bridge) ------------
        # ``lgb.Dataset(data=polars_df)`` would fall through ``__array__`` and lose Categorical codes. Route through the project's Arrow bridge so
        # numeric columns stay zero-copy and Categorical columns reach LightGBM with their dictionary intact (native cat-split path).
        X = _maybe_bridge_polars_to_pandas(X)

        # Remember the train frame's categorical-column dtypes so predict can realign incoming frames to the SAME
        # CategoricalDtype. LightGBM compares each predict frame's auto-detected cat schema against the fit-time one and
        # raises "categorical_feature do not match" when a column's category set (or its category-ness) differs.
        self._mlframe_train_cat_dtypes = (
            {_c: _dt for _c, _dt in X.dtypes.items() if str(_dt) == "category"}
            if hasattr(X, "dtypes") else {}
        )

        # ---- Train Dataset: cache-or-build ---------------------------
        train_key = _signature_of(X, categorical_feature)
        if self._cached_train_key == train_key and self._cached_train_dataset is not None:
            dtrain = self._cached_train_dataset
            # Validate that label cardinality matches the cached Dataset
            # before any set_label/set_weight: cache hit on ``X`` does NOT
            # guarantee ``y`` has the same length (caller can pass a
            # filtered y while keeping the cached frame). Without this
            # guard, LightGBM accepts the mismatched label silently and
            # downstream set_weight(ones(num_data)) crashes - or worse,
            # produces aligned-by-position garbage when sample_weight is
            # supplied at the wrong length.
            _y_arr = np.asarray(y_for_fit)
            if _y_arr.shape[0] != dtrain.num_data():
                raise ValueError(
                    f"lgb_shim: cached train Dataset has "
                    f"{dtrain.num_data()} rows but y has "
                    f"{_y_arr.shape[0]} rows. Invalidate the cache before "
                    f"refitting (call ._init_cache())."
                )
            dtrain.set_label(_y_arr)
            if sample_weight is not None:
                _w_arr = np.asarray(sample_weight)
                if _w_arr.shape[0] != dtrain.num_data():
                    raise ValueError(
                        f"lgb_shim: sample_weight length "
                        f"{_w_arr.shape[0]} != Dataset.num_data() "
                        f"{dtrain.num_data()}"
                    )
                dtrain.set_weight(_w_arr)
            else:
                # When sample_weight was set previously and now None is
                # passed, "uniform weights" is the desired semantic. We
                # set ones to clear any prior weight; LightGBM treats
                # an all-1 weight identically to no-weight.
                dtrain.set_weight(np.ones(dtrain.num_data(), dtype=np.float32))
            logger.debug(
                "[lgb-shim] reused cached train Dataset (id=%d)",
                id(dtrain),
            )
        else:
            dtrain = _build_dataset(
                X, y_for_fit, sample_weight,
                categorical_feature=categorical_feature,
                feature_name=feature_name,
                init_score=init_score,
            )
            self._cached_train_dataset = dtrain
            self._cached_train_key = train_key
            logger.debug(
                "[lgb-shim] built fresh train Dataset (id=%d)",
                id(dtrain),
            )

        # ---- Eval Dataset(s): cache-or-build -------------------------
        # Normalize eval_set to a canonical list-of-tuples ONCE at the boundary.
        # LightGBM's lgb.train() takes valid_sets as a list + parallel valid_names.
        # mlframe / vanilla LGBM sklearn sometimes pass a bare ``(X_val, y_val)``
        # tuple (or ``[X_val, y_val]`` list) for the single-eval-set case; left
        # un-normalized, iterating yields X then y and the (X, y) unpack would
        # destructure DataFrame column names into the LabelEncoder.
        eval_set = normalize_eval_set(eval_set)
        valid_sets: list[Any] = []
        valid_names: list[str] = []
        if eval_set:
            for i, pair in enumerate(eval_set):
                assert isinstance(pair, tuple) and len(pair) in (2, 3), (
                    f"lgb_shim: normalized eval_set item {i} is {type(pair).__name__} "
                    f"len {len(pair) if hasattr(pair, '__len__') else '?'}; expected a 2/3-tuple."
                )
                pair_seq = pair
                X_val, y_val_raw = pair_seq[0], pair_seq[1]
                # Same Arrow split-blocks bridge as train X: keeps Categorical dtype intact, avoids the ``__array__`` numpy fallthrough.
                X_val = _maybe_bridge_polars_to_pandas(X_val)
                # categorical_feature="auto" makes LightGBM re-detect categoricals from each frame's pandas dtypes
                # independently, so a column that is category-dtype in train X but a different CategoricalDtype (or object)
                # in X_val raises "train and valid dataset categorical_feature do not match". Align X_val's categorical
                # columns to the train frame's exact CategoricalDtype. Only X_val (the small val frame) is rebuilt -- via
                # assign for BlockManager reuse of un-cast columns -- the (potentially huge) train frame X is never touched.
                if hasattr(X, "columns") and hasattr(X_val, "columns"):
                    _val_cols = set(X_val.columns)
                    _realign: dict = {}
                    for _c, _dt in X.dtypes.items():
                        if str(_dt) != "category" or _c not in _val_cols or _c in _realign:
                            continue
                        try:
                            if X_val[_c].dtype != _dt:
                                _realign[_c] = X_val[_c].astype(_dt)
                        except Exception:
                            pass
                    if _realign:
                        X_val = X_val.assign(**_realign)
                w_val_inline = pair_seq[2] if len(pair_seq) >= 3 else None
                # Transform val labels through the encoder for classifier;
                # regressor returns y unchanged.
                y_val = self._transform_y_for_eval(y_val_raw)
                w_val = w_val_inline if w_val_inline is not None else (
                    eval_sample_weight[i]
                    if eval_sample_weight and i < len(eval_sample_weight)
                    else None
                )
                init_val = (
                    eval_init_score[i]
                    if eval_init_score and i < len(eval_init_score)
                    else None
                )
                val_key = _signature_of(X_val, categorical_feature)
                if (
                    self._cached_val_key == val_key
                    and self._cached_val_dataset is not None
                ):
                    dval = self._cached_val_dataset
                    dval.set_label(np.asarray(y_val))
                    if w_val is not None:
                        dval.set_weight(np.asarray(w_val))
                else:
                    dval = _build_dataset(
                        X_val, y_val, w_val,
                        reference=dtrain,  # share bin mapping
                        categorical_feature=categorical_feature,
                        feature_name=feature_name,
                        init_score=init_val,
                    )
                    self._cached_val_dataset = dval
                    self._cached_val_key = val_key
                valid_sets.append(dval)
                if eval_names and i < len(eval_names):
                    valid_names.append(eval_names[i])
                else:
                    # LGB's native convention for default eval-set name is
                    # ``valid_{i}`` (matches LGBMClassifier.fit + lgb.train).
                    # mlframe's LightGBMCallback defaults monitor_dataset to
                    # ``"valid_0"`` so this lookup has to agree - previously
                    # the shim used ``validation_{i}`` and the callback's
                    # set_default_monitor_metric raised on every shim-routed
                    # fit (test_tree_model_with_early_stopping[lgb], surfaced
                    # 2026-05-16 by tests/training run after the migration).
                    valid_names.append(f"valid_{i}")

        # ---- Resolve params for lgb.train() --------------------------
        # ``_process_params("fit")`` strips sklearn-only fields
        # (n_estimators, importance_type, class_weight) and resolves
        # ``objective`` based on _n_classes. It also sets
        # ``self._objective`` if it was None -- our pre-fit hook left
        # it None for that resolution path.
        #
        # Pre-fill ``n_jobs`` so LightGBM's _process_n_jobs() doesn't
        # shell out via joblib -> loky -> wmic to count physical cores
        # (~1.5s on Windows, per cProfile against
        # _count_physical_cores_win32). LightGBM only probes when n_jobs
        # is None; setting it here to os.cpu_count() short-circuits the
        # subprocess. Honour an explicit user choice if already set.
        if getattr(self, "n_jobs", None) is None:
            import os as _os
            self.n_jobs = _os.cpu_count() or 1
        params: dict = self._process_params("fit")
        # Wave 14 P1 (re-opened 2026-05-20): same shape as xgb_shim --
        # pre-fix `or 100` silently rewrote n_estimators=0 to 100. lightgbm
        # accepts n_estimators=0 (means untrained booster); the shim
        # silently overrode that.
        _n_est_raw = self.get_params().get("n_estimators", 100)
        n_estimators = 100 if _n_est_raw is None else _n_est_raw

        # Translate eval_metric -> params["metric"] / feval. String /
        # list-of-strings go into params; callables go to feval. Mirrors
        # LGBMModel.fit's handling.
        feval = None
        if eval_metric is not None:
            metric_strs: list[str] = []
            feval_callables: list[Any] = []
            metrics_iter = eval_metric if isinstance(eval_metric, (list, tuple)) else [eval_metric]
            for m in metrics_iter:
                if callable(m):
                    feval_callables.append(m)
                elif isinstance(m, str):
                    metric_strs.append(m)
            if metric_strs:
                # Merge with any pre-existing metric in params (preserve
                # both rather than overwrite -- LGBM accepts a list).
                existing = params.get("metric")
                if existing is None:
                    params["metric"] = metric_strs if len(metric_strs) > 1 else metric_strs[0]
                elif isinstance(existing, list):
                    params["metric"] = existing + metric_strs
                else:
                    params["metric"] = [existing] + metric_strs
            if feval_callables:
                # Wrap user callables in _EvalFunctionWrapper so the
                # native lgb.train sees the (preds, dataset) -> (name,
                # value, higher_is_better) shape it expects, while the
                # mlframe callable keeps its sklearn-style
                # (y_true, y_pred[, weight[, group]]) signature. Mirror
                # of LGBMModel.fit's eval-metric wiring.
                wrapped = [_EvalFunctionWrapper(f) for f in feval_callables]
                feval = wrapped if len(wrapped) > 1 else wrapped[0]

        # ---- Native lgb.train() --------------------------------------
        evals_result: dict = {}
        train_callbacks = list(callbacks) if callbacks else []
        if valid_sets:
            train_callbacks.append(lgb.record_evaluation(evals_result))
            # Default-on monotonic strict-decline overfitting stop, COMPLEMENTARY to native
            # early_stopping_rounds: stop once the first val set's metric strictly worsens for
            # ``monotonic_decline_patience`` consecutive rounds since the best. Skipped when the
            # caller already supplied one, or when disabled (None). The native best-iteration
            # rollback (EarlyStopException) keeps the global-best booster.
            from .callbacks.monotonic_decline import LGBMonotonicDeclineStop
            if monotonic_decline_patience is not None and not any(
                isinstance(cb, LGBMonotonicDeclineStop) for cb in train_callbacks
            ):
                train_callbacks.append(LGBMonotonicDeclineStop(patience=monotonic_decline_patience))

        booster = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=int(n_estimators),
            valid_sets=valid_sets or None,
            valid_names=valid_names or None,
            feval=feval,
            callbacks=train_callbacks or None,
            init_model=init_model,
        )

        # ---- Attach Booster + sklearn-convention metadata ------------
        # LGBMModel exposes ``booster_`` as a property returning
        # ``self._Booster``; ``predict`` / ``predict_proba`` route
        # through ``self._Booster.predict()``. The bookkeeping attrs
        # below mirror what ``LGBMModel.fit`` sets after its native
        # train() call so the inherited methods see the same shape of
        # state.
        self._Booster = booster
        self._n_features = booster.num_feature()
        self._n_features_in = self._n_features
        self._evals_result = evals_result
        self._best_iteration = booster.best_iteration
        self._best_score = booster.best_score
        # ``fitted_`` is the flag ``__sklearn_is_fitted__`` checks --
        # without it predict raises NotFittedError even though _Booster
        # is set. Mirror of LGBMModel.fit's final state-flip line.
        self.fitted_ = True

        # ``_class_weight`` is consumed by predict_proba (multiclass
        # path) and by some sample weighting helpers; set it to a safe
        # default if pre_fit_bookkeeping didn't already.
        if not hasattr(self, "_class_weight"):
            self._class_weight = getattr(self, "class_weight", None)

        return self

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _pre_fit_bookkeeping(self, y):
        """Subclass override: classifier sets ``_le`` / ``_classes`` /
        ``_n_classes`` and returns the LabelEncoder-transformed y;
        regressor returns y unchanged."""
        return y

    def _transform_y_for_eval(self, y):
        """Subclass override: classifier puts eval-set y through the
        same LabelEncoder used for train; regressor returns y unchanged."""
        return y

    def _align_cats_for_predict(self, X):
        """Recast the incoming frame's categorical columns to the train frame's CategoricalDtype so LightGBM's per-frame
        cat-schema check matches the fit-time one. No-op when the model trained without categoricals or nothing differs;
        only the incoming frame is rebuilt (assign block-reuse), never the caller's data."""
        train_cats = getattr(self, "_mlframe_train_cat_dtypes", None)
        if not train_cats:
            return X
        X = _maybe_bridge_polars_to_pandas(X)
        if not hasattr(X, "columns"):
            return X
        _cols = set(X.columns)
        _realign: dict = {}
        for _c, _dt in train_cats.items():
            if _c not in _cols or _c in _realign:
                continue
            try:
                if X[_c].dtype != _dt:
                    _realign[_c] = X[_c].astype(_dt)
            except Exception:
                pass
        return X.assign(**_realign) if _realign else X

    def predict(self, X, *args, **kwargs):
        return super().predict(self._align_cats_for_predict(X), *args, **kwargs)


# ---------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------

if _LGB_AVAILABLE:

    class LGBMClassifierWithDatasetReuse(_DatasetReuseMixin, LGBMClassifier):
        """LGBMClassifier with cached Dataset across fits.

        See module docstring for the full rationale and migration path.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._init_cache()

        def __sklearn_clone__(self):
            """Override sklearn's clone to produce a fresh instance with
            an empty cache. Default ``clone()`` would copy params via
            ``get_params(deep=False)`` and run ``__init__``, which already
            calls ``_init_cache()`` -- so default behaviour is correct.
            But we declare ``__sklearn_clone__`` explicitly to make the
            contract visible: cloned instance MUST NOT inherit cache."""
            return type(self)(**self.get_params(deep=False))

        def _pre_fit_bookkeeping(self, y):
            """Set _le / _classes / _n_classes / _objective so
            _process_params("fit") picks binary vs multiclass. Mirror of
            LGBMClassifier.fit's pre-train state setup."""
            y_arr = np.asarray(y)
            self._le = _LGBMLabelEncoder().fit(y_arr)
            y_encoded = self._le.transform(y_arr)
            self._class_map = dict(zip(self._le.classes_, self._le.transform(self._le.classes_)))
            self._classes = self._le.classes_
            self._n_classes = len(self._classes)
            # Leave _objective None so _process_params resolves it from
            # _n_classes (-> "binary" or "multiclass"). If a user already
            # set objective via params, _process_params handles that path.
            self._objective = None
            # _class_weight: the predict path inspects this in some
            # branches; mirror LGBMModel.fit's default.
            class_weight = getattr(self, "class_weight", None)
            if isinstance(class_weight, dict):
                self._class_weight = {self._class_map[k]: v for k, v in class_weight.items()}
            else:
                self._class_weight = class_weight
            return y_encoded

        def _transform_y_for_eval(self, y):
            """Pass eval y through the same LabelEncoder so train and val
            share the encoded space."""
            if self._le is None:
                return y
            return self._le.transform(np.asarray(y))

        def predict_proba(self, X, *args, **kwargs):
            return super().predict_proba(self._align_cats_for_predict(X), *args, **kwargs)


    class LGBMRegressorWithDatasetReuse(_DatasetReuseMixin, LGBMRegressor):
        """LGBMRegressor with cached Dataset across fits.

        See module docstring for the full rationale and migration path.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._init_cache()

        def __sklearn_clone__(self):
            return type(self)(**self.get_params(deep=False))

        # _pre_fit_bookkeeping / _transform_y_for_eval inherited as
        # identity -- regressor doesn't use a label encoder.


__all__ = [
    "lgb_dataset_reuse_capable",
    "normalize_eval_set",
    "LGBMClassifierWithDatasetReuse",
    "LGBMRegressorWithDatasetReuse",
]
