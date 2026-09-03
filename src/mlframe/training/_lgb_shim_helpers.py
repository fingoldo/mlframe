"""Module-level helpers and Dataset cache for ``lgb_shim.py``.

Split out of ``lgb_shim.py`` once that file crossed the project's 1000-LOC
monolith-split threshold (see CLAUDE.md's "New code goes in focused
submodules" rule) -- these are standalone functions with no dependency on
``_DatasetReuseMixin``, so they carve cleanly.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


try:
    import lightgbm as lgb

    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    lgb = None  # type: ignore


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


def _reset_weight_to_uniform(dataset: "lgb.Dataset") -> None:
    """Force ``dataset``'s weight field to true uniform (all-ones) at the C++ side, bypassing
    ``Dataset.set_weight``'s own all-ones short-circuit.

    LightGBM's ``Dataset.set_weight(weight)`` treats an all-ones array as equivalent to "no weight"
    (``if np.all(weight == 1): weight = None``) and then SKIPS the ``set_field`` call entirely when
    ``weight is None`` -- so calling ``set_weight(np.ones(...))`` on a Dataset that already has a
    REAL (non-uniform) weight set at the C++ side silently does nothing: ``get_weight()`` still
    returns the stale prior weight (confirmed live). Using the lower-level ``set_field`` directly
    bypasses that optimization and always performs the write; clearing the Python-side ``self.weight``
    cache afterward forces the next ``get_weight()`` to re-fetch from C++ instead of returning a
    stale cached value.
    """
    dataset.set_field("weight", np.ones(dataset.num_data(), dtype=np.float32))
    dataset.weight = None


def _is_pair_item(obj: Any) -> bool:
    """True when ``obj`` looks like an X/y array (DataFrame / ndarray / polars / Series), i.e. one element of a bare (X, y[, w]) bundle."""
    if isinstance(obj, (str, bytes)) or isinstance(obj, (list, tuple)):
        return False
    return bool(hasattr(obj, "shape") or hasattr(obj, "columns") or hasattr(obj, "iloc") or hasattr(obj, "dtypes"))


def normalize_eval_set(eval_set: Any) -> list[tuple] | None:
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
                _second_shape is not None
                and len(_second_shape) >= 2
                and _first_shape is not None
                and len(_first_shape) >= 2
                and _second_shape[1] == _first_shape[1]
            )
            if not _is_list_of_matrices:
                return [tuple(eval_set)]
        # Proper list of pairs.
        return [tuple(p) for p in eval_set]

    raise TypeError(f"lgb_shim: unsupported eval_set type {type(eval_set).__name__}; expected None, tuple, or list.")


def _build_dataset(
    X,
    y,
    sample_weight,
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
# Module-level Dataset cache, keyed by content fingerprint. Mirrors
# xgb_shim's ``_XGB_DMATRIX_CACHE``/``_xgb_cache_get``/``_xgb_cache_put``
# exactly. Without this, sklearn.clone() of an LGB shim (the same
# CompositeCrossTargetEnsemble OOF refit pattern documented as the
# motivating case for the XGB module cache) produces a fresh instance
# with an empty instance-level cache and silently rebuilds the whole
# binned Dataset from scratch on every clone -- the exact cost this
# shim exists to eliminate, just never actually delivered on the LGB
# side because no module-level fallback existed.
#
# ``MLFRAME_LGB_CACHE_DISABLE=1`` env var forces bypass (testing only).
from collections import OrderedDict as _OrderedDict
import os as _os
import threading as _threading

_LGB_DATASET_CACHE: "_OrderedDict[tuple, Any]" = _OrderedDict()
_LGB_DATASET_CACHE_CAP: int = 8
_LGB_DATASET_CACHE_LOCK = _threading.Lock()


def _lgb_cache_disabled() -> bool:
    """True when the ``MLFRAME_LGB_CACHE_DISABLE`` env var is set, forcing every Dataset cache lookup/store to be a no-op (diagnostics / A-B testing escape hatch)."""
    return bool(_os.environ.get("MLFRAME_LGB_CACHE_DISABLE"))


def _lgb_cache_get(key: tuple):
    """LRU-ordered lookup of a cached Dataset by content-signature ``key``; touches the entry to most-recently-used on hit, returns None on miss or when the cache is disabled."""
    if _lgb_cache_disabled() or key is None:
        return None
    with _LGB_DATASET_CACHE_LOCK:
        ds = _LGB_DATASET_CACHE.get(key)
        if ds is not None:
            _LGB_DATASET_CACHE.move_to_end(key)
        return ds


def _lgb_cache_put(key: tuple, dataset: Any) -> None:
    """Store ``dataset`` under ``key`` in the LRU Dataset cache, evicting the least-recently-used entry once the cache exceeds ``_LGB_DATASET_CACHE_CAP``."""
    if _lgb_cache_disabled() or key is None or dataset is None:
        return
    with _LGB_DATASET_CACHE_LOCK:
        if key in _LGB_DATASET_CACHE:
            _LGB_DATASET_CACHE.move_to_end(key)
        _LGB_DATASET_CACHE[key] = dataset
        while len(_LGB_DATASET_CACHE) > _LGB_DATASET_CACHE_CAP:
            _LGB_DATASET_CACHE.popitem(last=False)


def _lgb_cache_clear() -> None:
    """Release all cached Datasets (call between long-running suite invocations to free C++ memory)."""
    with _LGB_DATASET_CACHE_LOCK:
        _LGB_DATASET_CACHE.clear()
