"""``_maybe_get_or_build_cb_pool`` carved out of
``mlframe.training._cb_pool``.

Re-imported at the parent's module bottom so historical
``from mlframe.training._cb_pool import _maybe_get_or_build_cb_pool``
resolves transparently.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import pandas as pd

from mlframe.config import CATBOOST_MODEL_TYPES

logger = logging.getLogger(__name__)

# Guards concurrent first-time probing of the GPU info cache and the CB-GPU
# usable cache. Two parallel suite invocations can otherwise both pay the
# `nvidia-smi` subprocess cost or duplicate the tiny CB probe fit; the lock
# costs nothing on the hot path (single test + return). RLock (not Lock):
# _cb_gpu_usable acquires this lock and then calls _cached_gpu_info which
# acquires it again; with a plain Lock the second acquire deadlocks on the
# very first probe (before _GPU_INFO_PROBED is True).
_GPU_PROBE_LOCK = threading.RLock()


def _maybe_get_or_build_cb_pool(
    model_type_name: str,
    model: Any,
    train_df: Any,
    train_target: Any,
    fit_params: dict[str, Any],
) -> Any | None:
    """Return a cached/freshly-built ``catboost.Pool`` when the CB reuse
    fast-path applies; return None otherwise (caller falls back to
    ``model.fit(train_df, y, **fit_params)``).

    Fast-path activation requires ALL of:
      * ``model_type_name in CATBOOST_MODEL_TYPES``
      * installed CatBoost has Pool.set_label/set_weight
      * train_df is a recognised input type (polars/pandas/numpy)

    Cache-hit: swap label + weight in place, return the cached Pool.
    Cache-miss: build a new Pool, store, return it.
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from ._cb_pool import (
        _CB_POOL_CACHE,
        _CB_POOL_CACHE_MAX_ENTRIES,
        _cb_reuse_capable,
        _coerce_label_for_cb_pool,
    )
    if model_type_name not in CATBOOST_MODEL_TYPES:
        return None

    # Empty-target guard: if the caller passed a 0-length train_target
    # (e.g. RFECV inner CV fold collapsed after MRMR dropped all rows of
    # one class on rare-imbalance combos -- fuzz c0079), skip the
    # Pool-reuse fast-path and let CB raise a clearer error from the
    # sklearn wrapper. Using the cached Pool would silently set an empty
    # label and CB then crashes deep in ``_check_label_empty`` with no
    # context about which combo / fold triggered it.
    try:
        if train_target is not None and hasattr(train_target, "__len__") and len(train_target) == 0:
            logger.warning("[cb-pool-reuse] empty train_target -- skipping Pool reuse " "(would set zero-length label); deferring to sklearn fallback.")
            return None
    except Exception as e:
        logger.debug("swallowed exception in _cb_pool_build.py: %s", e)
        pass

    # Filter cat/text/embedding features to only those actually present
    # in train_df. Motivation: MRMR and similar selectors can drop columns
    # AFTER fit_params was built, leaving stale feature lists that CB's
    # Pool rejects with ``ValueError: 'feat' is not in list`` from the
    # sklearn-wrapper's ``_get_cat_feature_indices`` (observed 2026-04-21
    # on ``test_mrmr_with_text_column`` / ``_embedding_column``). Applied
    # to CB only -- XGB/LGB have their own handling for missing cols.
    try:
        _df_cols = set(train_df.columns) if hasattr(train_df, "columns") else None
    except Exception:
        _df_cols = None

    def _filter_to_df(feats):
        """Fetch ``fit_params[feats]`` and drop any column name no longer present in ``train_df``, so a stale CB feature list can't raise CatBoost's Pool ``ValueError``."""
        raw = fit_params.get(feats) or []
        if _df_cols is None:
            return tuple(sorted(raw))
        return tuple(sorted(c for c in raw if c in _df_cols))

    cat_features = _filter_to_df("cat_features")
    text_features = _filter_to_df("text_features")
    embedding_features = _filter_to_df("embedding_features")
    # Auto-widen cat_features with category-dtype columns that are present in
    # train_df but absent from the explicit list. CatBoost's Pool builder
    # rejects category-dtype columns missing from ``cat_features`` with
    # "has dtype 'category' but is not in cat_features list". The
    # skip_categorical_encoding auto-flip path leaves cat_features narrow
    # while the pre-pipeline converts upstream string cols to category for
    # joint train+val codebooks; this guard reconciles the two views.
    try:
        if isinstance(train_df, pd.DataFrame):
            _cat_dtype_cols = [c for c, dt in zip(train_df.columns, train_df.dtypes) if isinstance(dt, pd.CategoricalDtype)]
            # Any category-dtype column NOT already routed via text_features /
            # embedding_features must appear in cat_features - otherwise CB
            # Pool rejects it with "has dtype 'category' but is not in
            # cat_features list". Text/embedding columns are CB-supported via
            # their respective parameters, so we don't widen there.
            _missing = [c for c in _cat_dtype_cols if c not in cat_features and c not in text_features and c not in embedding_features]
            if _missing:
                logger.info(
                    "[cb-pool-reuse] auto-widening cat_features with %d category-dtype " "column(s) missing from explicit list: %s",
                    len(_missing),
                    _missing,
                )
                cat_features = tuple(sorted(set(cat_features) | set(_missing)))
                fit_params["cat_features"] = list(cat_features)
            # Category-dtype columns that ARE routed to text_features must
            # be cast back to string/object before Pool: CB's Pool builder
            # validates dtype-vs-feature-list consistency BEFORE consulting
            # the text_features arg, so a "category" column claimed as text
            # still trips the cat_features mismatch.
            _cat_dt_routed_as_text = [c for c in _cat_dtype_cols if c in text_features]
            if _cat_dt_routed_as_text:
                logger.info(
                    "[cb-pool-reuse] decategorising %d text-feature column(s) " "(category dtype -> object) before Pool: %s",
                    len(_cat_dt_routed_as_text),
                    _cat_dt_routed_as_text,
                )
                # Shallow copy: only the text-routed columns are cast below; deep-copying a 100+ GB train frame to recast a few columns OOMs. ``deep=False`` shares untouched buffers, caller frame unmutated.
                train_df = train_df.copy(deep=False)
                for _c in _cat_dt_routed_as_text:
                    train_df[_c] = train_df[_c].astype(object)
    except Exception as _exc:
        logger.debug("cat_features auto-widen failed: %r", _exc)
    # Update fit_params in place so the fallback sklearn path (when reuse
    # is disabled or Pool construction fails) also sees the filtered
    # lists. Callers may rely on the same fit_params dict downstream; we
    # only narrow, never widen.
    if _df_cols is not None:
        if fit_params.get("cat_features"):
            fit_params["cat_features"] = list(cat_features)
        if fit_params.get("text_features"):
            fit_params["text_features"] = list(text_features)
        if fit_params.get("embedding_features"):
            fit_params["embedding_features"] = list(embedding_features)

    if not _cb_reuse_capable():
        return None
    try:
        from catboost import Pool as _Pool
    except ImportError:
        return None

    sample_weight = fit_params.get("sample_weight")

    # Cache key: content-fingerprint via shared helper. Pre-2026-05-23
    # used ``id(train_df)`` which broke across ``sklearn.clone()`` and
    # ``train_df.iloc[...]`` -- same id(X)-cache-bug class found across
    # all four booster dataset caches (xgb_shim, lgb_shim, this train
    # Pool, val Pool). Now keyed on (cols, shape, 3-row content hash,
    # cat/text/embedding feature tuples).
    from .._dataset_cache_fingerprint import compute_signature
    key = compute_signature(
        train_df,
        extra=(cat_features, text_features, embedding_features),
    )

    # Verify train_target length matches train_df row count BEFORE the
    # Pool-reuse fast-path. RFECV's inner CV folds occasionally hand us
    # train_target / train_df pairs whose lengths disagree (subset of
    # rows but full target, or vice versa); the Pool then ends up with a
    # stale label and CB.fit raises "Labels variable is empty" deep in
    # C++ Pool init (fuzz c0079). Skip Pool reuse on mismatch and let
    # the sklearn fallback path build a fresh Pool with the current
    # (data, label) pair.
    _df_rows = train_df.shape[0] if hasattr(train_df, "shape") else None
    _tg_len = len(train_target) if train_target is not None and hasattr(train_target, "__len__") else None
    if _df_rows is not None and _tg_len is not None and _df_rows != _tg_len:
        # Hard contract violation -- raised 2026-04-28 (batch 4, was
        # logger.error+fallback). X/y length mismatch reaching this
        # point means an upstream slicing bug (fuzz c0079-style: RFECV
        # inner CV producing inconsistent train_target / train_df
        # lengths). Fall back to sklearn would just delay the same
        # error with less context; raise here gives a stack trace
        # rooted in mlframe's flow, not deep in CB's C++ ``Labels
        # variable is empty`` (which is misleading -- it's about
        # length, not emptiness).
        raise RuntimeError(
            f"[cb-pool-reuse] train_df rows ({_df_rows}) != "
            f"train_target len ({_tg_len}). This is a hard contract "
            f"violation; investigate upstream slicing (RFECV inner CV / "
            f"OD filter / aging trim) that produced the mismatch."
        )

    cached = _CB_POOL_CACHE.get(key)
    if cached is not None:
        # Installed CatBoost 1.2.10 rejects ``Pool.set_label`` on a
        # classification Pool (target type ``Integer``) -- the C++
        # ``SetNumericTarget`` path only accepts numeric / unset targets.
        # That means we can only reuse across WEIGHT swaps, not label
        # swaps, for classification pools. Strategy: skip ``set_label``
        # unless the caller actually supplied a genuinely different target
        # (by CONTENT fingerprint, not id() -- CPython can and does reuse a
        # just-freed array's memory address for a new allocation of matching
        # size, e.g. RFECV inner CV folds slicing a fresh train_target = y[fold_mask]
        # each iteration; an id() collision there would silently keep a stale
        # label on the reused Pool while training proceeds against fresh data).
        # Always mutate weight -- ``set_weight`` has no target-type restriction.
        from mlframe.training.pipeline._pipeline_cache import _full_target_content_hash
        last_target_sig = getattr(cached, "_mlframe_last_target_sig", None)
        try:
            _target_sig = _full_target_content_hash(train_target)
            if last_target_sig is None or _target_sig != last_target_sig:
                # Label swap. Cast to float32 -- the Pool was built with a
                # float32 label (see build path below), and CB's C++
                # ``SetNumericTarget`` rejects anything but Float/None. If
                # rejection happens anyway, fall through to rebuild.
                try:
                    # Route through the shared lossless guard: a large-magnitude
                    # regression target keeps float64 instead of silently
                    # collapsing adjacent values under float32 (~7 sig digits).
                    _label_for_swap = _coerce_label_for_cb_pool(train_target)
                except Exception:
                    _label_for_swap = train_target
                cached.set_label(_label_for_swap)
                cached._mlframe_last_target_sig = _target_sig
            if sample_weight is not None:
                cached.set_weight(sample_weight)
            # Post-swap verification: confirm the cached Pool's label is
            # non-empty (set_label can silently set a 0-length array if
            # _label_for_swap was empty after some upstream filter, then
            # CB.fit raises "Labels variable is empty" deep in _check_-
            # label_empty with no diagnostics, fuzz c0079). Evict and
            # rebuild on miss.
            try:
                _post_label = cached.get_label()
                if _post_label is not None and hasattr(_post_label, "__len__") and len(_post_label) == 0:
                    logger.info("[cb-pool-reuse] cached Pool ended up with empty label after swap " "-- evicting and rebuilding.")
                    _CB_POOL_CACHE.pop(key, None)
                    raise RuntimeError("empty cached label after set_label")
            except Exception as _verify_exc:
                if "empty cached label" in str(_verify_exc):
                    raise
                # get_label() not exposed on this CB build; trust set_label.
                pass
            logger.info(
                "[cb-pool-reuse] hit key=(id=%s,cat=%d,text=%d,emb=%d) " "swapped weight%s without rebuild",
                key[0],
                len(cat_features),
                len(text_features),
                len(embedding_features),
                " + label" if (last_target_sig is None or _target_sig != last_target_sig) else "",
            )
            return cached
        except Exception as exc:
            # Drop the stale entry and fall through to rebuild. Typical
            # trigger: classification Pool + set_label on Integer target
            # raises "SetNumericTarget requires numeric or unset target
            # type". Rebuild is safe.
            logger.info(
                "[cb-pool-reuse] swap path not usable (%s: %s); rebuilding Pool.",
                type(exc).__name__,
                str(exc).splitlines()[0][:120],
            )
            _CB_POOL_CACHE.pop(key, None)

    # Simple FIFO eviction -- unlikely to hit during normal runs (<= N
    # models x N tiers entries), but keeps the cache from growing
    # unboundedly across long-running sessions.
    while len(_CB_POOL_CACHE) >= _CB_POOL_CACHE_MAX_ENTRIES:
        _CB_POOL_CACHE.pop(next(iter(_CB_POOL_CACHE)))

    # Cast label to float32 at build time. CatBoost stores the label's
    # raw type on the Pool (Integer vs Float) and later ``Pool.set_label``
    # validates ``ERawTargetType == Float or None`` inside C++
    # ``SetNumericTarget`` -- if we built with Integer labels, subsequent
    # label swaps across classification targets would raise
    # ``SetNumericTarget requires numeric or unset target type, got
    # Integer``. Building with float32 pins the Pool's target type to
    # Float upfront; the user's upstream PR's classification tests all
    # pre-cast to float32 for exactly this reason. get_label() still
    # round-trips integer dtype via the Python-level ``target_type``
    # shadow on the Pool.
    try:
        _label_for_pool = _coerce_label_for_cb_pool(train_target)
    except Exception:
        _label_for_pool = train_target

    try:
        pool = _Pool(
            data=train_df,
            label=_label_for_pool,
            weight=sample_weight,
            cat_features=list(cat_features) or None,
            text_features=list(text_features) or None,
            embedding_features=list(embedding_features) or None,
        )
    except Exception as exc:
        # If Pool rejects the input (e.g. unsupported dtype combo),
        # fall back to the sklearn-wrapper path by returning None. The
        # operator sees the build-logger line above; we don't cache a
        # failed attempt.
        logger.warning("[cb-pool-reuse] Pool construction failed (%s: %s); falling back to rebuild-every-fit sklearn path.", type(exc).__name__, exc)
        return None

    from mlframe.training.pipeline._pipeline_cache import _full_target_content_hash
    pool._mlframe_last_target_sig = _full_target_content_hash(train_target)
    # Cache feature lists on the Pool so callers (notably the dynamic CB
    # ``text_processing`` injection in ``_train_model_with_fallback``)
    # can introspect them without round-tripping through fit_params,
    # which the Pool-reuse path strips before fit.
    pool._mlframe_text_features = list(text_features)
    pool._mlframe_cat_features = list(cat_features)
    pool._mlframe_embedding_features = list(embedding_features)
    _CB_POOL_CACHE[key] = pool
    logger.info(
        "[cb-pool-reuse] miss; stored fresh Pool (cache size=%d)",
        len(_CB_POOL_CACHE),
    )
    return pool
