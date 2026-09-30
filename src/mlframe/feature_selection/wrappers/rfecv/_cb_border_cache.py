"""CatBoost-only fast path for RFECV fold fits: quantization borders are computed once per fold and re-indexed per column subset.

Every RFECV iteration refits each CV fold on a new (smaller) column subset, and CatBoost recomputes the float-feature quantization borders from
scratch each time. Borders are a per-feature function of that feature's train values (plus the quantization params), so a subset fit that
re-uses the borders the fold's full-width fit already computed is bit-identical to fitting from scratch (``ignored_features`` on a full-width
Pool is NOT: it changes the model). Cat features carry no float borders and are untouched.

The cache is keyed by (fold train rows, y, weights) and valid only while the source frame it was built from is alive and unchanged identity-wise,
so a second ``fit`` on other data can never see stale borders. Anything unusual (GPU, text/embedding features, explicit per-feature quantization,
any exception) returns ``False`` and the caller runs the generic ``estimator.fit`` path.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
import weakref
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")

# Per-fold border tables are ~p * 254 floats; a handful of folds x a few distinct fold-row sets stays in the low MB.
_MAX_ENTRIES = 32
_QUANT_PARAMS = ("border_count", "max_bin", "feature_border_type", "nan_mode")
_UNSUPPORTED_PARAMS = ("per_float_feature_quantization", "ignored_features", "text_features", "embedding_features", "text_processing", "dictionaries")

_LOCK = threading.Lock()
TOTALS = {"hits": 0, "misses": 0, "fallbacks": 0}
_CACHES: dict = {}


class _FoldBorders:
    """Borders of one fold: ``lines[name]`` = tab-joined ``border[\\tnan_mode]`` rows; ``known`` = every column the borders were computed over."""

    __slots__ = ("lines", "known")

    def __init__(self) -> None:
        self.lines: dict = {}
        self.known: set = set()


class _SourceCache:
    """All fold tables built from one source frame; ``ref`` guards against id reuse after the frame is freed."""

    def __init__(self, source: Any) -> None:
        self.ref = weakref.ref(source)
        self.folds: dict = {}
        self.hits = 0
        self.misses = 0


def _is_catboost(estimator: Any) -> bool:
    return type(estimator).__name__ in ("CatBoostClassifier", "CatBoostRegressor")


def _digest(*arrays: Any) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        if a is None:
            h.update(b"-")
            continue
        arr = np.ascontiguousarray(np.asarray(a))
        h.update(str(arr.shape).encode())
        h.update(arr.dtype.str.encode())
        h.update(arr.view(np.uint8).reshape(-1).data if arr.dtype != object else str(arr.tolist()).encode())
    return h.digest()


def _get_cache(source: Any) -> Optional[_SourceCache]:
    key = id(source)
    with _LOCK:
        cache = _CACHES.get(key)
        if cache is not None and cache.ref() is not source:
            cache = None
        if cache is None:
            try:
                cache = _SourceCache(source)
            except TypeError:
                return None
            _CACHES[key] = cache
            weakref.finalize(source, _CACHES.pop, key, None)
        return cache


def cache_stats(source: Any) -> dict:
    """Hit/miss counters for a source frame's cache (test/diagnostic hook)."""
    with _LOCK:
        cache = _CACHES.get(id(source))
    if cache is None or cache.ref() is not source:
        return {"hits": 0, "misses": 0, "folds": 0}
    return {"hits": cache.hits, "misses": cache.misses, "folds": len(cache.folds)}


def _param(estimator: Any, name: str) -> Any:
    try:
        return estimator.get_params().get(name)
    except Exception:
        return None


def is_supported(estimator: Any, fit_params: dict) -> bool:
    """Whether this estimator + fit kwargs can use the cached-borders path (CatBoost, CPU, no text/embedding/per-feature quantization)."""
    if not _is_catboost(estimator) or os.environ.get("MLFRAME_RFECV_CB_CACHED_BORDERS", "1").strip().lower() in ("0", "false", "off", "no"):
        return False
    try:
        params = estimator.get_params()
    except Exception:
        return False
    if str(params.get("task_type") or "CPU").upper() != "CPU":
        return False
    if any(params.get(k) for k in _UNSUPPORTED_PARAMS):
        return False
    if any(fit_params.get(k) for k in ("text_features", "embedding_features", "baseline", "pairs", "sample_weight_eval_set")):
        return False
    return True


def _write_subset_borders(fold: _FoldBorders, fit_features: list, path: str) -> None:
    with open(path, "w") as fh:
        for new_idx, name in enumerate(fit_features):
            for row in fold.lines.get(name, ()):
                fh.write(f"{new_idx}\t{row}\n")


def _harvest(pool: Any, fit_features: list, fold: _FoldBorders) -> None:
    fd, path = tempfile.mkstemp(suffix=".borders.tsv")
    os.close(fd)
    try:
        pool.save_quantization_borders(path)
        new_lines: dict = {}
        with open(path) as fh:
            for raw in fh:
                idx, rest = raw.rstrip("\n").split("\t", 1)
                new_lines.setdefault(fit_features[int(idx)], []).append(rest)
    finally:
        os.remove(path)
    for name in fit_features:
        if name not in fold.known:
            fold.known.add(name)
            if name in new_lines:
                fold.lines[name] = new_lines[name]


def fit_catboost_with_cached_borders(
    fitted: Any,
    *,
    source: Any,
    X_train: Any,
    y_train: Any,
    fit_features: list,
    fit_params: dict,
    train_rows: Any,
    sample_weight: Any = None,
) -> bool:
    """Fit ``fitted`` on ``X_train`` through a Pool quantized with the fold's cached borders; ``True`` when done, ``False`` to use the generic fit.

    ``fit_params`` is the per-estimator kwargs the generic path would have passed (``eval_set``, ``cat_features``, ``use_best_model``,
    ``early_stopping_rounds`` ...); ``sample_weight`` must NOT be in it (it is passed separately). ``train_rows`` identifies the fold's train rows.
    """
    if not is_supported(fitted, fit_params):
        return False
    cache = _get_cache(source)
    if cache is None:
        return False
    try:
        from catboost import Pool

        params = fitted.get_params()
        cat = fit_params.get("cat_features")
        if cat is None:
            cat = params.get("cat_features")
        pool = Pool(X_train, y_train, cat_features=list(cat) if cat is not None and len(cat) else None, weight=sample_weight)
        fkey = _digest(train_rows, y_train if not hasattr(y_train, "to_numpy") else y_train.to_numpy(), sample_weight)
        qkw = {k: params[k] for k in _QUANT_PARAMS if params.get(k) is not None}
        with _LOCK:
            fold = cache.folds.get(fkey)
            if fold is None:
                if len(cache.folds) >= _MAX_ENTRIES:
                    cache.folds.pop(next(iter(cache.folds)))
                fold = cache.folds[fkey] = _FoldBorders()
        names = list(fit_features)
        with _LOCK:
            hit = all(n in fold.known for n in names)
            if hit:
                snapshot = _FoldBorders()
                snapshot.lines = {n: fold.lines[n] for n in names if n in fold.lines}
        if hit:
            fd, path = tempfile.mkstemp(suffix=".borders.tsv")
            os.close(fd)
            try:
                _write_subset_borders(snapshot, names, path)
                pool.quantize(input_borders=path, **qkw)
            finally:
                os.remove(path)
            with _LOCK:
                cache.hits += 1
                TOTALS["hits"] += 1
        else:
            pool.quantize(**qkw)
            local = _FoldBorders()
            _harvest(pool, names, local)
            with _LOCK:
                for n in names:
                    if n not in fold.known:
                        fold.known.add(n)
                        if n in local.lines:
                            fold.lines[n] = local.lines[n]
                cache.misses += 1
                TOTALS["misses"] += 1
        kwargs = {k: v for k, v in fit_params.items() if k not in ("cat_features", "eval_set", "sample_weight")}
        eval_set = fit_params.get("eval_set")
        if eval_set is not None and not isinstance(eval_set, Pool):
            Xv, yv = eval_set if not isinstance(eval_set, list) else eval_set[0]
            eval_set = Pool(Xv, yv, cat_features=list(cat) if cat is not None and len(cat) else None)
        fitted.fit(pool, eval_set=eval_set, **kwargs)
        return True
    except Exception as exc:
        TOTALS["fallbacks"] += 1
        logger.warning("RFECV CatBoost cached-borders fast path failed (%s: %s); falling back to the generic fit.", type(exc).__name__, exc)
        return False
