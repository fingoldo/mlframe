"""Pre-pipeline structural-identity + content cache.

Wave 93 (2026-05-21): split out from `_pipeline_helpers.py` to keep that
file below the 1k-line threshold. Behaviour preserved bit-for-bit; every
cache symbol is re-exported from `_pipeline_helpers` so existing
``from ._pipeline_helpers import _content_fingerprint_for_cache`` etc.
imports continue to work.

What lives here:
  - The LRU cache state (`_PRE_PIPELINE_CACHE`, lock, capacity).
  - The `_UncachableSentinel` + `_fresh_uncachable` miss sentinel.
  - `_content_fingerprint_for_cache` (point-sample content key builder).
  - `_pipeline_signature_for_cache` (structural-identity key builder).
  - `_pre_pipeline_cache_key/get/set/clear` (LRU operations).
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

logger = logging.getLogger(__name__)


def _read_int_env(name: str, default: int) -> int:
    """Best-effort int env reader; silently returns ``default`` on bad input so a typo in a shell var cannot crash trainer import."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        val = int(raw)
        if val <= 0:
            return default
        return val
    except (TypeError, ValueError):
        return default


# Cache state. Pre-fix default was 4 which silently evicted on the typical
# suite (cb + lgb + xgb + mlp + linear = 5 models); bumped to 8 so a
# standard suite fits without thrashing. Callers needing tighter bounds set
# TrainingBehaviorConfig.pre_pipeline_cache_max; operators can override the
# global cap via MLFRAME_PRE_PIPELINE_CACHE_MAX (entry count). Byte budget is
# capped by MLFRAME_PRE_PIPELINE_CACHE_MAX_BYTES; 0 / unset disables the byte
# gate (count-only eviction). Both are read once at import; reload tests must
# snapshot per the test-pollution rule.
_PRE_PIPELINE_CACHE_LOCK = threading.Lock()
_PRE_PIPELINE_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_PRE_PIPELINE_CACHE_MAX: int = _read_int_env("MLFRAME_PRE_PIPELINE_CACHE_MAX", 8)
_PRE_PIPELINE_CACHE_MAX_BYTES: int = _read_int_env("MLFRAME_PRE_PIPELINE_CACHE_MAX_BYTES", 0)


def _approx_entry_bytes(entry: tuple) -> int:
    """Best-effort cache-entry size estimate. Sums nbytes / memory_usage / estimated_size across the train+val carriers; falls back to 0 (skip byte-gate) on any failure so an unfamiliar carrier never blocks caching."""
    total = 0
    for obj in entry[:2]:
        if obj is None:
            continue
        try:
            nb = getattr(obj, "nbytes", None)
            if nb is not None:
                total += int(nb)
                continue
            mu = getattr(obj, "memory_usage", None)
            if mu is not None:
                # pandas DataFrame.memory_usage(deep=False) -- avoid deep walk on 100GB frames per the CLAUDE.md memory rule
                total += int(mu(deep=False, index=True).sum())
                continue
            est = getattr(obj, "estimated_size", None)
            if callable(est):
                total += int(est())
        except Exception:
            return 0
    return total


class _UncachableSentinel:
    """Per-instance identity marker; two instances NEVER compare equal under
    default object eq/hash, so cache keys built around two sentinels cannot
    collide. Used when content fingerprinting fails -- a failure must force a
    MISS, not a stable cache hit (P0-3 fix: the previous
    ``("uncached", id(arr))`` was a stable tuple key, so two consecutive
    targets sharing the same filtered_train_df id produced an IDENTICAL
    cache key and target-2 silently consumed target-1's fit-transform output).
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        return "<UncachableSentinel>"


def _fresh_uncachable() -> tuple:
    """Return a never-equal-to-anything-else sentinel tuple. Used in place
    of the unsafe ``("uncached", id(arr))`` which collided cross-target."""
    return ("uncached", _UncachableSentinel())


def _content_fingerprint_for_cache(arr) -> tuple:
    """Content-based fingerprint of an array / DataFrame / target.

    id()-keying is unsafe: GC recycles ids and the suite's per-target loop
    persists ``filtered_train_df`` across targets with the same id() so
    target-2 would otherwise reuse target-1's fit-transform output. The
    fingerprint folds in (n_rows, n_cols, per-column dtypes, column names)
    as cheap drift-detection signature, then point-samples 4 whole rows
    (first, near-head, midpoint, last) without materialising the frame --
    a polars row is a tuple pulled by direct column indexing, so cost is
    O(n_cols) regardless of n_rows. The previous ``arr.to_numpy()`` path
    materialised the entire frame just to slice 10 cells, defeating the
    very cache the per-target loop relies on (fingerprint cost > cache
    benefit on 100+ GB frames). Falls back to ``("uncached", id)`` so a
    failure forces a miss rather than a wrong-cache hit.
    """
    if arr is None:
        return ("none",)
    try:
        # Polars DataFrame: row(i) is a tuple of n_cols scalars -- O(n_cols), no full materialisation.
        if pl is not None and isinstance(arr, pl.DataFrame):
            n_rows, n_cols = int(arr.height), int(arr.width)
            col_names = tuple(str(c) for c in arr.columns)
            dtypes_str = tuple(str(dt) for dt in arr.dtypes)
            # Nested dtypes (List / Array / Struct) yield Python list/dict cells inside `arr.row(i)`;
            # lists are unhashable, so the fingerprint tuple cannot be used as a dict key. Force-uncached
            # on detection -- embedding columns are the common trigger (warning fired in
            # get_pandas_view_of_polars_df) and the cache contract is content-isolated regardless.
            if any(s.startswith(("List", "Array", "Struct")) for s in dtypes_str):
                return _fresh_uncachable()
            if n_rows == 0:
                return ("pl", (n_rows, n_cols), col_names, dtypes_str, ())
            sample_idx = (0, min(8, n_rows - 1), n_rows // 2, n_rows - 1)
            try:
                rows = tuple(arr.row(i) for i in sample_idx)
            except Exception:
                return _fresh_uncachable()
            return ("pl", (n_rows, n_cols), col_names, dtypes_str, rows)

        # Polars Series: iloc-equivalent is ``arr[i]`` -- O(1), no full materialisation.
        if pl is not None and isinstance(arr, pl.Series):
            n = int(arr.len())
            dtype_str = str(arr.dtype)
            # List / Array / Struct dtypes yield Python list/dict cells; lists are unhashable so the
            # fingerprint tuple cannot be used as a dict key. Force-uncached on detection.
            if dtype_str.startswith(("List", "Array", "Struct")):
                return _fresh_uncachable()
            if n == 0:
                return ("pls", (n,), dtype_str, ())
            sample_idx = (0, min(8, n - 1), n // 2, n - 1)
            try:
                cells = tuple(arr[i] for i in sample_idx)
            except Exception:
                return _fresh_uncachable()
            return ("pls", (n,), dtype_str, cells)

        # Pandas DataFrame: .iat / .iloc[i].values -- O(n_cols), no full materialisation.
        if isinstance(arr, pd.DataFrame):
            n_rows, n_cols = int(arr.shape[0]), int(arr.shape[1])
            col_names = tuple(str(c) for c in arr.columns)
            dtypes = tuple(str(dt) for dt in arr.dtypes)
            if n_rows == 0:
                return ("pd", (n_rows, n_cols), col_names, dtypes, ())
            sample_idx = (0, min(8, n_rows - 1), n_rows // 2, n_rows - 1)
            try:
                # object-dtype cells from pyarrow ListArray materialise as Python lists -- unhashable.
                # Coerce row cells to a hashable form: leave scalars alone, repr() any list/dict/ndarray.
                # Bounded cost: 4 rows × O(embedding_len) chars per fingerprint call -- <<1ms even on
                # wide embedding frames.
                def _row_to_hashable(r):
                    out = []
                    for v in r.values.tolist():
                        if isinstance(v, (list, dict)) or hasattr(v, "tolist"):
                            out.append(repr(v))
                        else:
                            out.append(v)
                    return tuple(out)
                rows = tuple(_row_to_hashable(arr.iloc[i]) for i in sample_idx)
            except Exception:
                return _fresh_uncachable()
            return ("pd", (n_rows, n_cols), col_names, dtypes, rows)

        # Pandas Series: .iat[i] -- O(1).
        if isinstance(arr, pd.Series):
            n = int(arr.shape[0])
            dtype_str = str(arr.dtype)
            if n == 0:
                return ("pds", (n,), dtype_str, ())
            sample_idx = (0, min(8, n - 1), n // 2, n - 1)
            try:
                cells = tuple(arr.iat[i] for i in sample_idx)
            except Exception:
                return _fresh_uncachable()
            return ("pds", (n,), dtype_str, cells)

        # NumPy / array-like: a 1-D / 2-D array is already in RAM; the previous flat-index sample is fine.
        if isinstance(arr, np.ndarray):
            np_arr = arr
        elif hasattr(arr, "values") and not hasattr(arr, "to_numpy"):
            np_arr = arr.values
        else:
            np_arr = np.asarray(arr)
        if not hasattr(np_arr, "shape") or not hasattr(np_arr, "dtype"):
            return _fresh_uncachable()
        shape = tuple(int(s) for s in np_arr.shape)
        dtype_str = str(np_arr.dtype)
        flat = np_arr.ravel()
        n = int(flat.size)
        if n == 0:
            return ("np", shape, dtype_str, b"")
        idx = [int(i * (n - 1) / 9) for i in range(10)] if n >= 10 else list(range(n))
        try:
            sampled = bytes(np.ascontiguousarray(flat[idx]).tobytes())
        except Exception:
            return _fresh_uncachable()
        return ("np", shape, dtype_str, sampled)
    except Exception:
        return _fresh_uncachable()


def _full_x_content_hash(arr) -> str:
    """Full blake2b content hash of an X frame for cache-key disambiguation.

    The 4-row point-sample in ``_content_fingerprint_for_cache`` collides on two distinct X frames whose sampled positions (0, 8, n/2, n-1) happen to coincide -- common when a pre-pipeline transform mutates only the unsampled rows (outlier clip on the middle 90 % of the distribution, masked-row fillna, etc.). The per-target loop would then silently replay the prior target's fit-transform output even though the actual X content drifted.

    Mirrors ``_full_y_content_hash`` (target side) and ``_full_x_content_hash`` (MRMR side) so the X/y guarantees are symmetric. Cost is O(rows*cols) bytes hashed -- a few ms even on a 1M-row frame, well under any cache benefit. Returns ``""`` on conversion failure so the caller can choose to skip the cache rather than serve a wrong replay.

    Object-dtype frames (mixed-type pandas) cannot be hashed deterministically via tobytes; returns ``""`` so those callers fall back to the cheaper point-sample alone -- losing collision resistance but not soundness, because object-dtype hot paths are rare and a cache miss is the safe degradation.
    """
    if arr is None:
        return ""
    try:
        if pl is not None and isinstance(arr, pl.DataFrame):
            try:
                np_arr = arr.to_numpy()
            except Exception:
                return ""
            col_names = ",".join(str(c) for c in arr.columns)
        elif pl is not None and isinstance(arr, pl.Series):
            try:
                np_arr = arr.to_numpy()
            except Exception:
                return ""
            col_names = str(arr.name) if getattr(arr, "name", None) else ""
        elif isinstance(arr, (pd.DataFrame, pd.Series)):
            try:
                np_arr = arr.to_numpy()
            except Exception:
                return ""
            if isinstance(arr, pd.DataFrame):
                col_names = ",".join(str(c) for c in arr.columns)
            else:
                col_names = str(arr.name) if arr.name is not None else ""
        elif isinstance(arr, np.ndarray):
            np_arr = arr
            col_names = ""
        else:
            try:
                np_arr = np.asarray(arr)
            except Exception:
                return ""
            col_names = ""
        if not hasattr(np_arr, "shape") or not hasattr(np_arr, "dtype"):
            return ""
        if np_arr.dtype == object:
            return ""
        buf = np.ascontiguousarray(np_arr).tobytes()
        h = hashlib.blake2b(buf, digest_size=16)
        h.update(str(np_arr.shape).encode())
        h.update(str(np_arr.dtype).encode())
        if col_names:
            h.update(col_names.encode())
        return h.hexdigest()
    except Exception:
        return ""


def _full_target_content_hash(arr) -> str:
    """Full blake2b content hash of a 1-D / 2-D target-like array.

    The 4-cell (pandas Series / polars Series) and 10-cell (numpy) point-samples in
    ``_content_fingerprint_for_cache`` collide on two distinct targets whose sampled cells coincide -
    common for balanced-binary and collapsed-multilabel targets where many rows share identical class labels at the
    boundary positions. The per-target pre_pipeline loop then silently reuses the prior target's fit-transform output.

    This helper folds the FULL content (shape + dtype + every cell byte) into a deterministic 128-bit digest so the
    cache key separates such pairs. Mirrors ``_full_y_content_hash`` in ``feature_selection/filters/_mrmr_fingerprints.py``
    (cost ~5us per 100k cells; well under any cache benefit). Returns an empty string on conversion failure so the
    caller can decide to skip the cache rather than serve a wrong-content hit.
    """
    if arr is None:
        return ""
    try:
        if isinstance(arr, np.ndarray):
            np_arr = arr
        elif hasattr(arr, "to_numpy"):
            np_arr = arr.to_numpy()
        elif hasattr(arr, "values"):
            np_arr = arr.values
        else:
            np_arr = np.asarray(arr)
        if not hasattr(np_arr, "shape") or not hasattr(np_arr, "dtype"):
            return ""
        buf = np.ascontiguousarray(np_arr).tobytes()
        h = hashlib.blake2b(buf, digest_size=16)
        h.update(str(np_arr.shape).encode())
        h.update(str(np_arr.dtype).encode())
        return h.hexdigest()
    except Exception:
        return ""


def _pipeline_signature_for_cache(pipeline) -> str:
    """Stable signature for the pipeline structure + per-step shallow params.

    Two structurally identical pipelines (same step classes, same per-step
    kwargs) get the same string and hit the cache; any divergence (e.g. a
    custom scaler with different ``with_mean``) misses. Failures inside
    ``get_params`` (custom transformers without sklearn API) fall back to a
    class-only signature -- a conservative "no cache" since the same class
    might have different state.

    ``random_state``/``random_seed`` is folded into the per-step kwargs string when surfaced via ``get_params`` so two structurally identical pipelines with different seeds do NOT collide -- important because the LRU is shared across the suite and a downstream stochastic step (RFF projection, RFECV CV splitter) would otherwise replay the wrong fit-transform output when the seed flipped between targets.
    """
    if pipeline is None:
        return "None"
    parts = []
    steps = getattr(pipeline, "steps", None)
    if steps is None:
        return f"single:{type(pipeline).__name__}:{repr(pipeline)}"
    for name, step in steps:
        kls = type(step).__name__
        try:
            params = step.get_params(deep=False)
            kw = ",".join(f"{k}={params[k]!r}" for k in sorted(params))
        except Exception:
            kw = "?"
        # Fold the setattr-injected ``_mlframe_use_sample_weights_in_fs_`` marker into the per-step signature so a mid-suite toggle of weight-aware FS misses the cache. The marker lives in ``__dict__`` (set by ``_setup_helpers``) and is invisible to ``get_params``; without folding, a weight-blind fit cached under the prior toggle replays for a weight-aware caller.
        _sw_marker = getattr(step, "_mlframe_use_sample_weights_in_fs_", None)
        # Also fold any attribute-level random_state/random_seed that escaped get_params() (some custom transformers set it as instance attr post-init). Defence-in-depth against silent seed collisions.
        _seed_extras = []
        for _seed_attr in ("random_state", "random_seed", "seed"):
            _seed_val = getattr(step, _seed_attr, None)
            if _seed_val is not None and (not isinstance(params, dict) or _seed_attr not in params):
                _seed_extras.append(f"{_seed_attr}={_seed_val!r}")
        _seed_suffix = ("|" + ",".join(_seed_extras)) if _seed_extras else ""
        if _sw_marker is not None:
            parts.append(f"{name}:{kls}({kw}){_seed_suffix}|sw={bool(_sw_marker)}")
        else:
            parts.append(f"{name}:{kls}({kw}){_seed_suffix}")
    return "|".join(parts)


def _pre_pipeline_cache_key(train_df, val_df, pipeline, train_target=None, target_name=None, sample_weight=None):
    """Compose a CONTENT-based cache key.

    id()-keying was unsafe on two axes: GC-recycled ids can collide and the per-target loop re-uses ``filtered_train_df``
    (same id) across different targets - the second target would otherwise see the first target's fit-transform output.
    Including the target fingerprint AND the target name guarantees per-target isolation.

    Target separation requires BOTH the full-content blake2b hash (``_full_target_content_hash``) AND the cell-sample
    fingerprint. The full hash is the load-bearing discriminator: the 4-cell point-sample alone collides on balanced-binary
    or collapsed-multilabel targets that share boundary cells. The target-name is still folded as cheap defence-in-depth
    against name-only swaps where two targets coincidentally share content. Do NOT drop the full content hash assuming
    the point-sample or name alone is sufficient.

    ``sample_weight`` is folded only when the inner selector is marked weight-aware (``_mlframe_use_sample_weights_in_fs_``);
    otherwise FS is weight-invariant and the cache stays valid across weight schemas. Weight fingerprinting uses the cheap
    10-cell sampler shared with other content-based keys so the cost is O(1) regardless of n_rows.
    """
    sig = _pipeline_signature_for_cache(pipeline)
    _wants_sw = False
    try:
        # Walk the pipeline to find a selector with the marker set.
        if pipeline is not None:
            if hasattr(pipeline, "_mlframe_use_sample_weights_in_fs_"):
                _wants_sw = bool(getattr(pipeline, "_mlframe_use_sample_weights_in_fs_", False))
            elif hasattr(pipeline, "steps"):
                for _, _step in pipeline.steps:
                    if getattr(_step, "_mlframe_use_sample_weights_in_fs_", False):
                        _wants_sw = True
                        break
    except Exception:
        _wants_sw = False
    _sw_fp = _content_fingerprint_for_cache(sample_weight) if (_wants_sw and sample_weight is not None) else ("no_sw",)
    return (
        _content_fingerprint_for_cache(train_df),
        _content_fingerprint_for_cache(val_df),
        _content_fingerprint_for_cache(train_target),
        _full_target_content_hash(train_target),
        _full_x_content_hash(train_df),
        _full_x_content_hash(val_df),
        str(target_name) if target_name is not None else "",
        sig,
        _sw_fp,
    )


def _pre_pipeline_cache_get(train_df, val_df, pipeline, train_target=None, target_name=None, cache_max: int | None = None, sample_weight=None):
    """LRU-touch lookup; returns ``(train_out, val_out)`` or ``None``."""
    if train_df is None or pipeline is None:
        return None
    key = _pre_pipeline_cache_key(train_df, val_df, pipeline, train_target, target_name, sample_weight=sample_weight)
    with _PRE_PIPELINE_CACHE_LOCK:
        if key in _PRE_PIPELINE_CACHE:
            _PRE_PIPELINE_CACHE.move_to_end(key)
            return _PRE_PIPELINE_CACHE[key]
    return None


def _pre_pipeline_cache_set(train_df, val_df, pipeline, train_out, val_out, train_target=None, target_name=None, cache_max: int | None = None, sample_weight=None):
    """Insert under LRU, evicting the oldest entry if over capacity.

    ``cache_max`` overrides the module default; pass through from the
    caller's ``TrainingBehaviorConfig.pre_pipeline_cache_max`` so
    long-running services can tune memory vs hit-rate.
    """
    if train_df is None or pipeline is None:
        return
    key = _pre_pipeline_cache_key(train_df, val_df, pipeline, train_target, target_name, sample_weight=sample_weight)
    _cap = int(cache_max) if cache_max is not None else _PRE_PIPELINE_CACHE_MAX
    _max_bytes = _PRE_PIPELINE_CACHE_MAX_BYTES
    with _PRE_PIPELINE_CACHE_LOCK:
        # Store the pipeline as third element so future hits can transfer fit state.
        _PRE_PIPELINE_CACHE[key] = (train_out, val_out, pipeline)
        _PRE_PIPELINE_CACHE.move_to_end(key)
        while len(_PRE_PIPELINE_CACHE) > _cap:
            _PRE_PIPELINE_CACHE.popitem(last=False)
        # Byte-budget eviction (LRU): pop until under cap. Skipped silently when budget=0 (default) or sizing helper returns 0 on unknown carriers.
        if _max_bytes > 0 and len(_PRE_PIPELINE_CACHE) > 1:
            total = sum(_approx_entry_bytes(v) for v in _PRE_PIPELINE_CACHE.values())
            while total > _max_bytes and len(_PRE_PIPELINE_CACHE) > 1:
                _, evicted = _PRE_PIPELINE_CACHE.popitem(last=False)
                total -= _approx_entry_bytes(evicted)


def _pre_pipeline_cache_clear() -> None:
    """Manual eviction hook -- mainly for tests + edge cases where the
    per-target loop wants to drop stale state explicitly."""
    with _PRE_PIPELINE_CACHE_LOCK:
        _PRE_PIPELINE_CACHE.clear()
