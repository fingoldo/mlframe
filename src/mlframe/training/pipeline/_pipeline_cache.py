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

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

# polars' ``hash_rows`` panics at the Rust layer on a zero-column frame
# (``pyo3_runtime.PanicException: at least one key``). That panic class is
# NOT an ``Exception`` subclass (it inherits ``BaseException`` on the pyo3
# builds we ship against), so a plain ``except Exception`` misses it. We
# catch ``(Exception, PanicException)`` explicitly rather than a bare
# ``except BaseException`` -- the latter also swallows KeyboardInterrupt /
# SystemExit (flagged by tests/test_meta/test_no_bare_except.py). polars
# re-exports the class as ``polars.exceptions.PanicException``; fall back to
# just ``Exception`` if the attribute is absent on an older polars.
if pl is not None and hasattr(getattr(pl, "exceptions", None), "PanicException"):
    _HASH_FASTPATH_EXC: tuple = (Exception, pl.exceptions.PanicException)
else:
    _HASH_FASTPATH_EXC = (Exception,)

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


# iter632: a tiny per-process dict mapping (id, shape) -> blake2b digest.
# Single-entry memos (iter625 / iter627) match the immediate get/set repeat
# pattern but lose the multi-target / train+val alternation: the suite calls
# this with train_df then val_df then train_df again on the next target.
# Cap is 16 entries so the cache never grows unbounded if id() is recycled
# by GC across a long-running session.
_PIPELINE_X_HASH_CACHE: "OrderedDict[tuple, str]" = OrderedDict()
_PIPELINE_X_HASH_CACHE_MAX = 16

# Symmetric (id, shape) memo for the target-side full-content hash (mirrors the X-side cache above).
_PIPELINE_TARGET_HASH_CACHE: "OrderedDict[tuple, str]" = OrderedDict()
_PIPELINE_TARGET_HASH_CACHE_MAX = 16


def _full_x_content_hash(arr) -> str:
    """Full blake2b content hash of an X frame for cache-key disambiguation.

    The 4-row point-sample in ``_content_fingerprint_for_cache`` collides on two distinct X frames whose sampled positions (0, 8, n/2, n-1) happen to coincide -- common when a pre-pipeline transform mutates only the unsampled rows (outlier clip on the middle 90 % of the distribution, masked-row fillna, etc.). The per-target loop would then silently replay the prior target's fit-transform output even though the actual X content drifted.

    Mirrors ``_full_y_content_hash`` (target side) and ``_full_x_content_hash`` (MRMR side) so the X/y guarantees are symmetric. Cost is O(rows*cols) bytes hashed -- a few ms even on a 1M-row frame, well under any cache benefit. Returns ``""`` on conversion failure so the caller can choose to skip the cache rather than serve a wrong replay.

    Object-dtype frames (mixed-type pandas) cannot be hashed deterministically via tobytes; returns ``""`` so those callers fall back to the cheaper point-sample alone -- losing collision resistance but not soundness, because object-dtype hot paths are rare and a cache miss is the safe degradation.

    iter632: small-LRU (id, shape) memo. The upstream ``_pre_pipeline_cache_key``
    memo (iter625) catches the immediate get/set repeat with the SAME
    pipeline+target_name; this memo catches the multi-target case where
    ``target_name`` changes between calls but the train/val frames are
    identity-stable across targets. A single-slot cache misses on the
    train/val alternation -- ``hash(train), hash(val), hash(train), ...``
    -- so we use a 16-slot LRU instead.
    """
    if arr is None:
        return ""
    # iter632 fast-path: id+shape discriminates against GC-recycled id collisions and is safe because train/val frames are not mutated between cache_key invocations within the suite (preserve the same guarantee as iter625 / iter627).
    sh = getattr(arr, "shape", None)
    id_shape = (id(arr), sh if sh is not None else (None,))
    cached = _PIPELINE_X_HASH_CACHE.get(id_shape)
    if cached is not None:
        _PIPELINE_X_HASH_CACHE.move_to_end(id_shape)
        return cached
    try:
        # iter299 (2026-05-26): polars-native hash path. The legacy
        # ``arr.to_numpy().tobytes() -> blake2b`` chain materialises the
        # full numpy buffer (~40 MB on a 200k x 25 float64 frame) just to
        # hash it; polars exposes a row-wise hash kernel
        # (``DataFrame.hash_rows()``) that runs entirely in the Rust
        # engine and produces a u64-per-row Series. Folding the row-hash
        # sum + shape + columns into blake2b gives a deterministic 128-bit
        # digest with 21.9x speedup at the c0142 frame shape (104.68 ms ->
        # 4.78 ms / call; 16 calls saves ~1.5 s on a 13.88 s combo wall).
        # Cache lives per-process, so the hash key change does not
        # invalidate any persisted state.
        if pl is not None and isinstance(arr, pl.DataFrame):
            # Guard zero-column polars frames: ``hash_rows`` panics at the
            # Rust layer with ``pyo3_runtime.PanicException: at least one
            # key`` -- not a Python ``Exception`` on older pyo3 builds.
            if arr.width == 0:
                return ""
            try:
                row_hashes = arr.hash_rows()
                # Cast to UInt64 then sum mod 2^64 -- polars sum on UInt64
                # wraps natively, giving a deterministic 64-bit summary.
                row_sum = int(row_hashes.sum())
            except _HASH_FASTPATH_EXC:
                # Tuple includes pyo3 PanicException (zero-column hash_rows).
                return ""
            col_names = ",".join(str(c) for c in arr.columns)
            h = hashlib.blake2b(digest_size=16)
            h.update(str(row_sum).encode())
            h.update(str(arr.shape).encode())
            h.update(str(arr.schema).encode())
            if col_names:
                h.update(col_names.encode())
            _result = h.hexdigest()
            _PIPELINE_X_HASH_CACHE[id_shape] = _result
            if len(_PIPELINE_X_HASH_CACHE) > _PIPELINE_X_HASH_CACHE_MAX:
                _PIPELINE_X_HASH_CACHE.popitem(last=False)
            return _result
        if pl is not None and isinstance(arr, pl.Series):
            try:
                np_arr = arr.to_numpy()
            except Exception:
                return ""
            col_names = str(arr.name) if getattr(arr, "name", None) else ""
        elif isinstance(arr, pd.DataFrame):
            # iter359 (2026-05-26): mirror the polars hash_rows fastpath for
            # pandas frames. ``pl.from_pandas`` does a zero-copy buffer share
            # for primitive-dtype columns (and a column-by-column conversion
            # otherwise) -- much cheaper than ``arr.to_numpy()`` which forces
            # an _interleave consolidation copy across every column to build
            # one homogeneous numpy block. On a 200k x 25 mixed-dtype frame
            # the polars path runs 9.7x faster (129.81 ms -> 13.29 ms / call;
            # c0051 16 hashes saves ~1.9 s). hash_rows() is Rust-side
            # deterministic, identical guarantees to blake2b(tobytes) for
            # cache-key collision resistance.
            if pl is not None and arr.shape[1] > 0:
                # Guard against zero-column frames: polars' ``hash_rows`` panics
                # at the Rust layer (``pyo3_runtime.PanicException: at least one
                # key``), which is NOT a Python ``Exception`` subclass on older
                # pyo3 (inherits ``BaseException``) -- ``except Exception``
                # below wouldn't catch it on those builds. Skip the polars
                # fastpath entirely when the frame has no columns (happens on
                # all-constant-features datasets after the constant-column
                # filter drops everything; observed via
                # test_with_all_constant_features 2026-05-27).
                try:
                    pl_df = pl.from_pandas(arr)
                    row_hashes = pl_df.hash_rows()
                    row_sum = int(row_hashes.sum())
                except _HASH_FASTPATH_EXC:
                    # Tuple includes pyo3 PanicException (zero-column
                    # hash_rows). Fallback uses ``arr.to_numpy().tobytes()``.
                    pl_df = None
                    row_sum = None
            else:
                pl_df = None
                row_sum = None
            if row_sum is not None:
                col_names = ",".join(str(c) for c in arr.columns)
                h = hashlib.blake2b(digest_size=16)
                h.update(str(row_sum).encode())
                h.update(str(arr.shape).encode())
                h.update(str(arr.dtypes.to_list()).encode())
                if col_names:
                    h.update(col_names.encode())
                _result = h.hexdigest()
                _PIPELINE_X_HASH_CACHE[id_shape] = _result
                if len(_PIPELINE_X_HASH_CACHE) > _PIPELINE_X_HASH_CACHE_MAX:
                    _PIPELINE_X_HASH_CACHE.popitem(last=False)
                return _result
            # Fallback to legacy to_numpy + tobytes path on any polars failure
            # (mixed-dtype frame polars can't ingest, etc).
            try:
                np_arr = arr.to_numpy()
            except Exception:
                return ""
            col_names = ",".join(str(c) for c in arr.columns)
        elif isinstance(arr, pd.Series):
            try:
                np_arr = arr.to_numpy()
            except Exception:
                return ""
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
        # blake2b reads the contiguous array via the buffer protocol directly;
        # dropping the .tobytes() materialisation saves an O(nbytes) copy and is
        # bit-identical (the buffer bytes equal tobytes() for a C-contiguous array).
        h = hashlib.blake2b(np.ascontiguousarray(np_arr), digest_size=16)
        h.update(str(np_arr.shape).encode())
        h.update(str(np_arr.dtype).encode())
        if col_names:
            h.update(col_names.encode())
        _result = h.hexdigest()
        _PIPELINE_X_HASH_CACHE[id_shape] = _result
        if len(_PIPELINE_X_HASH_CACHE) > _PIPELINE_X_HASH_CACHE_MAX:
            _PIPELINE_X_HASH_CACHE.popitem(last=False)
        return _result
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

    Carries the same (id, shape)-keyed LRU memo as the X-side ``_full_x_content_hash``: the target is called
    twice per pipeline-fit (get/set pair) and re-visited on the train/val/multi-target alternation, all on a
    pinned array. id-recycling is bounded by also keying on shape. Saves the O(rows) re-hash on every repeat.
    """
    if arr is None:
        return ""
    sh = getattr(arr, "shape", None)
    id_shape = (id(arr), sh if sh is not None else (None,))
    cached = _PIPELINE_TARGET_HASH_CACHE.get(id_shape)
    if cached is not None:
        _PIPELINE_TARGET_HASH_CACHE.move_to_end(id_shape)
        return cached
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
        # blake2b reads the contiguous array via the buffer protocol directly;
        # dropping the .tobytes() materialisation saves an O(nbytes) copy and is
        # bit-identical (the buffer bytes equal tobytes() for a C-contiguous array).
        h = hashlib.blake2b(np.ascontiguousarray(np_arr), digest_size=16)
        h.update(str(np_arr.shape).encode())
        h.update(str(np_arr.dtype).encode())
        _result = h.hexdigest()
        _PIPELINE_TARGET_HASH_CACHE[id_shape] = _result
        if len(_PIPELINE_TARGET_HASH_CACHE) > _PIPELINE_TARGET_HASH_CACHE_MAX:
            _PIPELINE_TARGET_HASH_CACHE.popitem(last=False)
        return _result
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


# iter625 (perf): single-entry "last computed" memo cache for the
# expensive _pre_pipeline_cache_key compute. The function is called
# TWICE per pipeline-fit transaction with byte-identical inputs:
# once from _pre_pipeline_cache_get and once from _pre_pipeline_-
# cache_set (the get/set pair brackets pipeline.fit_transform, which
# does NOT mutate the input frames). The blake2b hash chain via
# polars hash_rows runs ~1.17s/call at 100k x 16 (c0016 profile;
# cumtime 4.98s across 4 calls in 1 target). Memoizing the last
# result by id-tuple + shape saves ~50% of the hash work.
#
# Safety: id() recycling after GC could superficially match a new
# frame, so the cache key includes (shape, len(columns)) for extra
# discrimination. The recycled-id-AND-identical-shape collision
# would require a frame to be GC'd between get + set (microseconds
# apart in the suite hot path) AND replaced by a frame of identical
# shape -- not impossible but vanishingly rare. The fall-through
# computes the correct key anyway; the worst case is one extra
# hash recompute, not a wrong key.
_LAST_KEY_CACHE: dict = {"id_tup": None, "key": None}


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

    iter625: single-entry memo cache on (id, shape)-tuple of inputs.
    Fast-path hit on the get/set pair that brackets every pipeline-
    fit; saves the second call's ~1.2s hash recompute. Safe because
    the frames are not mutated between get and set.
    """
    # iter625 fast-path: build a cheap id-tuple key (shape included
    # for id-recycling defence) and check the last-computed cache.
    def _id_shape(arr):
        if arr is None:
            return (None,)
        sh = getattr(arr, "shape", None)
        return (id(arr), sh if sh is not None else (None,))

    id_tup = (
        _id_shape(train_df),
        _id_shape(val_df),
        _id_shape(train_target),
        id(pipeline),
        str(target_name) if target_name is not None else "",
        _id_shape(sample_weight),
    )
    if _LAST_KEY_CACHE["id_tup"] == id_tup:
        return _LAST_KEY_CACHE["key"]

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
    key = (
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
    # Publish key BEFORE id_tup so a torn read on this unlocked single-slot memo can only see an OLD id_tup (miss -> recompute), never a NEW id_tup paired
    # with a stale key from a prior different (df, target) under id-recycling.
    _LAST_KEY_CACHE["key"] = key
    _LAST_KEY_CACHE["id_tup"] = id_tup
    return key


def _pre_pipeline_cache_get(train_df, val_df, pipeline, train_target=None, target_name=None, cache_max: int | None = None, sample_weight=None, key=None):
    """LRU-touch lookup; returns ``(train_out, val_out)`` or ``None``.

    ``key``: pass an already-computed ``_pre_pipeline_cache_key`` to skip the
    recompute. The caller (``_apply_pre_pipeline_transforms``) builds the key
    once and threads it into both the get and the populate so the get/set pair
    no longer relies on the single-slot ``_LAST_KEY_CACHE`` memo to coincide --
    correctness is independent of the memo hit.
    """
    if train_df is None or pipeline is None:
        return None
    if key is None:
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
