"""
Content fingerprint + cache-key dataclasses.

The in-memory cache uses cheap session-keyed identity (id() within
suite -- safe because we hold a strong ref); content hashing happens
only at disk-cache writes, computed once at suite start in a
background thread.

Two key types:

* :class:`InMemoryKey` -- ``(session_id, df_token, train_idx_token,
  column, params_canonical_hash, provider_signature)``. All within
  one ``train_mlframe_models_suite`` call where df / train_idx are
  immutable; ``id()`` is safe because we hold strong refs throughout.

* :class:`DiskKey` -- ``(content_hash, column, params_canonical_hash,
  provider_signature)``. Used only when ``cache.persistence != "off"``.
  ``content_hash`` is :class:`ContentFingerprint`-derived, computed
  ONCE at suite start.

The :func:`fingerprint_df` builder uses a deterministic linspace
stride sample (``np.unique`` on ``np.linspace``-rounded indices
avoids duplicates for ``n < 4096``; tiny frames bypass the stride
and hash the entire frame). Universal across polars and pandas.
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import orjson

try:
    import xxhash  # type: ignore[import-untyped]
    _HAVE_XX = True
except ImportError:  # pragma: no cover -- optional accel
    xxhash = None  # type: ignore[assignment]
    _HAVE_XX = False


# Per-process memo cache for repeated ``fingerprint_df`` calls on the same frame. Key is
# ``(id(df), n_cols, columns_signature)`` where ``columns_signature`` is a cheap digest over the
# tuple of column names. Pre-fix the key was ``(id(df), n_cols)``: two frames with the same column
# count but different schemas could collide if ``id()`` from a recently-collected frame got
# recycled mid-suite (rare but possible after explicit ``del`` / GC inside a loop). Bounded to
# ``_FP_CACHE_MAX`` entries (LRU); the strong-ref guarantee that makes ``id()`` safe inside a suite
# holds here too.
#
# 128 entries × ~200 bytes per ContentFingerprint ≈ 25 KB working set -- negligible. Long-running
# Jupyter sessions that loop dozens of suites with many transient frame ids can override via
# ``MLFRAME_FP_CACHE_MAX`` so they don't thrash; default stays 128.
def _fp_cache_max_default() -> int:
    _raw = os.environ.get("MLFRAME_FP_CACHE_MAX")
    if _raw:
        try:
            _v = int(_raw)
            if _v > 0:
                return _v
        except ValueError:
            pass
    return 128


_FP_CACHE_MAX = _fp_cache_max_default()
_fingerprint_cache: "OrderedDict[Tuple[int, int, int], ContentFingerprint]" = OrderedDict()
# Module lock that serialises mutations of the fingerprint memo and the
# session token. Without it, ``_fp_cache_put`` / ``_fp_cache_get`` racing
# with ``reset_session`` (which clears the memo) can interleave through
# OrderedDict's internal links and either drop a fresh entry or revive a
# stale one; ``reset_session`` itself replaces the module-global session
# pointer non-atomically with the memo clear, so two suite starts can
# leave each other observing a half-rotated state.
_FP_LOCK = threading.Lock()


def _fp_cache_key(df: Any) -> "Optional[Tuple[int, int, int]]":
    """Build the ``(id, n_cols, columns_signature)`` cache key. Returns ``None`` on any backend
    failure so the caller skips the memo (defensive: cache miss is the safe fallback)."""
    try:
        cols = list(df.columns)
    except Exception:
        return None
    # Hash the tuple of column names so two same-id-same-count frames with different schemas
    # never collide. ``hash(tuple(...))`` is process-local but stable within one interpreter run,
    # which is exactly the scope this memo lives in.
    try:
        col_sig = hash(tuple(cols))
    except Exception:
        return None
    return (id(df), len(cols), col_sig)


def _fp_cache_get(df: Any) -> "Optional[ContentFingerprint]":
    key = _fp_cache_key(df)
    if key is None:
        return None
    with _FP_LOCK:
        val = _fingerprint_cache.get(key)
        if val is not None:
            _fingerprint_cache.move_to_end(key)
        return val


def _fp_cache_put(df: Any, fp: "ContentFingerprint") -> None:
    key = _fp_cache_key(df)
    if key is None:
        return
    with _FP_LOCK:
        _fingerprint_cache[key] = fp
        _fingerprint_cache.move_to_end(key)
        while len(_fingerprint_cache) > _FP_CACHE_MAX:
            _fingerprint_cache.popitem(last=False)


# =====================================================================
# Session token
# =====================================================================


@dataclass
class SessionToken:
    """Per-suite-call identity. Constructed at the start of
    ``train_mlframe_models_suite``; lives for the duration of that
    call.

    The point: lookup keys based on ``id(train_df)`` / ``id(train_idx)``
    are safe within one session because we hold strong refs. Across
    sessions (next process / next suite call), ``id()`` may collide
    with a freshly-allocated object -- that's OK because the
    ``session_id`` differs.
    """
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)


_CURRENT_SESSION: Optional[SessionToken] = None


def current_session() -> SessionToken:
    """Get or create the suite-level session token."""
    global _CURRENT_SESSION
    with _FP_LOCK:
        if _CURRENT_SESSION is None:
            _CURRENT_SESSION = SessionToken()
        return _CURRENT_SESSION


def reset_session() -> SessionToken:
    """Force a new session. Called at the start of every
    ``train_mlframe_models_suite`` call so cross-suite cache entries
    don't collide."""
    global _CURRENT_SESSION
    with _FP_LOCK:
        _CURRENT_SESSION = SessionToken()
        # Drop the per-process fingerprint memo under the same lock that
        # guards memo mutations: frame ids from the prior suite are no
        # longer reachable and Python may have recycled some of them.
        # Keeping the memo across sessions risks a silent stale hit on a
        # recycled id within the new suite.
        _fingerprint_cache.clear()
        return _CURRENT_SESSION


# =====================================================================
# In-memory key
# =====================================================================


@dataclass(frozen=True)
class InMemoryKey:
    """Cheap in-memory cache key.

    Safe within ONE ``train_mlframe_models_suite`` call because:
      * ``session_id`` rotates per suite call;
      * ``df_token`` = ``id(train_df)``; we hold a strong ref so
        Python won't recycle the id;
      * ``train_idx_token`` = ``id(train_idx)`` likewise;
      * ``params_canonical_hash`` is deterministic.

    Cross-session reuse is intentionally NOT supported by this key
    type -- use :class:`DiskKey` for that.
    """
    session_id: str
    df_token: int
    train_idx_token: int
    column: str
    params_canonical_hash: str
    provider_signature: str


# =====================================================================
# Disk key (with content fingerprint)
# =====================================================================


@dataclass(frozen=True)
class ContentFingerprint:
    """Content fingerprint for cross-session disk cache identity.

    Computed once at suite start (background thread when persistence
    is enabled). Components:

    * ``n_rows`` -- total row count of the source df
    * ``n_cols`` -- total column count
    * ``column_dtypes_hash`` -- blake2b of sorted ``(name, dtype)``
      tuples; catches schema changes across runs
    * ``sampled_rows_hash`` -- blake2b of stride-sampled rows
      (default 4096 rows; full hash for tiny frames)

    Less strict than full content hash (single-row edits in unsampled
    positions go undetected) but ~1000x cheaper. Round-3 user feedback:
    "subset рядов; меня устроит менее строгий, но быстрый".
    """
    n_rows: int
    n_cols: int
    column_dtypes_hash: str
    sampled_rows_hash: str

    def short(self) -> str:
        """8-char identifier for cache file paths."""
        return self.sampled_rows_hash[:8]


@dataclass(frozen=True)
class DiskKey:
    """Cross-session disk cache key. Computed once per suite via
    :func:`fingerprint_df`."""
    content: ContentFingerprint
    column: str
    params_canonical_hash: str
    provider_signature: str

    def filename(self) -> str:
        """Returns ``{fingerprint_short}__{column_hash}__{params_hash}.bin``.
        Round-3 S2 fix: column name hashed (not embedded literally)
        to neutralise path-traversal via maliciously-crafted column
        names."""
        col_hash = hashlib.blake2b(self.column.encode("utf-8"), digest_size=8).hexdigest()
        return f"{self.content.short()}__{col_hash}__{self.params_canonical_hash[:8]}.bin"


# =====================================================================
# Hashers
# =====================================================================


def canonical_params_hash(params: Any) -> str:
    """Deterministic hash of a params object / dict.

    Accepts: pydantic ``BaseModel``, dict, or scalar. Converts to a JSON-canonical-form string with
    keys sorted recursively (orjson ``OPT_SORT_KEYS``) then blake2b to 16 bytes. orjson is preferred
    over stdlib ``json`` (user memory rule `feedback_orjson_compile_regex`) and the sort-keys option
    is mandatory for any payload used in a hash (user memory rule `feedback_json_hash_sort_keys`) --
    pre-fix the scalar branch dropped ``sort_keys`` which is harmless for true scalars but a footgun
    if a caller ever passed a dict-shaped non-dict (e.g. ``OrderedDict``) down that path.

    ``default=str`` mirrors the prior stdlib fallback so custom objects coerce instead of raising.
    """
    try:
        from pydantic import BaseModel
        if isinstance(params, BaseModel):
            payload = params.model_dump_json()
            h = hashlib.blake2b(payload.encode("utf-8"), digest_size=16)
            return h.hexdigest()
    except ImportError:  # pragma: no cover
        pass
    payload = orjson.dumps(
        params,
        option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        default=str,
    ).decode("utf-8")
    h = hashlib.blake2b(payload.encode("utf-8"), digest_size=16)
    return h.hexdigest()


def fingerprint_df(
    df: Any,
    columns: Optional[Tuple[str, ...]] = None,
    n_sample: int = 4096,
) -> ContentFingerprint:
    """Compute a :class:`ContentFingerprint` for a polars or pandas
    DataFrame, sampling deterministically.

    Round-3 R2-3 fix: ``np.linspace(...).round().astype(int64)`` then
    ``np.unique`` to dedupe so tiny frames don't degenerate.

    Caching: a per-process LRU memo (cleared at ``reset_session`` boundaries) hits when the same
    frame is fingerprinted twice, avoiding the costly Arrow/IPC pass entirely. Only fires when
    ``columns`` is ``None`` because a caller-supplied column subset would change the result.

    Hash strategy: when ``xxhash`` is available we use ``DataFrame.hash_rows`` -> uint64 Series ->
    ``xxh3_64`` on its bytes -- the bench in ``_benchmarks/bench_fingerprint.py`` records this as
    100-700x faster than the legacy ``to_arrow().to_pandas().to_csv()`` triple-convert. Falls back to
    the legacy path when xxhash isn't installed so caller's deps stay minimal.
    """
    cached = _fp_cache_get(df) if columns is None else None
    if cached is not None:
        return cached
    n_rows = len(df)

    # Determine columns (sorted for deterministic dtype hash).
    if columns is None:
        try:
            cols = list(df.columns)
        except Exception:
            cols = []
    else:
        cols = list(columns)
    # Wave 61 (2026-05-20): pandas tolerates mixed-type column labels (e.g.
    # [0, "a", 1] from a stitched join / pivot). sorted() raises TypeError on
    # str-vs-int; coerce via str-key so fingerprinting stays robust.
    cols_sorted = sorted(cols, key=str)
    n_cols = len(cols_sorted)

    # Stride-sample row indices.
    if n_rows == 0:
        sample_idx = np.array([], dtype=np.int64)
    elif n_rows <= max(1, n_sample):
        sample_idx = np.arange(n_rows, dtype=np.int64)
    else:
        raw = np.linspace(0, n_rows - 1, n_sample).round().astype(np.int64)
        sample_idx = np.unique(raw)

    # ---- column_dtypes_hash --------------------------------------
    dtype_lines = []
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            for c in cols_sorted:
                dtype_lines.append(f"{c}:{df.schema[c]}")
    except ImportError:  # pragma: no cover
        pass
    if not dtype_lines:
        try:
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                for c in cols_sorted:
                    dtype_lines.append(f"{c}:{df[c].dtype}")
        except ImportError:  # pragma: no cover
            pass
    column_dtypes_hash = hashlib.blake2b(
        ";".join(dtype_lines).encode("utf-8"), digest_size=8
    ).hexdigest()

    # ---- sampled_rows_hash ---------------------------------------
    payload_parts: list = [str(n_rows).encode(), str(n_cols).encode()]
    if len(sample_idx) > 0:
        try:
            import polars as pl
            if isinstance(df, pl.DataFrame):
                # polars accepts numpy int64 indices via gather()
                sub = df.select(cols_sorted)[sample_idx.tolist()]
                if _HAVE_XX:
                    # Fast path: ~100-700x faster than to_csv (see bench_fingerprint.py).
                    # ``hash_rows`` is the polars-native rowwise xxhash; we re-hash its bytes to
                    # collapse N×u64 into a fingerprint identical-length to the legacy blake path.
                    payload_parts.append(sub.hash_rows().to_numpy().tobytes())
                else:
                    # xxhash-absent fallback. Legacy
                    # ``to_arrow().to_pandas().to_csv()`` materialised an
                    # entire pandas DataFrame and CSV-encoded every cell
                    # - ~45 ms / 173 kB on a 4096-row 5-column frame.
                    # ``write_ipc`` to a BytesIO is the polars-native Arrow
                    # IPC stream: zero-copy from the polars buffer, byte-
                    # deterministic across runs, and ~140x faster (~0.3 ms
                    # on the same frame).
                    buf = io.BytesIO()
                    sub.write_ipc(buf, compression="uncompressed")
                    payload_parts.append(buf.getvalue())
            else:
                raise TypeError("not polars")
        except (ImportError, TypeError):
            try:
                import pandas as pd
                if isinstance(df, pd.DataFrame):
                    sub = df.iloc[sample_idx][cols_sorted]
                    if _HAVE_XX:
                        # pandas fast path: hash each column's underlying numpy buffer via xxhash
                        # then xxhash the concatenated digests. Avoids a CSV materialisation.
                        h_outer = xxhash.xxh3_64()
                        for c in cols_sorted:
                            try:
                                buf = sub[c].to_numpy().tobytes()
                            except Exception:
                                buf = str(sub[c].to_list()).encode("utf-8")
                            h_outer.update(xxhash.xxh3_64(buf).digest())
                        payload_parts.append(h_outer.digest())
                    else:
                        # xxhash-absent pandas fallback. CSV-encoding
                        # every cell was the original hot path; per-column
                        # numpy ``tobytes`` skips the string roundtrip on
                        # numeric columns and only falls back to
                        # ``str(to_list())`` for object columns where
                        # ``tobytes`` is undefined.
                        parts: list = []
                        for c in cols_sorted:
                            try:
                                parts.append(sub[c].to_numpy().tobytes())
                            except Exception:
                                parts.append(str(sub[c].to_list()).encode("utf-8"))
                            parts.append(f"|{c}|".encode("utf-8"))
                        payload_parts.append(b"".join(parts))
            except ImportError:  # pragma: no cover
                pass

    h = hashlib.blake2b(digest_size=16)
    for p in payload_parts:
        h.update(p)
    sampled_rows_hash = h.hexdigest()

    fp = ContentFingerprint(
        n_rows=n_rows,
        n_cols=n_cols,
        column_dtypes_hash=column_dtypes_hash,
        sampled_rows_hash=sampled_rows_hash,
    )
    if columns is None:
        _fp_cache_put(df, fp)
    return fp


__all__ = [
    "SessionToken",
    "current_session",
    "reset_session",
    "InMemoryKey",
    "ContentFingerprint",
    "DiskKey",
    "canonical_params_hash",
    "fingerprint_df",
]
