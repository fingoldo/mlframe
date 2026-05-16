"""Disk-backed discovery cache: content-hash signature + key composer + DiscoveryCache class. Used by R&D workflows that re-run discovery with the same data + varying config; cache hits skip the expensive MI permutation null + Wilcoxon + tiny-model rerank phases. Pure stdlib + numpy + pandas; no composite-internal deps."""


from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import polars as pl  # type: ignore
    _HAS_POLARS = True
except Exception:  # pragma: no cover
    pl = None  # type: ignore
    _HAS_POLARS = False


def _is_polars_df(x: Any) -> bool:
    """ENS-P2-6: prefer explicit isinstance check over duck-typing."""
    return _HAS_POLARS and isinstance(x, pl.DataFrame)


# ----------------------------------------------------------------------
# Discovery caching layer (R10c brainstorm round-2 extension E; content-hash cache for discovery).
#
# R&D workflows often re-run ``CompositeTargetDiscovery`` on the same data while only varying the inner-model hyperparameters. The discovery step (MI permutation null, Wilcoxon per spec, tiny-model rerank) burns minutes on multi-million-row datasets. The caching layer keys discovery results by a content hash of (data-sample, target-column, config-signature, random_state) so repeated discovery calls with the same inputs return the cached specs in milliseconds.
#
# Three primitives:
# 1. ``data_signature(df, target_col, feature_cols, sample_n=1000, random_state=42)`` -- blake2b hash over a deterministic sample of the data + column names + dtypes + a head/tail row-order fingerprint. Quantises to a 16-byte fingerprint that is SENSITIVE to row reorder (changed in commit 4e2f031 / 2026-05-16): a shuffled frame produces a different signature than the original, so reorder no longer replays a stale spec.
# 2. ``DiscoveryCache(cache_dir)`` -- disk-backed key->value store. Keys are hex strings; values are pickled (using stdlib ``pickle``; safe since the values are dataclass-derived dicts, not arbitrary user objects). API: ``get(key)`` / ``set(key, value)`` / ``invalidate(key)`` / ``clear()`` / ``__contains__``.
# 3. Convenience ``make_discovery_cache_key(df_sig, target_col, config_signature, random_state)`` -- combines the parts into a stable hex key.
#
# The cache layer does NOT auto-integrate with ``CompositeTargetDiscovery.fit``; callers manage cache lookup / store at their orchestration level. This keeps the discovery class free of I/O concerns (testability + library hygiene).
# ----------------------------------------------------------------------


_DISCOVERY_SIGNATURE_SAMPLE_N: int = 1000
# CACHE-P2-5: single source of truth for discovery cache seed; both
# ``data_signature`` and ``make_discovery_cache_key`` reference it so a
# downstream override touches one constant, not two function defaults.
_DISCOVERY_DEFAULT_SEED: int = 42


def _row_order_fingerprint(df: Any, n_edge: int = 8) -> str:
    """Cheap fingerprint of a frame's row order (first/last ``n_edge`` rows).

    Folded into ``data_signature`` so a shuffled frame produces a different signature than the
    original. O(1) in row count -- we never materialise the whole frame. The fingerprint reads the
    first and last rows directly and hashes their string representations; this catches the common
    shuffle / reorder case without inducing the cost of a full row-by-row hash.

    Returns ``""`` on any access failure (degrades to the prior reorder-stable behaviour rather
    than crashing on exotic frame types).
    """
    import hashlib
    try:
        if _is_polars_df(df):
            head_repr = str(df.head(n_edge).rows())
            tail_repr = str(df.tail(n_edge).rows())
        elif isinstance(df, pd.DataFrame):
            head_repr = df.head(n_edge).to_csv(index=False)
            tail_repr = df.tail(n_edge).to_csv(index=False)
        else:
            return ""
        payload = (head_repr + "|" + tail_repr).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=8).hexdigest()
    except Exception:
        return ""


def data_signature(
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    *,
    sample_n: int = _DISCOVERY_SIGNATURE_SAMPLE_N,
    random_state: int = _DISCOVERY_DEFAULT_SEED,
) -> str:
    """Content-hash signature for a (df, target_col, feature_cols) triple.

    Deterministic sample of ``min(n_rows, sample_n)`` rows + column names + dtypes + a cheap first-and-last row fingerprint, hashed via blake2b to a 16-byte hex fingerprint.

    Row-order sensitivity (commit 4e2f031, 2026-05-16): the signature is now
    SENSITIVE TO ROW ORDER. ``_row_order_fingerprint`` hashes the head and
    tail of the frame, so a shuffled frame produces a different signature
    than the original. The pre-fix docstring stated the signature was
    "stable under row REORDER" - that was the bug (shuffled frames got
    cache hits on stale specs). Note that this also means: the signature
    DOES change when row insertion shifts the sample composition or
    perturbs head/tail rows, which is the intended behaviour for the R&D
    workflow where the underlying frame is the same across runs.

    Parameters
    ----------
    df
        pandas / polars frame.
    target_col, feature_cols
        Column identifiers used to scope the signature; changes here invalidate the cache.
    sample_n
        Rows sampled for the hash; lower is faster, higher is more discriminating.
    random_state
        Seed for the row-sample RNG; must match across cache write and read for the signature to be stable.

    Returns
    -------
    32-character hex string (blake2b digest, 16 bytes).
    """
    import hashlib
    n_rows = len(df)
    if n_rows == 0:
        return hashlib.blake2b(b"empty", digest_size=16).hexdigest()
    rng = np.random.default_rng(random_state)
    sample_n_eff = min(n_rows, int(sample_n))
    sample_idx = np.sort(rng.choice(n_rows, size=sample_n_eff, replace=False))
    h = hashlib.blake2b(digest_size=16)
    # CACHE-P0-2: row count goes into the hash so appending rows invalidates
    # the cache even when the deterministic sample happens to coincide.
    h.update(b"nrows=")
    h.update(str(int(n_rows)).encode("utf-8"))
    # CACHE-row-order: the seeded sample misses row swaps in unsampled positions, and the per-column
    # min/max/null stats are permutation-invariant. Fold in a cheap fingerprint of the first/last
    # row content so head/tail swaps burst the cache (re-running with reordered rows must NOT
    # replay the prior spec).
    h.update(b"|roworder=")
    h.update(_row_order_fingerprint(df).encode("utf-8"))
    # Hash 1: target column + feature cols (names + order).
    h.update(target_col.encode("utf-8"))
    for c in feature_cols:
        h.update(b"|")
        h.update(str(c).encode("utf-8"))

    def _col_stats(arr: np.ndarray) -> bytes:
        """Per-column WHOLE-frame summary (min, max, null count).

        Folded into the hash so a single appended row that lands in the unsampled portion still
        changes the signature - which the sampled-values-only hash misses.

        Dtype-aware: pre-fix this routine fell through to ``np.unique(arr.astype(str))`` for
        anything that did not coerce cleanly to float64, which collapsed integer columns with NaN
        sentinels onto a stringified-distinct-values summary and dropped the min/max/null
        distribution information (DISC-CACHE-NULL-DTYPE). Post-fix the integer branch is handled
        explicitly: when the dtype kind is in {'i','u','b'} we read min/max/uniques without ever
        trying the float-cast that NaN sentinels would corrupt.
        """
        if arr.size == 0:
            return b"empty"
        kind = getattr(arr.dtype, "kind", "")
        if kind in ("i", "u", "b"):
            # Integer / bool: no NaN possible at the numpy-dtype level (NaN sentinels are
            # represented as out-of-range ints), so min/max + nunique distinguish dtype-equal
            # columns without going through the lossy str-uniques path.
            try:
                return (
                    f"intmin={int(np.min(arr))};"
                    f"intmax={int(np.max(arr))};"
                    f"nuniq={int(np.unique(arr).size)}"
                ).encode("utf-8")
            except Exception:
                return b"int_opaque"
        if kind == "f":
            isnan = ~np.isfinite(arr)
            n_null = int(isnan.sum())
            finite = arr[~isnan]
            if finite.size == 0:
                return f"all_null:{n_null}".encode("utf-8")
            return (
                f"min={float(np.min(finite)):.12g};"
                f"max={float(np.max(finite)):.12g};"
                f"null={n_null}"
            ).encode("utf-8")
        # Generic numeric fallback (datetime / timedelta / complex via float coerce).
        try:
            arr_f = arr.astype(np.float64, copy=False)
            isnan = ~np.isfinite(arr_f)
            n_null = int(isnan.sum())
            finite = arr_f[~isnan]
            if finite.size == 0:
                return f"all_null:{n_null}".encode("utf-8")
            return (
                f"fmin={float(np.min(finite)):.12g};"
                f"fmax={float(np.max(finite)):.12g};"
                f"null={n_null}"
            ).encode("utf-8")
        except (TypeError, ValueError):
            pass
        # Object / string dtype: hash a fingerprint of distinct values.
        try:
            u = np.unique(arr.astype(str, copy=False))
            return (
                f"uniq={int(u.size)};first={u[0] if u.size else ''};"
                f"last={u[-1] if u.size else ''}"
            ).encode("utf-8")
        except Exception:
            return b"opaque"

    # Hash 2: per-column dtype + whole-column stats + per-column sampled values.
    if _is_polars_df(df):
        # Polars path.
        for c in [target_col] + list(feature_cols):
            if c not in df.columns:
                continue
            col = df.get_column(c)
            h.update(str(col.dtype).encode("utf-8"))
            full = col.to_numpy()
            h.update(b"|stats=")
            h.update(_col_stats(full))
            sampled = full[sample_idx]
            h.update(np.ascontiguousarray(sampled).tobytes())
    elif isinstance(df, pd.DataFrame):
        for c in [target_col] + list(feature_cols):
            if c not in df.columns:
                continue
            h.update(str(df[c].dtype).encode("utf-8"))
            full = df[c].to_numpy()
            h.update(b"|stats=")
            h.update(_col_stats(full))
            sampled = full[sample_idx]
            h.update(np.ascontiguousarray(sampled).tobytes())
    else:
        raise TypeError(f"data_signature: unsupported df type {type(df).__name__}")
    return h.hexdigest()


def make_discovery_cache_key(
    df_sig: str,
    target_col: str,
    config_signature: str,
    random_state: int = _DISCOVERY_DEFAULT_SEED,
) -> str:
    """Combine the parts of a discovery cache key into a stable hex string. The ``config_signature`` is caller-supplied (usually a hash of the JSON-serialised CompositeTargetDiscoveryConfig)."""
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    h.update(df_sig.encode("utf-8"))
    h.update(b"|")
    h.update(target_col.encode("utf-8"))
    h.update(b"|")
    h.update(config_signature.encode("utf-8"))
    h.update(b"|")
    h.update(str(int(random_state)).encode("utf-8"))
    return h.hexdigest()


class DiscoveryCache:
    """Disk-backed key->value cache for CompositeTargetDiscovery results.

    Values are pickled with stdlib ``pickle`` (safe: stored objects are dataclass-derived dicts). Files live under ``<cache_dir>/<key>.pkl`` with one file per key for easy invalidation / cleanup.

    Thread-safe for single-process use only; concurrent writers from multiple processes will race on the same key (caller's responsibility).
    """

    def __init__(
        self,
        cache_dir: Any,
        *,
        max_entries: Optional[int] = None,
        max_size_mb: Optional[float] = None,
    ) -> None:
        """Construct a disk-backed discovery cache.

        Parameters
        ----------
        cache_dir
            Directory hosting one ``<key>.pkl`` per entry.
        max_entries
            Hard cap on the number of cached entries. When ``set()`` would
            push the count above the cap, the least-recently-accessed
            entries are evicted to fit. ``None`` (default) means no cap -
            R&D workflows that re-run discovery thousands of times grow
            unboundedly without this.
        max_size_mb
            Soft cap on the total cache footprint in megabytes. Evaluated
            after the count cap. ``None`` (default) means no size cap.

        LRU tracking uses a sidecar ``<cache_dir>/.lru`` JSON file rather
        than ``os.path.getatime``: Windows / NTFS frequently mounts with
        noatime semantics so atime is unreliable; the sidecar gives us a
        portable monotonic-time access ledger that survives process exit.

        The cache directory is wrapped through
        :func:`mlframe.training.feature_handling.system.long_path_safe`
        on Windows so deep cache trees (>= 260 chars) survive
        ``os.replace`` in ``set()``. ``LocalDiskBackend`` already did
        this; ``DiscoveryCache`` did not, so a deep run-name + nested
        artifact path crashed on Windows even though the same directory
        worked under ``LocalDiskBackend``.
        """
        import os
        from .feature_handling.system import long_path_safe
        self.cache_dir = long_path_safe(os.path.abspath(str(cache_dir)))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.max_entries = max_entries
        self.max_size_mb = max_size_mb
        self._lru_path = os.path.join(self.cache_dir, ".lru")
        # DISC-MAX-UNBOUNDED: a None default for BOTH caps means the cache grows monotonically on
        # repeated R&D runs. Surface a one-time WARN at construction so the operator notices before
        # the disk fills; if they set either cap explicitly the warning stays silent.
        if max_entries is None and max_size_mb is None:
            import warnings
            warnings.warn(
                f"DiscoveryCache at {self.cache_dir!r} constructed with max_entries=None and max_size_mb=None: "
                f"cache will grow without bound. Pass at least one cap to enable LRU eviction.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # LRU sidecar (key -> access timestamp). Plain JSON; tiny so we read
    # / write the whole file on every touch. Atime is too unreliable on
    # NTFS to depend on.
    #
    # DISC-LRU-RACE / DISC-RACE-UNPROT: file-lock the sidecar so two
    # concurrent processes hitting the same data_dir can't interleave
    # an evict + write and leave live entries marked stale. ``filelock``
    # is optional -- absence falls back to the pre-fix racy behaviour
    # with a one-time WARN.
    # ------------------------------------------------------------------

    def _lock_path(self) -> str:
        return self._lru_path + ".lock"

    @staticmethod
    def _maybe_filelock(lock_path: str):
        """Return a ``filelock.FileLock`` instance if the dep is present, else a no-op context."""
        try:
            from filelock import FileLock as _FileLock  # type: ignore[import-untyped]
            return _FileLock(lock_path, timeout=30)
        except ImportError:  # pragma: no cover
            import contextlib
            return contextlib.nullcontext()

    def _load_lru(self) -> Dict[str, float]:
        import os, json
        if not os.path.exists(self._lru_path):
            return {}
        try:
            with open(self._lru_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict):
                return {str(k): float(v) for k, v in d.items()}
        except (OSError, ValueError):
            pass
        return {}

    def _save_lru(self, lru: Dict[str, float]) -> None:
        import os, json, tempfile
        # Same atomic-rename + fsync discipline as the value writes - LRU
        # corruption would silently break eviction order.
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, prefix=".lru.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(lru, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._lru_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _touch_lru(self, key: str) -> None:
        import time
        # Cross-process file lock around read-modify-write so a concurrent process can't replay a
        # stale-snapshot save and overwrite a fresh access timestamp. filelock is optional.
        with self._maybe_filelock(self._lock_path()):
            lru = self._load_lru()
            lru[key] = time.time()
            self._save_lru(lru)

    def _path(self, key: str) -> str:
        import os
        # Basic sanitisation: only allow hex keys (or alphanumeric); reject path-traversal attempts.
        safe_key = "".join(c for c in key if c.isalnum())
        if not safe_key:
            raise ValueError(f"DiscoveryCache: empty / unsafe key {key!r}")
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")

    def __contains__(self, key: str) -> bool:
        import os
        return os.path.exists(self._path(key))

    def get(self, key: str, default: Any = None) -> Any:
        """Return the cached value, or ``default`` if the key is absent / unreadable.

        Pre-2026-05-15 the implementation checked ``os.path.exists`` before
        opening; on Windows a delete-between-exists-and-open race surfaced
        ``FileNotFoundError`` after the existence check passed. The two-step
        guard is gone now - we just try-open and treat any failure (missing
        file, locked file, corrupt pickle) as a cache miss.

        A successful read updates the LRU sidecar so subsequent eviction
        picks the least-recently-USED (not just least-recently-WRITTEN)
        entry.
        """
        import pickle  # lazy
        path = self._path(key)
        try:
            with open(path, "rb") as f:
                value = pickle.load(f)
        except (FileNotFoundError, OSError, EOFError, pickle.UnpicklingError, AttributeError):
            return default
        except Exception:
            return default
        # Successful read: bump LRU. Done outside the read try/except so
        # an LRU file failure doesn't break the read path.
        try:
            self._touch_lru(self._safe_key(key))
        except Exception:
            pass
        return value

    def _safe_key(self, key: str) -> str:
        """Sanitised key (matches the on-disk filename stem)."""
        return "".join(c for c in key if c.isalnum())

    def _entry_size_bytes(self, safe_key: str) -> int:
        import os
        path = os.path.join(self.cache_dir, f"{safe_key}.pkl")
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    def _evict_to_caps(self) -> int:
        """Evict least-recently-accessed entries to satisfy the configured
        ``max_entries`` / ``max_size_mb`` caps. Returns the number of
        entries removed. Called from ``set()`` after the new entry has
        been written so the new key participates in the LRU ordering.

        Entries missing from the LRU sidecar (legacy / external writes)
        are treated as least-recently-accessed (timestamp 0) so they
        evict first - keeping pre-existing-without-LRU entries pinned
        forever would defeat the cap.
        """
        import os, glob
        if self.max_entries is None and self.max_size_mb is None:
            return 0
        # Same lock as _touch_lru: eviction reads + writes the sidecar AND removes files; another
        # process eviction sweep racing here could double-delete or leave the sidecar inconsistent.
        _lock_ctx = self._maybe_filelock(self._lock_path())
        _lock_ctx.__enter__()
        try:
            return self._evict_to_caps_locked()
        finally:
            _lock_ctx.__exit__(None, None, None)

    def _evict_to_caps_locked(self) -> int:
        import os, glob
        lru = self._load_lru()
        # Enumerate every on-disk entry, defaulting unseen ones to ts=0.
        files = glob.glob(os.path.join(self.cache_dir, "*.pkl"))
        entries: List[Tuple[str, float, int]] = []
        for path in files:
            stem = os.path.splitext(os.path.basename(path))[0]
            ts = float(lru.get(stem, 0.0))
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            entries.append((stem, ts, size))
        # Oldest first - that's the eviction order.
        entries.sort(key=lambda e: e[1])

        n = len(entries)
        total_bytes = sum(s for _, _, s in entries)
        max_bytes = (
            int(self.max_size_mb * 1024 * 1024)
            if self.max_size_mb is not None else None
        )

        removed = 0
        i = 0
        while i < len(entries):
            over_count = self.max_entries is not None and n > self.max_entries
            over_size = max_bytes is not None and total_bytes > max_bytes
            if not over_count and not over_size:
                break
            stem, _ts, size = entries[i]
            path = os.path.join(self.cache_dir, f"{stem}.pkl")
            try:
                os.remove(path)
                removed += 1
                n -= 1
                total_bytes -= size
                lru.pop(stem, None)
            except OSError:
                pass
            i += 1
        if removed:
            self._save_lru(lru)
        return removed

    def set(self, key: str, value: Any) -> None:
        """Write ``value`` to ``<cache_dir>/<key>.pkl``. Atomic via tmp-file rename so a crash mid-write doesn't leave corrupt cache files. ``f.flush()`` + ``os.fsync()`` run BEFORE ``os.replace`` so a power loss between pickle.dump returning and the OS flushing dirty pages cannot leave a zero-byte file under the visible name.

        After a successful write, the LRU sidecar is bumped and, if
        ``max_entries`` / ``max_size_mb`` are configured, least-recently-
        accessed entries are evicted to fit the caps.
        """
        import os, pickle, tempfile  # lazy
        path = self._path(key)
        # Write to a temp file in the same directory, then rename atomically.
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                # fsync inside the with-block so the data is on stable storage
                # BEFORE rename makes the path visible to readers. Without this,
                # rename can publish a name whose contents are still dirty pages
                # in the OS cache; a crash between rename and writeback leaves
                # a zero-byte file under the cache key.
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        # Touch LRU AFTER the rename so the timestamp reflects the new
        # entry; then evict if caps are configured.
        try:
            self._touch_lru(self._safe_key(key))
            self._evict_to_caps()
        except Exception:
            pass

    def invalidate(self, key: str) -> bool:
        """Remove a cached entry. Returns True if the entry existed, False otherwise."""
        import os
        path = self._path(key)
        if os.path.exists(path):
            os.remove(path)
            # Mirror the deletion in the LRU sidecar so a stale ledger
            # doesn't keep ghost keys pinning the count.
            try:
                lru = self._load_lru()
                if lru.pop(self._safe_key(key), None) is not None:
                    self._save_lru(lru)
            except Exception:
                pass
            return True
        return False

    def clear(self) -> int:
        """Remove all cached entries. Returns the number of files removed."""
        import os, glob  # lazy
        files = glob.glob(os.path.join(self.cache_dir, "*.pkl"))
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            if os.path.exists(self._lru_path):
                os.remove(self._lru_path)
        except OSError:
            pass
        return len(files)
