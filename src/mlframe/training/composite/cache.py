"""Disk-backed discovery cache: content-hash signature + key composer + DiscoveryCache class. Used by R&D workflows that re-run discovery with the same data + varying config; cache hits skip the expensive MI permutation null + Wilcoxon + tiny-model rerank phases. Pure stdlib + numpy + pandas; no composite-internal deps."""


from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import logging
import os
import pickle
import re
import tempfile
import time
import warnings
from typing import Any, Dict, List, NewType, Optional, Sequence, Tuple

from mlframe.utils.safe_pickle import (
    PickleVerificationError,
    safe_load as _safe_pickle_load,
    verify_sidecar as _safe_pickle_verify_sidecar,
    write_sidecar as _safe_pickle_write_sidecar,
)

# Typed alias for discovery-cache config signatures (see ``compute_config_signature_v1`` below).
# Keeping it a ``NewType`` over ``str`` means callers that build their own legacy string
# signatures keep working (assignment-compatible at runtime), but new code that types its
# signatures gets a stricter check from mypy / pyright that an arbitrary blake2b digest cannot
# be passed where a config signature is expected.
ConfigSignatureV1 = NewType("ConfigSignatureV1", str)

logger = logging.getLogger(__name__)

# Audit D L-4 (2026-05-18): pre-compiled hex matcher replaces the
# ``all(c in "0123456789abcdefABCDEF" for c in key)`` generator-expression in ``_safe_key`` per
# the user's ``feedback_orjson_compile_regex`` rule (compile once, reuse). The regex anchors with
# ``\A`` / ``\Z`` to require the WHOLE key to be hex (otherwise ``re.match`` only anchors to the
# start and ``abc-def`` would slip through).
_HEX_KEY_RE = re.compile(r"\A[0-9a-fA-F]+\Z")

# One-shot guard so the "filelock missing -> LRU/eviction races" warning fires at most once per process instead of on every cache touch.
_FILELOCK_WARNED = False

import numpy as np
import pandas as pd

try:
    import numba as _numba  # type: ignore[import-untyped]
    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is a core dep but allow graceful skip
    _numba = None  # type: ignore[assignment]
    _HAS_NUMBA = False

# Threshold below which the numpy path wins (JIT call overhead + bytes-shuffling dominates the
# arithmetic). Matches the pattern in ``composite_unary_transforms._YJ_NUMBA_MIN_N`` -- chosen
# from ``_benchmarks/bench_arch_d.py``: at n=500k the numba kernel is ~187x faster, and the
# crossover where numpy ties is around n=50k. The threshold is generous: pure-numpy is the
# correct choice for unit-test-sized columns.
_COL_STATS_NUMBA_MIN_N: int = 50_000


if _HAS_NUMBA:
    @_numba.njit(cache=True, fastmath=False, parallel=False)
    def _col_stats_float_numba_kernel(arr):  # type: ignore[no-untyped-def]
        # Serial reduction: a `prange` parallel reduction over (min, max) needs atomics that
        # numba doesn't lend itself to without explicit blocking. The serial loop is still
        # >100x faster than the numpy fancy-index path at n=500k (bench: bench_arch_d) because
        # it fuses the isfinite-mask + min + max + count into a single pass instead of three
        # full-array materialisations.
        n = arr.shape[0]
        n_null = 0
        mn = np.inf
        mx = -np.inf
        for i in range(n):
            v = arr[i]
            # `v != v` catches NaN; the infinity checks mirror the ``~np.isfinite`` mask in the
            # numpy path so a column with `inf` sentinels lands on the all_null branch when every
            # value is non-finite (parity with the legacy digest).
            if v != v or v == np.inf or v == -np.inf:
                n_null += 1
            else:
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
        return mn, mx, n_null

try:
    import polars as pl  # type: ignore
    _HAS_POLARS = True
except Exception:  # pragma: no cover
    pl = None  # type: ignore
    _HAS_POLARS = False


def _is_polars_df(x: Any) -> bool:
    """ENS-P2-6: prefer explicit isinstance check over duck-typing."""
    return _HAS_POLARS and isinstance(x, pl.DataFrame)


# Discovery caching layer: key discovery results by a content hash of (data-sample, target-column, config-signature, random_state) so re-runs that only vary inner hyperparameters skip the minutes-long MI-null / Wilcoxon / tiny-rerank phases.
# Primitives: ``data_signature`` (blake2b over a deterministic sample + dtypes + a reorder-sensitive row fingerprint), ``DiscoveryCache(cache_dir)`` (disk key->pickle store: get/set/invalidate/clear/__contains__), ``make_discovery_cache_key`` (stable hex key). The layer does NOT auto-integrate with fit(); callers manage lookup/store at their orchestration level to keep the discovery class free of I/O.


_DISCOVERY_SIGNATURE_SAMPLE_N: int = 1000
# Single source of truth for discovery cache seed; both
# ``data_signature`` and ``make_discovery_cache_key`` reference it so a
# downstream override touches one constant, not two function defaults.
# Audit D L-9 (2026-05-18): a caller running multiple parallel discoveries with different RNGs
# but relying on the default ``random_state`` of ``data_signature`` will see signature collisions
# (same default seed → same sampled rows → same digest). This is a documented design choice: the
# default keeps the signature stable across re-runs of the same workflow. Parallel-RNG callers
# MUST pass an explicit ``random_state`` matching the discovery's RNG seed.
_DISCOVERY_DEFAULT_SEED: int = 42


_ROW_ORDER_PREFIX_ROWS: int = 256


def _row_order_fingerprint(df: Any, n_edge: int = 8) -> str:
    """Cheap fingerprint of a frame's row order, sensitive to INNER reorders.

    Folded into ``data_signature`` so a shuffled frame produces a different signature than the
    original. Pre-fix only the first/last ``n_edge`` rows were hashed -- an inner shuffle
    (``df.sample(frac=0.99)``) that did not touch the head or tail rows produced an identical
    fingerprint, silently replaying stale specs on R&D workflows.

    Post-fix (audit D P1-2, 2026-05-18): polars path uses ``hash_rows()`` which produces a row
    hash for every row in O(N) (vectorised C++); we slice the first ``_ROW_ORDER_PREFIX_ROWS``
    and hash those bytes, so an inner reorder that lands inside the prefix bursts the cache.
    The pandas path hashes the head ``n_edge`` rows' raw bytes (``to_numpy().tobytes()``) instead
    of the prior ``to_csv`` round-trip (the codebase migrated away from CSV-roundtrip hashing in
    fingerprint.py for the same reason -- it's the slowest legacy path).

    ``n_edge`` retained as parameter for the pandas branch only; polars now hashes a wider
    prefix unconditionally.

    Returns ``""`` on any access failure (degrades to the prior reorder-stable behaviour rather
    than crashing on exotic frame types).
    """
    try:
        if _is_polars_df(df):
            # ``hash_rows()`` produces one u64 per row (vectorised); slicing the prefix gives a
            # bounded-cost fingerprint that catches inner reorders inside the first N rows.
            n_take = min(df.height, _ROW_ORDER_PREFIX_ROWS)
            if n_take == 0:
                return ""
            # Slice FIRST, then hash: hash_rows() is row-local so slicing the
            # prefix before hashing is digest-identical to hashing the whole
            # frame and slicing after -- but O(n_take*C) instead of O(N*C) plus
            # an N-row u64 allocation. On the 100+ GB frames this module targets
            # the prior whole-frame scan silently undid the data_signature
            # gather optimisation (multi-second + multi-GB per cache lookup).
            row_hashes = df.slice(0, n_take).hash_rows().to_numpy()
            payload = np.ascontiguousarray(row_hashes).tobytes()
            return hashlib.blake2b(payload, digest_size=8).hexdigest()
        elif isinstance(df, pd.DataFrame):
            n_take = min(len(df), n_edge)
            if n_take == 0:
                return ""
            # ``df.to_numpy().tobytes()`` on a frame with object/string columns
            # coerces the whole block to an object ndarray whose bytes are
            # PyObject* ADDRESSES -> non-deterministic across processes (and
            # re-materialised strings within one), silently breaking the cache.
            # ``hash_pandas_object`` is the content-based, row-order-sensitive
            # pandas analogue of polars ``hash_rows`` (uint64 per row).
            from pandas.util import hash_pandas_object
            head_bytes = np.ascontiguousarray(
                hash_pandas_object(df.head(n_take), index=False).to_numpy()
            ).tobytes()
            n_tail = min(len(df), n_edge)
            tail_bytes = np.ascontiguousarray(
                hash_pandas_object(df.tail(n_tail), index=False).to_numpy()
            ).tobytes()
            payload = head_bytes + b"|" + tail_bytes
            return hashlib.blake2b(payload, digest_size=8).hexdigest()
        else:
            return ""
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
                # ``null=0`` instead of an ``nuniq`` full ``np.unique`` sort: numpy int/bool dtypes hold no NaN so the null count is structurally zero (mirrors the polars int digest; drops the per-int-column O(n log n) sort -- one-time on-disk cache invalidation).
                return (
                    f"intmin={int(np.min(arr))};"
                    f"intmax={int(np.max(arr))};"
                    f"null=0"
                ).encode("utf-8")
            except Exception:
                return b"int_opaque"
        if kind == "f":
            # Numba kernel wins from n=~50k upward (bench: bench_arch_d.bench_col_stats_numba
            # measured ~187x at n=500k, 200 cols). Below threshold the JIT call overhead +
            # bytes-shuffling between numpy and numba dominates the arithmetic; the boolean-mask
            # numpy path is faster.
            if _HAS_NUMBA and arr.size >= _COL_STATS_NUMBA_MIN_N and arr.dtype == np.float64:
                mn_v, mx_v, n_null = _col_stats_float_numba_kernel(arr)
                if not np.isfinite(mn_v):
                    return f"all_null:{int(n_null)}".encode("utf-8")
                return (
                    f"min={float(mn_v):.12g};"
                    f"max={float(mx_v):.12g};"
                    f"null={int(n_null)}"
                ).encode("utf-8")
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
    # CACHE-P0-1/2 (audit D 2026-05-18): the per-column ``to_numpy()`` of the WHOLE column was
    # the dominant cost of ``data_signature`` on multi-million-row frames -- 200 columns x 10M
    # rows = 2 full materialisations per signature call (one for stats, one for the sample
    # gather). Polars can compute min / max / null-count natively in a single lazy ``select``
    # over all needed columns, and the sample gather is ``col.gather(sample_idx)`` which is
    # O(sample_n) instead of O(N). Result on a 200-col 10M-row frame: ~100x speedup measured
    # in tests/training/_benchmarks/bench_data_signature.py.
    cols_to_hash = [target_col] + [c for c in feature_cols if c != target_col]
    if _is_polars_df(df):
        present = [c for c in cols_to_hash if c in df.columns]
        if present:
            # Single polars expression returning min/max/null for every column in one pass.
            # ``cast(Utf8)`` on min/max keeps the digest dtype-agnostic so int / float / string /
            # categorical all flow through the same byte path. The numeric branches inside
            # ``_col_stats`` produce more discriminating digests for floats (NaN-aware) and ints,
            # so we delegate to ``_col_stats`` for numeric columns by sampling-then-stats. For
            # non-numeric columns we fold min/max/null directly without ever materialising the
            # column.
            numeric_cols: List[str] = []
            non_numeric_cols: List[str] = []
            for c in present:
                dt = df.schema[c]
                if dt.is_numeric():
                    numeric_cols.append(c)
                else:
                    non_numeric_cols.append(c)
            # Non-numeric: one single ``select`` for min / max / null_count across all of them.
            if non_numeric_cols:
                _stats_row = df.select(
                    [pl.col(c).cast(pl.Utf8).min().alias(f"_mn_{c}") for c in non_numeric_cols]
                    + [pl.col(c).cast(pl.Utf8).max().alias(f"_mx_{c}") for c in non_numeric_cols]
                    + [pl.col(c).null_count().alias(f"_nc_{c}") for c in non_numeric_cols]
                ).row(0)
                n = len(non_numeric_cols)
                for i, c in enumerate(non_numeric_cols):
                    h.update(str(df.schema[c]).encode("utf-8"))
                    h.update(b"|stats=")
                    mn = _stats_row[i]
                    mx = _stats_row[i + n]
                    nc = _stats_row[i + 2 * n]
                    h.update(f"strmin={mn};strmax={mx};null={int(nc) if nc is not None else 0}".encode("utf-8"))
                    # Sample bytes via gather (O(sample_n), no full materialisation).
                    # Hash the string CONTENT, not the object array's pointer bytes:
                    # ``.to_numpy()`` on a Utf8 column yields an object array whose
                    # ``.tobytes()`` is PyObject* addresses -> non-deterministic across
                    # processes (and re-materialised strings within one) -> the cache
                    # NEVER hits on real string/datetime/categorical frames.
                    sampled = df.get_column(c).gather(sample_idx).cast(pl.Utf8).to_list()
                    h.update("\x00".join("\x01" if v is None else v for v in sampled).encode("utf-8"))
            # Numeric branch: compute min / max / null in one polars expression for ALL numeric
            # columns at once (one Arrow batch), then route the per-column stats through the
            # existing ``_col_stats`` byte format for digest stability. We rebuild a tiny
            # ``stats_arr`` from the polars-computed scalars rather than materialising the column.
            if numeric_cols:
                _agg_row = df.select(
                    [pl.col(c).min().alias(f"_mn_{c}") for c in numeric_cols]
                    + [pl.col(c).max().alias(f"_mx_{c}") for c in numeric_cols]
                    + [pl.col(c).null_count().alias(f"_nc_{c}") for c in numeric_cols]
                ).row(0)
                n = len(numeric_cols)
                for i, c in enumerate(numeric_cols):
                    h.update(str(df.schema[c]).encode("utf-8"))
                    h.update(b"|stats=")
                    mn = _agg_row[i]
                    mx = _agg_row[i + n]
                    nc = int(_agg_row[i + 2 * n] or 0)
                    # Match the legacy ``_col_stats`` byte format so a digest computed before
                    # this refactor (replayed against the same frame) is stable. Float branch:
                    # ``min=...;max=...;null=...``; int / bool branch: ``intmin=...;intmax=...;
                    # nuniq=...``. ``nuniq`` previously required a full column scan; we drop it
                    # in favour of ``null=`` for ints too -- this is a deliberate digest shape
                    # change (digest will differ from pre-fix for int columns), accepted because
                    # the pre-fix digest required the full-column materialisation we are
                    # eliminating. Downstream callers that need cache invalidation across this
                    # refactor must clear their on-disk caches once.
                    dt = df.schema[c]
                    kind = "f" if dt in (pl.Float32, pl.Float64) else ("i" if dt.is_integer() else "u")
                    if kind == "f":
                        if mn is None or mx is None:
                            h.update(f"all_null:{nc}".encode("utf-8"))
                        else:
                            h.update(
                                f"min={float(mn):.12g};max={float(mx):.12g};null={nc}".encode("utf-8")
                            )
                    else:
                        if mn is None or mx is None:
                            h.update(f"all_null:{nc}".encode("utf-8"))
                        else:
                            h.update(
                                f"intmin={int(mn)};intmax={int(mx)};null={nc}".encode("utf-8")
                            )
                    # Sample bytes via gather (O(sample_n) materialisation only).
                    sampled = df.get_column(c).gather(sample_idx).to_numpy()
                    h.update(np.ascontiguousarray(sampled).tobytes())
    elif isinstance(df, pd.DataFrame):
        # Pandas: no equivalent of polars lazy-frame multi-column aggregate without materialising
        # the column, so we still call ``df[c].to_numpy()`` once per column. We at least avoid the
        # second materialisation for the sample bytes by reusing ``full`` (the gather slice is a
        # view on ``full``). bench-attempt-rejected: a single ``df[cols_to_hash].to_numpy()`` batches the dispatch but coerces all columns to
        # the common dtype (object on mixed-dtype frames), which breaks the per-column ``str(dtype)`` folded into the signature.
        for c in cols_to_hash:
            if c not in df.columns:
                continue
            h.update(str(df[c].dtype).encode("utf-8"))
            full = df[c].to_numpy()
            h.update(b"|stats=")
            h.update(_col_stats(full))
            sampled = full[sample_idx]
            if full.dtype.kind in ("O", "U", "S"):
                # Object/str: tobytes() on an object array hashes PyObject*
                # ADDRESSES, which differ across processes (and re-materialised
                # strings within one) -> non-deterministic signature -> the
                # discovery cache never hits on real frames. Hash the content.
                h.update("\x00".join(map(str, sampled.tolist())).encode("utf-8"))
            else:
                h.update(np.ascontiguousarray(sampled).tobytes())
    else:
        raise TypeError(f"data_signature: unsupported df type {type(df).__name__}")
    return h.hexdigest()


def compute_config_signature_v1(
    config: Any,
    *,
    library_versions: Optional[Dict[str, str]] = None,
) -> ConfigSignatureV1:
    """Factory that builds a ``ConfigSignatureV1`` from a discovery config + library versions.

    Folds EVERY field of ``config`` (preferring ``model_dump`` / ``dict`` over ``__dict__`` over
    ``repr``) plus the supplied ``library_versions`` map (mlframe + sklearn + boostings + polars
    + numpy + scipy + pandas + python major.minor) into a single blake2b digest. The typed
    return value lets ``make_discovery_cache_key`` accept ``ConfigSignatureV1`` instead of a
    bare ``str``, so a caller cannot accidentally pass any other hex string in its place.

    Field folding contract: callers that add a new config field MUST ensure it appears in
    ``model_dump`` (pydantic) or ``__dict__`` (dataclass). Fields not exposed via one of these
    routes are silently dropped from the signature, which would let two semantically-different
    configs share a cache entry. The bench at ``_benchmarks/bench_arch_d.py`` includes a
    coverage smoke that flags missing fields.
    """
    payload: dict[str, Any] = {}
    try:
        if hasattr(config, "model_dump"):
            payload["config"] = config.model_dump(mode="json")
        elif hasattr(config, "dict"):
            payload["config"] = config.dict()
        else:
            payload["config"] = (
                {str(k): str(v) for k, v in sorted(getattr(config, "__dict__", {}).items())}
                or repr(config)
            )
    except Exception as _e:
        payload["config_repr"] = repr(config)
        payload["config_dump_error"] = str(_e)
    if library_versions is not None:
        payload["versions"] = dict(sorted(library_versions.items()))
    blob = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()
    return ConfigSignatureV1(digest)


def make_discovery_cache_key(
    df_sig: str,
    target_col: str,
    config_signature: ConfigSignatureV1,
    _legacy_random_state_sentinel: int = _DISCOVERY_DEFAULT_SEED,
    random_state: int | None = None,
) -> str:
    """Combine the parts of a discovery cache key into a stable hex string. The ``config_signature`` is a ``ConfigSignatureV1`` produced by :func:`compute_config_signature_v1`; a plain ``str`` still works at runtime (``NewType`` is structurally compatible) but type-checkers will warn callers that bypass the factory.

    ``_legacy_random_state_sentinel``: hash-only contributor for back-compat with the historical 4-arg form. The actual seed is already folded into ``df_sig`` (row-sample seeding) and ``config_signature`` (dataclass dump), so this kwarg should NEVER be used as a discriminator by callers. The name advertises this so a future reader cannot misread ``random_state=0`` as the seed in use. ``random_state`` is accepted as an alias for back-compat and routes to the same slot -- pass it explicitly to override the positional. Default ``None`` (NOT ``_DISCOVERY_DEFAULT_SEED``) so the conditional below only fires when the caller actually supplied a kwarg; otherwise the prior default-equal value silently overwrote any positional sentinel the caller passed.
    """
    if random_state is not None:
        _legacy_random_state_sentinel = random_state
    h = hashlib.blake2b(digest_size=16)
    h.update(df_sig.encode("utf-8"))
    h.update(b"|")
    h.update(target_col.encode("utf-8"))
    h.update(b"|")
    h.update(str(config_signature).encode("utf-8"))
    h.update(b"|")
    h.update(str(int(_legacy_random_state_sentinel)).encode("utf-8"))
    return h.hexdigest()


class DiscoveryCache:
    """Disk-backed key->value cache for CompositeTargetDiscovery results.

    Values are pickled with stdlib ``pickle`` (safe: stored objects are dataclass-derived dicts). Files live under ``<cache_dir>/<key>.pkl`` with one file per key for easy invalidation / cleanup.

    Concurrency: value writes are crash-safe via atomic ``os.replace`` (tmp-file + fsync + rename), the LRU sidecar and eviction sweep are guarded by a cross-process ``filelock`` (when ``filelock`` is installed), and ``invalidate`` is idempotent under concurrent callers. ``filelock`` is optional: without it the LRU/eviction read-modify-write can race between processes sharing ``cache_dir`` (a stale-snapshot save may overwrite a fresh access timestamp), though the value files themselves stay consistent.
    """

    def __init__(
        self,
        cache_dir: Any,
        *,
        max_entries: Optional[int] = 1000,
        max_size_mb: Optional[float] = 2000.0,
    ) -> None:
        """Construct a disk-backed discovery cache.

        Parameters
        ----------
        cache_dir
            Directory hosting one ``<key>.pkl`` per entry.
        max_entries
            Hard cap on the number of cached entries. When ``set()`` would
            push the count above the cap, the least-recently-accessed
            entries are evicted to fit. Default 1000 - protects against
            unbounded R&D growth. Pass ``None`` to disable count-based
            eviction explicitly.
        max_size_mb
            Soft cap on the total cache footprint in megabytes. Evaluated
            after the count cap. Default 2000 MB. Pass ``None`` to disable.

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
        from ..feature_handling.system import long_path_safe
        self.cache_dir = long_path_safe(os.path.abspath(str(cache_dir)))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.max_entries = max_entries
        self.max_size_mb = max_size_mb
        self._lru_path = os.path.join(self.cache_dir, ".lru")
        # Both caps None means the cache grows monotonically on repeated R&D runs and silently
        # fills the disk. CI / test runs commonly suppress warnings, so a WARN-only signal
        # disappears in practice -- promote to a hard ValueError so the operator is forced to make
        # an explicit choice. Pass ``max_entries=10**9`` or ``max_size_mb=float("inf")`` if the
        # genuine intent is "no eviction".
        if max_entries is None and max_size_mb is None:
            raise ValueError(
                f"DiscoveryCache at {self.cache_dir!r} constructed with max_entries=None and max_size_mb=None: "
                "the cache would grow without bound. Pass at least one explicit cap (or float('inf') / 10**9 "
                "to opt into unbounded growth) so the choice is auditable."
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
            global _FILELOCK_WARNED
            if not _FILELOCK_WARNED:
                _FILELOCK_WARNED = True
                logger.warning(
                    "DiscoveryCache: 'filelock' not installed; LRU/eviction read-modify-write is "
                    "unprotected across processes sharing the cache dir (value files stay consistent). "
                    "Install 'filelock' to close the race."
                )
            return contextlib.nullcontext()

    def _load_lru(self) -> Dict[str, float]:
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
        # Same atomic-rename + fsync discipline as the value writes - LRU
        # corruption would silently break eviction order.
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, prefix=".lru.", suffix=".tmp")
        # ``fd`` ownership tracking: BufferedWriter from os.fdopen adopts on success;
        # if os.fdopen itself raises (rare: MemoryError) we manually close fd to avoid
        # a per-failure fd leak. _save_lru runs on every cache touch so leaks compound
        # quickly under sustained load (Windows 8192 default fd ceiling).
        _fd_adopted = False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                _fd_adopted = True
                json.dump(lru, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._lru_path)
        except Exception:
            if not _fd_adopted:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _touch_lru(self, key: str) -> None:
        # Cross-process file lock around read-modify-write so a concurrent process can't replay a
        # stale-snapshot save and overwrite a fresh access timestamp. filelock is optional.
        #
        # Audit D L-1 (2026-05-18): ``time.time()`` is wall-clock (subject to NTP step-back) but
        # we deliberately do NOT use ``time.monotonic()`` here -- the LRU sidecar is shared across
        # processes, and monotonic clock values are not comparable across processes (each
        # process's monotonic clock starts from an arbitrary reference). Cross-process LRU
        # ordering requires a shared reference frame -- wall clock is the only portable option.
        # Single-host NTP step-backs are rare and only mis-order entries within the step delta.
        #
        # Audit D P2-2 (2026-05-18): the full JSON rewrite per touch is O(N entries). For caches
        # >10,000 entries the rewrite cost dominates the read; we keep the simple
        # write-everything-each-touch design because:
        #   (a) the rewrite happens under the cross-process filelock, so a "mark dirty, flush
        #       later" batching strategy would require additional cross-process flush
        #       synchronisation;
        #   (b) the atomic-rename + fsync guarantee survives a mid-touch crash;
        #   (c) typical R&D cache sizes are <500 entries where the rewrite is sub-millisecond.
        # Operators hitting the >10K-entry regime should pass ``max_entries`` to keep the file
        # bounded.
        with self._maybe_filelock(self._lock_path()):
            lru = self._load_lru()
            lru[key] = time.time()
            self._save_lru(lru)

    def _path(self, key: str) -> str:
        safe_key = self._safe_key(key)
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")

    def __contains__(self, key: str) -> bool:
        # Audit D L-5 (2026-05-18): ``os.path.exists`` is racy by design (TOCTOU: the file
        # can be deleted between the check and a subsequent ``get``). Callers should use
        # ``get(key, default=_SENTINEL)`` and check ``is _SENTINEL`` instead of the
        # ``key in cache`` + ``cache.get(key)`` pattern. ``get`` opens the file directly and
        # treats ``FileNotFoundError`` as a miss, so the race is closed there.
        return os.path.exists(self._path(key))

    def get(self, key: str, default: Any = None) -> Any:
        """Return the cached value, or ``default`` if the key is absent / unreadable.

        We deliberately omit any ``os.path.exists`` check before opening:
        on Windows a delete-between-exists-and-open race surfaced
        ``FileNotFoundError`` after the existence check passed. The
        implementation just try-opens and treats any failure (missing
        file, locked file, corrupt pickle) as a cache miss.

        A successful read updates the LRU sidecar so subsequent eviction
        picks the least-recently-USED (not just least-recently-WRITTEN)
        entry.
        """
        path = self._path(key)
        # Size-clamp: refuse to load a cache entry that exceeds the configured byte
        # ceiling (default 1 GiB; override via MLFRAME_DISCOVERY_CACHE_MAX_BYTES).
        # Discovery cache entries SHOULD be small (kB-MB scale: spec lists + scalar
        # metadata + per-base float32 arrays at screen-sample size). A multi-GB
        # entry indicates a bug upstream (e.g. an _auto_base_pool entry sized to
        # FULL train rows leaked into the pickled discovery instance) and loading
        # it would spike RAM at the worst possible time -- right before composite
        # discovery starts its own allocations. Treat oversize as a miss so the
        # caller falls back to a fresh recompute; the stale entry is left on disk
        # so the operator can inspect / delete it.
        try:
            _max_bytes_raw = os.environ.get("MLFRAME_DISCOVERY_CACHE_MAX_BYTES")
            _max_bytes = int(_max_bytes_raw) if _max_bytes_raw else 1024 * 1024 * 1024
        except (TypeError, ValueError):
            _max_bytes = 1024 * 1024 * 1024
        try:
            _file_size = os.path.getsize(path)
        except OSError:
            _file_size = -1
        if _file_size > _max_bytes:
            logger.warning(
                "DiscoveryCache: skipping oversized entry at %s (%.2f GiB > %.2f GiB ceiling); "
                "treating as cache miss. Set MLFRAME_DISCOVERY_CACHE_MAX_BYTES to raise the cap "
                "or delete the file to recompute. An oversized entry usually means an unintended "
                "full-train ndarray got pickled into the discovery instance upstream.",
                path,
                _file_size / 1024 ** 3,
                _max_bytes / 1024 ** 3,
            )
            return default
        # Route the load through safe_pickle so a corrupt-sidecar finding (digest mismatch from
        # tampering or partial write) raises PickleVerificationError instead of silently returning
        # stale data. allow_unverified=True keeps the migration story: legacy entries written before
        # the sidecar landed remain readable (cache miss is the safe fallback if they're broken).
        # Operators who want strict-only behaviour set MLFRAME_DISCOVERY_CACHE_STRICT=1.
        _strict = os.environ.get("MLFRAME_DISCOVERY_CACHE_STRICT", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        try:
            if _strict:
                value = _safe_pickle_load(path)
            else:
                # allow_unverified=True: missing sidecar is OK (WARN-logged once by verify_sidecar);
                # digest mismatch still raises so a tampered file does not slip into the cache hit
                # path. The broad except below converts the mismatch to a cache miss so callers see
                # consistent semantics.
                value = _safe_pickle_load(path, allow_unverified=True)
        except FileNotFoundError:
            return default
        except Exception as _e:
            # A persistent corrupt / unverifiable entry would otherwise return a silent miss every run, triggering unbounded multi-minute recomputes with no operator signal. Surface it once per read, then treat as a miss.
            logger.warning(
                "DiscoveryCache: unreadable/unverifiable entry %s (%s: %s); treating as miss",
                path, type(_e).__name__, _e,
            )
            return default
        # Successful read: bump LRU. Done outside the read try/except so
        # an LRU file failure doesn't break the read path.
        try:
            self._touch_lru(self._safe_key(key))
        except Exception:
            pass
        return value

    def _safe_key(self, key: str) -> str:
        """Sanitised key (matches the on-disk filename stem).

        Collision-proof: pure-hex keys (the format ``make_discovery_cache_key`` emits)
        pass through unchanged. Any other key is hashed via blake2b and tagged with
        ``__h`` plus the BYTE length of the original; this prevents the old
        "strip non-alnum" sanitiser collapsing ``abc-def`` and ``abcdef`` (or
        ``abc/../def`` and ``abcdef``) to the same filename.
        """
        if not key:
            raise ValueError(f"DiscoveryCache: empty key {key!r}")
        # Pure-hex (the format make_discovery_cache_key emits) passes through unchanged.
        if _HEX_KEY_RE.match(key):
            return key.lower()
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
        return f"{digest}__h{len(key.encode('utf-8'))}"

    def _entry_size_bytes(self, safe_key: str) -> int:
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
        if self.max_entries is None and self.max_size_mb is None:
            return 0
        # Same lock as _touch_lru: eviction reads + writes the sidecar AND removes files; another
        # process eviction sweep racing here could double-delete or leave the sidecar inconsistent.
        # Wave 52 (2026-05-20): capture sys.exc_info() so the in-flight exception is
        # forwarded to the lock manager's __exit__ (preserves CM contract), AND wrap
        # __exit__ itself in try/except so a filelock cleanup error doesn't mask the
        # eviction-body exception.
        import sys as _sys
        _lock_ctx = self._maybe_filelock(self._lock_path())
        _lock_ctx.__enter__()
        try:
            return self._evict_to_caps_locked()
        finally:
            _exc = _sys.exc_info()
            try:
                _lock_ctx.__exit__(*_exc)
            except Exception as _exit_err:
                logger.warning("DiscoveryCache eviction filelock __exit__ failed: %s", _exit_err)

    def _evict_to_caps_locked(self) -> int:
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
                # Drop the sidecar only after the value file is gone: if the value remove raised (e.g. Windows lock) the entry survives and must keep its sidecar, else strict load refuses the still-present entry forever.
                try:
                    os.remove(path + ".sha256")
                except OSError:
                    pass
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
        path = self._path(key)
        # Write to a temp file in the same directory, then rename atomically.
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        # ``fd`` ownership tracking (see io.py:atomic_write_bytes for the canonical pattern).
        # On the rare path where os.fdopen itself raises BEFORE the BufferedWriter adopts fd,
        # the raw fd would otherwise leak. set() runs on every cache write so leaks compound
        # quickly. Track adoption + manually close on failure.
        _fd_adopted = False
        try:
            with os.fdopen(fd, "wb") as f:
                _fd_adopted = True
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                # fsync inside the with-block so the data is on stable storage
                # BEFORE rename makes the path visible to readers. Without this,
                # rename can publish a name whose contents are still dirty pages
                # in the OS cache; a crash between rename and writeback leaves
                # a zero-byte file under the cache key.
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            # Write the sha256 sidecar AFTER the rename so the digest matches the on-disk bytes.
            # Sidecar write failures are logged at DEBUG (the value file is already durable); a
            # subsequent strict load will refuse the entry until the sidecar is regenerated, which
            # is the correct fail-closed behaviour.
            try:
                _safe_pickle_write_sidecar(path)
            except OSError as _sc_err:
                logger.debug("DiscoveryCache.set sidecar write failed (value written OK): %s", _sc_err)
            # Audit D L-10 (2026-05-18): POSIX requires ``fsync(dirfd)`` to make the new entry's
            # directory metadata durable across a power loss; without it the rename is visible
            # to readers but may revert after a crash on journaled-data-mode-off filesystems.
            # Windows NTFS does NOT expose directory fsync via ``os.fsync(dirfd)``
            # (``OSError: [Errno 13] Permission denied`` opening a dir for fsync), so we skip
            # the dir fsync on Windows and rely on the journaled-metadata guarantee NTFS already
            # provides for renames. On POSIX we attempt the dir fsync but treat failure as
            # non-fatal: the file fsync already happened, only metadata durability is at risk.
            if os.name == "posix":
                try:
                    _dir_fd = os.open(self.cache_dir, os.O_RDONLY)
                    try:
                        os.fsync(_dir_fd)
                    finally:
                        os.close(_dir_fd)
                except OSError:
                    pass
        except Exception:
            if not _fd_adopted:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        # Touch LRU AFTER the rename so the timestamp reflects the new
        # entry; then evict if caps are configured.
        #
        # Audit D L-6 (2026-05-18): the previous ``except Exception: pass`` silently swallowed
        # disk-full / lock-timeout / corrupt-LRU errors during eviction. Log at DEBUG so an
        # operator running with ``logging.DEBUG`` sees the underlying cause, while normal runs
        # are unaffected (the write itself already succeeded before this block).
        try:
            self._touch_lru(self._safe_key(key))
            self._evict_to_caps()
        except Exception as _evict_err:
            logger.debug("DiscoveryCache.set LRU/eviction failed (entry written OK): %s", _evict_err)

    def invalidate(self, key: str) -> bool:
        """Remove a cached entry. Returns True if the entry existed, False otherwise."""
        path = self._path(key)
        # Wave 48 (2026-05-20): TOCTOU race -- parallel hyperopt suites sharing
        # cache_dir can both call invalidate(same_key); the prior exists+remove
        # pattern raised uncaught FileNotFoundError on the loser. Replace with
        # try/except so concurrent invalidations of the same key are idempotent.
        try:
            os.remove(path)
        except FileNotFoundError:
            return False
        # Also drop the sha256 sidecar so a future write of the same key starts fresh.
        try:
            os.remove(path + ".sha256")
        except OSError:
            pass
        # Mirror the deletion in the LRU sidecar so a stale ledger
        # doesn't keep ghost keys pinning the count.
        try:
            lru = self._load_lru()
            if lru.pop(self._safe_key(key), None) is not None:
                self._save_lru(lru)
        except Exception:
            pass
        return True

    def clear(self) -> int:
        """Remove all cached entries. Returns the number of files removed."""
        files = glob.glob(os.path.join(self.cache_dir, "*.pkl"))
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
            try:
                os.remove(f + ".sha256")
            except OSError:
                pass
        try:
            if os.path.exists(self._lru_path):
                os.remove(self._lru_path)
        except OSError:
            pass
        return len(files)


def _discovery_cache_bytes_total(cache: DiscoveryCache) -> int:
    """Best-effort on-disk byte total for a :class:`DiscoveryCache`. Mirrors ``_mrmr_cache_bytes_total`` / ``SuiteArtefactCache._total_bytes_locked`` so callers comparing against ``max_size_mb`` don't inline a per-call directory walk. Counts the .pkl plus its .pkl.sha256 sidecar -- both contribute to the on-disk budget."""
    total = 0
    try:
        with os.scandir(cache.cache_dir) as it:
            for de in it:
                if not de.is_file():
                    continue
                if not (de.name.endswith(".pkl") or de.name.endswith(".pkl.sha256")):
                    continue
                try:
                    total += de.stat().st_size
                except OSError:
                    pass
    except FileNotFoundError:
        return 0
    return total
