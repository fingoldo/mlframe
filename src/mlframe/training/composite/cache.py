"""Disk-backed discovery cache: content-hash signature + key composer + DiscoveryCache class. Used by R&D workflows that re-run discovery with the same data + varying config; cache hits skip the expensive MI permutation null + Wilcoxon + tiny-model rerank phases. Pure stdlib + numpy + pandas; no composite-internal deps."""


from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, NewType, Optional, Sequence

# The disk-backed ``DiscoveryCache`` store (which owns the pickle / filelock / tempfile / glob
# machinery and the ``mlframe.utils.safe_pickle`` imports) lives in the sibling
# ``cache_store.py`` and is re-exported at the bottom of this module.

# Typed alias for discovery-cache config signatures (see ``compute_config_signature_v1`` below).
# Keeping it a ``NewType`` over ``str`` means callers that build their own legacy string
# signatures keep working (assignment-compatible at runtime), but new code that types its
# signatures gets a stricter check from mypy / pyright that an arbitrary blake2b digest cannot
# be passed where a config signature is expected.
ConfigSignatureV1 = NewType("ConfigSignatureV1", str)

logger = logging.getLogger(__name__)

# On-disk discovery-cache schema/code-version stamp, folded UNCONDITIONALLY into every
# ``compute_config_signature_v1`` digest (see that function). The code-version component must be
# present even when a caller does NOT pass ``library_versions`` (the production path in
# ``_phase_composite_discovery`` does, but a direct R&D user following the module recipe at the
# top of this file does not) -- otherwise a docstring-following caller builds keys with NO
# mlframe-version component and replays stale specs across mlframe upgrades that change MI /
# Wilcoxon / boosting / transform semantics.
#
# ``_DISCOVERY_CACHE_SCHEMA_VERSION`` is the on-disk payload/semantics epoch: BUMP IT whenever
# the cached payload shape changes OR discovery semantics change in a way that must invalidate
# every cached spec regardless of the library tuple (e.g. a digest-format change in
# ``data_signature``, a new pickled field, a transform-registry semantic change). It is folded
# in addition to ``mlframe.__version__`` so even a hot-patched same-version checkout that
# changes the payload can force invalidation without a version bump.
#
# Folding either of these is a deliberate one-time on-disk digest change: existing entries miss
# and recompute once (the correct fail-safe -- a stale spec replayed under new code is the bug
# this closes). See ``compute_config_signature_v1`` docstring for the bump discipline.
_DISCOVERY_CACHE_SCHEMA_VERSION: int = 1

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
    """Prefer an explicit isinstance check over duck-typing."""
    return _HAS_POLARS and isinstance(x, pl.DataFrame)


# Discovery caching layer: key discovery results by a content hash of (data-sample, target-column, config-signature, random_state) so re-runs that only vary inner hyperparameters skip the minutes-long MI-null / Wilcoxon / tiny-rerank phases.
# Primitives: ``data_signature`` (blake2b over a deterministic sample + dtypes + a reorder-sensitive row fingerprint), ``DiscoveryCache(cache_dir)`` (disk key->pickle store: get/set/invalidate/clear/__contains__), ``make_discovery_cache_key`` (stable hex key), ``compute_config_signature_v1`` (config -> ConfigSignatureV1). The layer does NOT auto-integrate with fit(); callers manage lookup/store at their orchestration level to keep the discovery class free of I/O.
# Code-version safety: ``compute_config_signature_v1`` UNCONDITIONALLY folds the on-disk schema epoch (``_DISCOVERY_CACHE_SCHEMA_VERSION``) and ``mlframe.__version__`` into every config signature, so cache keys built by a direct caller who passes no ``library_versions`` still invalidate across mlframe upgrades. Pass ``library_versions`` (as the suite does) for the richer sklearn/boosting/polars/numpy/scipy/pandas/python tuple on top.


_DISCOVERY_SIGNATURE_SAMPLE_N: int = 1000
# Single source of truth for discovery cache seed; both
# ``data_signature`` and ``make_discovery_cache_key`` reference it so a
# downstream override touches one constant, not two function defaults.
# A caller running multiple parallel discoveries with different RNGs but relying on the default
# ``random_state`` of ``data_signature`` will see signature collisions (same default seed → same
# sampled rows → same digest). This is a deliberate design choice: the default keeps the signature
# stable across re-runs of the same workflow. Parallel-RNG callers MUST pass an explicit
# ``random_state`` matching the discovery's RNG seed.
_DISCOVERY_DEFAULT_SEED: int = 42


_ROW_ORDER_PREFIX_ROWS: int = 256


def _row_order_fingerprint(df: Any, n_edge: int = 8) -> str:
    """Cheap fingerprint of a frame's row order, sensitive to INNER reorders.

    Folded into ``data_signature`` so a shuffled frame produces a different signature than the
    original. Pre-fix only the first/last ``n_edge`` rows were hashed -- an inner shuffle
    (``df.sample(frac=0.99)``) that did not touch the head or tail rows produced an identical
    fingerprint, silently replaying stale specs on R&D workflows.

    The polars path uses ``hash_rows()`` which produces a row hash for every row in O(N)
    (vectorised C++); we slice the first ``_ROW_ORDER_PREFIX_ROWS`` and hash those bytes, so an
    inner reorder that lands inside the prefix bursts the cache. The pandas path hashes the head
    ``n_edge`` rows' raw bytes (``to_numpy().tobytes()``) rather than a ``to_csv`` round-trip
    (CSV-roundtrip hashing is the slowest path).

    The polars branch also hashes a bounded TAIL slice in addition to the prefix, mirroring the
    pandas head+tail coverage: a prefix-only fingerprint misses a reorder that touches only the
    tail rows (e.g. ``df`` vs ``df`` with the last block shuffled), producing an identical
    fingerprint -- a blind spot relative to the pandas path. Both slices are bounded
    (``_ROW_ORDER_PREFIX_ROWS`` head + tail) so the O(N)-scan hazard does not return.

    KNOWN RESIDUAL BLIND SPOT: this is an edge-sampling fingerprint, NOT a whole-frame hash. A
    reorder confined to the MIDDLE of a frame larger than head+tail rows (and disjoint from the
    ``data_signature`` seeded sample) is not detected and replays the prior spec. Widening the
    edge windows trades cost for coverage; the bounded edges are the deliberate cost/coverage
    point for the 100+ GB frames this module targets. Callers needing strict mid-frame
    sensitivity should bump ``random_state`` (re-seeds the ``data_signature`` sample) or raise
    ``_ROW_ORDER_PREFIX_ROWS``.

    ``n_edge`` retained as parameter for the pandas branch only; polars now hashes a bounded
    head + tail unconditionally.

    Returns ``""`` on any access failure (degrades to the prior reorder-stable behaviour rather
    than crashing on exotic frame types).
    """
    try:
        if _is_polars_df(df):
            # ``hash_rows()`` produces one u64 per row (vectorised); slicing head + tail gives a
            # bounded-cost fingerprint that catches inner reorders inside either edge window.
            height = df.height
            n_take = min(height, _ROW_ORDER_PREFIX_ROWS)
            if n_take == 0:
                return ""
            # Slice FIRST, then hash: hash_rows() is row-local so slicing an
            # edge before hashing is digest-identical to hashing the whole
            # frame and slicing after -- but O(n_take*C) instead of O(N*C) plus
            # an N-row u64 allocation. On the 100+ GB frames this module targets
            # the prior whole-frame scan silently undid the data_signature
            # gather optimisation (multi-second + multi-GB per cache lookup).
            head_hashes = df.slice(0, n_take).hash_rows().to_numpy()
            payload = np.ascontiguousarray(head_hashes).tobytes()
            # Tail slice: only add a distinct tail when the frame is wider than the prefix,
            # else head already covers every row and a tail slice would re-hash the same rows
            # (the head-only digest stays unchanged for small frames -- bounded duplication).
            if height > n_take:
                n_tail = min(height - n_take, _ROW_ORDER_PREFIX_ROWS)
                tail_hashes = df.slice(height - n_tail, n_tail).hash_rows().to_numpy()
                payload += b"|" + np.ascontiguousarray(tail_hashes).tobytes()
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

    Row-order sensitivity: the signature is SENSITIVE TO ROW ORDER.
    ``_row_order_fingerprint`` hashes the head and tail of the frame, so a
    shuffled frame produces a different signature than the original (a
    reorder-stable signature would hand shuffled frames cache hits on stale
    specs). This also means the signature DOES change when row insertion
    shifts the sample composition or perturbs head/tail rows, which is the
    intended behaviour for the R&D workflow where the underlying frame is the
    same across runs.

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
    # Row count goes into the hash so appending rows invalidates the cache
    # even when the deterministic sample happens to coincide.
    h.update(b"nrows=")
    h.update(str(int(n_rows)).encode("utf-8"))
    # The seeded sample misses row swaps in unsampled positions, and the per-column min/max/null
    # stats are permutation-invariant. Fold in a cheap fingerprint of the first/last row content
    # so head/tail swaps burst the cache (re-running with reordered rows must NOT replay the prior
    # spec).
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

        Dtype-aware: falling through to ``np.unique(arr.astype(str))`` for anything that does not
        coerce cleanly to float64 collapses integer columns with NaN sentinels onto a
        stringified-distinct-values summary and drops the min/max/null distribution information.
        The integer branch is handled explicitly instead: when the dtype kind is in {'i','u','b'}
        we read min/max/uniques without ever trying the float-cast that NaN sentinels would
        corrupt.
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
    # The per-column ``to_numpy()`` of the WHOLE column is the dominant cost of ``data_signature``
    # on multi-million-row frames -- 200 columns x 10M rows = 2 full materialisations per signature
    # call (one for stats, one for the sample gather). Polars can compute min / max / null-count
    # natively in a single lazy ``select``
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

    Always-folded code-version stamp: regardless of whether ``library_versions`` is passed, the
    digest UNCONDITIONALLY folds ``_DISCOVERY_CACHE_SCHEMA_VERSION`` and a best-effort
    ``mlframe.__version__`` under the reserved ``_schema`` key. An opt-in version component (only
    present when a caller supplied ``library_versions``) lets a direct R&D user following the
    module recipe at the top of this file build keys with no code-version component and replay
    stale specs across mlframe upgrades. ``library_versions`` remains the richer override -- the production
    path (``_phase_composite_discovery._discovery_config_signature``) still supplies the full
    sklearn/boosting/polars/numpy/scipy/pandas/python tuple; the ``_schema`` fold is additive,
    so it does NOT replace that richer map, it just guarantees a floor of version sensitivity
    for callers who pass nothing.

    Bump discipline: increment ``_DISCOVERY_CACHE_SCHEMA_VERSION`` whenever the cached payload
    shape changes or discovery semantics change in a way that must invalidate every cached spec
    regardless of the library tuple (digest-format change in ``data_signature``, a new pickled
    field, a transform-registry semantic change). Folding the schema version is a deliberate
    one-time on-disk digest change -- existing entries miss and recompute once.

    Field folding contract: callers that add a new config field MUST ensure it appears in
    ``model_dump`` (pydantic) or ``__dict__`` (dataclass). Fields not exposed via one of these
    routes are silently dropped from the signature, which would let two semantically-different
    configs share a cache entry. The bench at ``_benchmarks/bench_arch_d.py`` includes a
    coverage smoke that flags missing fields.
    """
    payload: dict[str, Any] = {}
    # Unconditional code-version / schema epoch stamp. Best-effort mlframe version: if the
    # import fails (partial install / circular import during teardown) we still fold the schema
    # version so the digest is never version-blind. Kept under a reserved ``_schema`` key so it
    # never collides with a config field or the ``versions`` override map below.
    try:
        from mlframe import __version__ as _mlframe_version
    except Exception:
        _mlframe_version = "?"
    payload["_schema"] = {
        "discovery_cache_schema_version": _DISCOVERY_CACHE_SCHEMA_VERSION,
        "mlframe": str(_mlframe_version),
    }
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


# ----------------------------------------------------------------------------------------------
# Prebinned-feature-matrix cache (in-process, content-hash-keyed).
#
# ``_prebin_feature_columns`` (screening.py) turns the SMALL screen-sized float feature matrix
# into an (n_sample, n_feat) int16/int32 code matrix via per-column ``np.quantile`` +
# ``np.searchsorted`` -- O(n*F*log n). The codes are a DETERMINISTIC function of the binned bytes
# (the exact sampled rows) and ``nbins`` only: NOTHING else in the discovery config touches them.
# So a second discovery on the SAME data + sample + nbins but a DIFFERENT config (different
# ``mi_estimator`` re-enabling bin, different transform set, different rerank knobs) recomputes
# the identical codes. This cache stores those codes keyed by a content hash of the float matrix
# bytes + nbins so the second run skips the quantile binning and reuses bit-identical codes.
#
# 100GB-frame rule: the matrix that gets binned is ALREADY the screen sample (``mi_sample_n``
# rows, typ. 20-100k), never the raw 100GB frame -- discovery builds it once via
# ``_build_feature_matrix(df, features, train_idx_screen)``. The cached value is the int16/int32
# code matrix (HALF/QUARTER the float bytes). We STILL gate on byte size: a code matrix above
# ``_PREBIN_CACHE_MAX_BYTES`` (default 512 MiB, override via env) is NOT stored, so a pathological
# huge screen sample never pins hundreds of MB in the per-process cache. The signature itself is
# computed over the float matrix bytes (cheap blake2b over a contiguous buffer), never a frame copy.
# ----------------------------------------------------------------------------------------------

_PREBIN_CACHE_MAX_BYTES_DEFAULT: int = 512 * 1024 * 1024  # 512 MiB per code-matrix ceiling.
_PREBIN_CACHE_MAX_ENTRIES_DEFAULT: int = 32  # FIFO/LRU cap on stored code matrices.


def _prebin_cache_max_bytes() -> int:
    """Per-entry byte ceiling for the prebin code cache (env-overridable)."""
    raw = os.environ.get("MLFRAME_PREBIN_CACHE_MAX_BYTES")
    try:
        return int(raw) if raw else _PREBIN_CACHE_MAX_BYTES_DEFAULT
    except (TypeError, ValueError):
        return _PREBIN_CACHE_MAX_BYTES_DEFAULT


def prebin_matrix_signature(feature_matrix: np.ndarray, nbins: int) -> str:
    """Content-hash key for the (feature_matrix, nbins) -> bin-codes mapping.

    The bin codes are a deterministic function of the matrix VALUES + dtype + shape + ``nbins``
    (per-column ``np.quantile`` cut edges then ``searchsorted``). Folding all four into a blake2b
    digest gives a key that hits iff a later call would recompute byte-identical codes, and misses
    on any change (a different sample, a re-ordered/rescaled column, a different nbins). The hash
    is over the contiguous matrix buffer -- O(matrix bytes), no copy of the source frame.
    """
    arr = np.ascontiguousarray(feature_matrix)
    h = hashlib.blake2b(digest_size=16)
    h.update(b"prebin_v1|nbins=")
    h.update(str(int(nbins)).encode("utf-8"))
    h.update(b"|dtype=")
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(b"|shape=")
    h.update(str(arr.shape).encode("utf-8"))
    h.update(b"|data=")
    h.update(arr.tobytes())
    return h.hexdigest()


class PrebinCache:
    """In-process LRU of (signature -> int16/int32 bin-code matrix).

    Keeps the SMALL deterministic prebin codes alive across discovery runs in one process so a
    re-discovery with a different config (same data/sample/nbins) skips the quantile binning. The
    stored array is the caller's code matrix held by reference (no copy -- the caller already owns
    it and treats prebin codes as read-only downstream). Size-gated per entry (``max_bytes``) and
    count-capped (``max_entries``) so a pathological screen sample never pins hundreds of MB.

    Thread-safe: a single ``RLock`` guards the OrderedDict (discovery's per-base loop can dispatch
    in parallel and several workers may probe the cache concurrently).
    """

    def __init__(
        self,
        *,
        max_entries: int = _PREBIN_CACHE_MAX_ENTRIES_DEFAULT,
        max_bytes: Optional[int] = None,
    ) -> None:
        import collections
        import threading
        self._store: "collections.OrderedDict[str, np.ndarray]" = collections.OrderedDict()
        self._lock = threading.RLock()
        self.max_entries = int(max_entries)
        self._max_bytes_override = max_bytes
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.skipped_oversize = 0

    @property
    def max_bytes(self) -> int:
        return self._max_bytes_override if self._max_bytes_override is not None else _prebin_cache_max_bytes()

    def get(self, signature: str) -> Optional[np.ndarray]:
        with self._lock:
            val = self._store.get(signature)
            if val is None:
                self.misses += 1
                return None
            self._store.move_to_end(signature)  # LRU: mark most-recently-used.
            self.hits += 1
            return val

    def put(self, signature: str, codes: np.ndarray) -> bool:
        """Store ``codes`` under ``signature``. Returns False (and stores nothing) if the array
        exceeds the per-entry byte ceiling (100GB-frame guard). Never copies ``codes``."""
        nbytes = int(getattr(codes, "nbytes", 0))
        if nbytes > self.max_bytes:
            with self._lock:
                self.skipped_oversize += 1
            logger.debug(
                "PrebinCache: not caching %d-byte code matrix (> %d ceiling)", nbytes, self.max_bytes,
            )
            return False
        with self._lock:
            self._store[signature] = codes
            self._store.move_to_end(signature)
            self.stores += 1
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)  # evict least-recently-used.
        return True

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Module-level singleton: discovery runs within one process share the prebin codes. A process
# boundary (the 100GB-frame "never pickle a frame" rule) is the natural cache boundary -- we never
# persist these codes to disk (they are cheap to rebuild relative to the discovery they feed, and
# disk-persisting a per-run code matrix risks the oversize-pickle hazard DiscoveryCache guards).
_PREBIN_CACHE_SINGLETON: Optional[PrebinCache] = None


def get_prebin_cache() -> PrebinCache:
    """Return the shared per-process :class:`PrebinCache` (lazily constructed)."""
    global _PREBIN_CACHE_SINGLETON
    if _PREBIN_CACHE_SINGLETON is None:
        _PREBIN_CACHE_SINGLETON = PrebinCache()
    return _PREBIN_CACHE_SINGLETON


# The disk-backed ``DiscoveryCache`` store + its byte-total helper live in the sibling
# ``cache_store.py``. Re-exported here so existing imports
# (``from mlframe.training.composite.cache import DiscoveryCache``) keep resolving.
from .cache_store import (  # noqa: E402,F401
    DiscoveryCache,
    _discovery_cache_bytes_total,
)
