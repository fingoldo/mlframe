"""``MRMR`` top-level helpers: histogram, fingerprint/hash, replay, chunker.

Split out of ``mrmr.py`` to keep the parent below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; the parent re-exports every
moved symbol so historical
``from mlframe.feature_selection.filters.mrmr import histogram`` (and the
other moved names) imports continue to resolve.

What lives here:
  - ``histogram`` (astropy histogram with np.histogram fallback)
  - ``_canonicalise_dtype_str``
  - ``_mrmr_compute_y_fingerprint_sample`` / ``_mrmr_compute_x_fingerprint``
  - ``_hashable_params_signature`` / ``_content_array_signature``
  - ``_target_to_numpy_values`` / ``_target_name_signature``
  - ``_full_y_content_hash`` / ``_replay_fitted_state``
  - ``_lazy_chunks``

``_replay_fitted_state`` takes ``target: MRMR, source: MRMR`` only as a
string annotation (``from __future__ import annotations`` is active in
the parent and inherited here), so it does not need the class at import
time -- only at runtime.
"""
from __future__ import annotations

import hashlib
import logging
from itertools import islice
from typing import TYPE_CHECKING

import numpy as np

# Not used directly in this module -- load-bearing for import ORDER. This module is reached via
# filters/mrmr/__init__.py -> _legacy.py (circular: _legacy does ``from .mrmr import MRMR`` back into
# the package being loaded). Forcing wrappers.RFECV to resolve HERE, before that circular re-entry,
# keeps a working import order; removing it (2026-07-05 ruff "unused import" sweep) let a different
# external entry point (importing RFECV directly) trigger the circular mrmr<->_legacy re-entry first
# instead, which crashed the interpreter natively deep in the C-extension init chain. Restored with a
# suppression comment rather than re-deleted -- see CLAUDE.md's project-wide-rewrite-without-review incident.
from mlframe.feature_selection.wrappers import RFECV  # noqa: F401

# astropy is imported lazily on first histogram() call: the top-level
# ``import astropy`` costs ~0.6s and this fingerprints module is on the
# eager MRMR import path, yet most fits never call histogram().
_astropy_histogram = None
_astropy_resolved = False


def _resolve_astropy_histogram():
    """Lazily imports and memoises astropy's ``histogram``, caching a permanent ``None`` on failure so a
    missing/broken astropy install is only probed once per process instead of retried on every call."""
    global _astropy_histogram, _astropy_resolved
    if not _astropy_resolved:
        try:
            from astropy.stats import histogram as _h
            _astropy_histogram = _h
        except (ImportError, AttributeError):
            _astropy_histogram = None
        _astropy_resolved = True
    return _astropy_histogram


if TYPE_CHECKING:
    from .mrmr import MRMR


def histogram(a, bins="auto", **kwargs):
    """Astropy histogram with np.histogram fallback. See
    ``mlframe.feature_engineering.numerical.histogram`` for the rationale.
    """
    _h = _resolve_astropy_histogram()
    if _h is not None:
        return _h(a, bins=bins, **kwargs)
    return np.histogram(a, bins=bins, **kwargs)


logger = logging.getLogger(__name__)


# Cross-target identity cache for MRMR.fit. A prod log showed
# MRMR running 88 min on the SAME X for two composite targets
# (raw y + a monotonic-residual variant) -- both calls returned identity
# (no features dropped, no engineered features added). The second
# 88 min was pure loss because the result was already determined by
# the first call. With ``mrmr_skip_when_prior_was_identity=True``,
# subsequent fits on the same X-fingerprint short-circuit to identity
# output without running screen / FE / Hermite.
#
# X-fingerprint: blake2b hash of (sorted col names, n_rows, canonical
# dtypes + optional sample-content). Cheap to compute. Process-level
# cache, not persisted across processes. Reads / writes are guarded by
# ``_MRMR_IDENTITY_FP_LOCK`` so concurrent fits (multi-target parallel
# discovery in future / external joblib threading) do not race on the
# dict mutation.
import threading as _threading

_MRMR_IDENTITY_FP_CACHE: dict[str, bool] = {}
_MRMR_IDENTITY_FP_LOCK = _threading.Lock()

# ---------------------------------------------------------------------------------------------------------------
# Layer 3 pre-batch: threshold for the dispatch_batch_pair_mi_chunked pre-fill path in _run_fe_step.
#
# * _MRMR_BATCH_PRECOMPUTE_MIN_PAIRS: smallest pair count where the dispatcher overhead amortises. Below this
#   the per-pair joblib path is competitive (and avoids a redundant numba.cuda first-call compile when no GPU
#   speedup would materialise).
#
# There is deliberately NO pool-size cap here. Until 2026-07-09 a flat ``_MRMR_BATCH_PRECOMPUTE_MAX_K=200``
# ceiling forced any wider pool onto a ~35s/pair legacy joblib fallback -- a realistic several-hundred-column
# production pool (well within normal use) fell off a catastrophic-runtime cliff. The batch path now enumerates
# pairs via ``dispatch_batch_pair_mi_chunked`` (see ``batch_pair_mi_gpu.py``), which processes the C(k,2) pair
# space in RAM-bounded row-block chunks instead of materialising it all at once, so it is always the path taken
# for pool sizes up to whatever ``sis_screen_threshold`` (Gate A, ``_mrmr_sis_screen.py``) lets through -- at true
# extreme width (10^5+ raw columns) an exhaustive C(k,2) sweep is fundamentally intractable regardless of
# implementation, which is exactly what that separate front-gate exists to bound.
_MRMR_BATCH_PRECOMPUTE_MIN_PAIRS = 8


def _canonicalise_dtype_str(dt) -> str:
    """Polars / pandas-agnostic dtype canonical form.

    Mirrors the table in
    ``mlframe.training.core._phase_train_one_target._canonicalise_dtype``;
    duplicated here to avoid the import-time cycle (mrmr -> training).
    Same on-disk dtype yields the same canonical form across polars / pandas; identity-cache hits work irrespective of which backend the call site uses.
    """
    s = str(dt).strip().lower()
    if s.startswith("int"):
        return "i" + s[len("int") :]
    if s.startswith("uint"):
        return "u" + s[len("uint") :]
    if s.startswith("float"):
        return "f" + s[len("float") :]
    if s in ("boolean", "bool"):
        return "b"
    if s in ("utf8", "string", "object", "str"):
        return "s"
    if s in ("categorical", "category"):
        return "c"
    return s


def _mrmr_compute_y_fingerprint_sample(y, max_sample: int = 1000) -> str:
    """Cheap fingerprint of y's first ``max_sample`` values. Used by the identity cache when ``mrmr_identity_cache_include_y=True`` to distinguish legitimately different targets on the same X. Sample-based to keep the cost O(max_sample) even on 4M-row y.

    T2#11 2026-05-18 ``max_sample=1000`` rationale: on a 4M-row y with
    ``blake2b`` of a 1000-element float64 buffer the hash cost is
    O(8KB) ~= 5us, which is well below the >88-min cost of a single MRMR
    fit (the cache hit avoids). Collision probability across distinct
    targets on the same X is dominated by the y-distribution stride
    pattern - 1000 evenly-spaced samples capture both the head and tail
    of a 4M-row y so two truly different targets reliably hash apart.
    Raise if you observe spurious cache hits across distinct targets in
    your data (consult the cache HIT/STORED log line for collision
    diagnostics).
    """
    try:
        arr = np.asarray(y).reshape(-1)
        n = arr.shape[0]
        if n == 0:
            return "yfp_empty"
        # Use first + last + n_strided sample for entropy across the whole y.
        if n <= max_sample:
            sample = arr
        else:
            step = max(1, n // max_sample)
            sample = arr[::step][:max_sample]
        # Bit-exact ``tobytes()`` instead of the prior 6-decimal round. Two truly-equal floats already produce identical bytes, so the rounding only papered over EQUIVALENT-but-not-identical inputs - which is the collision case we DON'T want to merge (regression targets with legitimate precision below 1e-6, e.g. log-returns, normalised labels).
        payload = sample.astype(np.float64).tobytes()
        return hashlib.blake2b(payload, digest_size=10).hexdigest()
    except Exception:
        return f"yfp_id{id(y):x}"


def _mrmr_compute_x_fingerprint(X) -> str:
    """Stable per-process fingerprint of the X argument used to key the identity-cache.

    Captures column names, row count, CANONICAL dtype repr, and a 10-cell evenly-spaced
    content sample per column so two structurally-equivalent frames -- one polars,
    one pandas -- produce the SAME fingerprint while differently-content frames on
    the same schema produce DIFFERENT fingerprints. The cell sample mirrors
    ``_content_array_signature`` (10 evenly-spaced positions, rounded float repr,
    O(1) per column). On a 4M-row frame the sampling cost is O(n_cols) value
    reads and adds <1ms compared to the dtype-only path.
    """
    try:
        # Polars LazyFrame fast-path: collect_schema() once so we don't
        # trigger six "Determining the column names of a LazyFrame
        # requires resolving its schema" PerformanceWarnings (one per
        # X.columns / X.schema[c] access). For eager polars.DataFrame
        # and pandas these attribute accesses are O(1) and warn-free.
        _is_lazy_polars = hasattr(X, "collect_schema") and type(X).__name__ == "LazyFrame"
        if _is_lazy_polars:
            _resolved_schema = X.collect_schema()
            cols = tuple(sorted(str(c) for c in _resolved_schema.names()))
        elif hasattr(X, "columns"):
            cols = tuple(sorted(str(c) for c in X.columns))
        elif hasattr(X, "shape"):
            cols = tuple(str(i) for i in range(X.shape[1] if X.ndim > 1 else 1))
        else:
            cols = ()
        n_rows = int(X.shape[0]) if hasattr(X, "shape") else 0
        # Polars: ``X.schema[c]`` is the canonical accessor; ``X.dtypes`` is a
        # positional LIST so name-indexing fails. Pandas: ``X.dtypes`` is a
        # Series indexable by column name. Check ``schema`` first to route
        # polars correctly.
        if _is_lazy_polars:
            try:
                dtypes_repr = tuple((str(c), _canonicalise_dtype_str(_resolved_schema[c])) for c in _resolved_schema.names())
            except Exception:
                dtypes_repr = ()
        elif hasattr(X, "schema") and hasattr(X, "columns"):
            try:
                dtypes_repr = tuple((str(c), _canonicalise_dtype_str(X.schema[c])) for c in X.columns)
            except Exception:
                dtypes_repr = ()
        elif hasattr(X, "dtypes") and hasattr(X, "columns"):
            try:
                dtypes_repr = tuple((str(c), _canonicalise_dtype_str(X.dtypes[c])) for c in X.columns)
            except Exception:
                dtypes_repr = ()
        else:
            dtypes_repr = ()
        # Cell-content sample: 10 evenly-spaced positions per column. Prevents
        # same-schema-different-content X frames from colliding in the cache.
        # Skip cell-sampling for LazyFrame: each column read triggers a full
        # materialisation, the cost dominates the schema-only fingerprint.
        # The schema+dtype repr above is already collision-resistant enough.
        cell_sample: tuple = ()
        try:
            n_sample = min(10, n_rows) if n_rows > 0 else 0
            if n_sample > 0 and hasattr(X, "columns") and not _is_lazy_polars:
                step = max(1, n_rows // n_sample)
                positions = [i * step for i in range(n_sample) if i * step < n_rows]
                samples = []
                for c in X.columns:
                    try:
                        col = X[c] if not hasattr(X, "schema") else X.get_column(c)
                        # Polars Series + pandas Series both support iteration / indexing by int positions.
                        if hasattr(col, "to_numpy"):
                            arr = col.to_numpy()
                        else:
                            arr = np.asarray(col)
                        # Bit-exact cell bytes (matches the deliberately-tightened y-side ``tobytes()``); ``repr`` of a float papered over distinct-but-near values.
                        vals = tuple(np.asarray(arr[p]).tobytes() for p in positions if p < len(arr))
                        samples.append((str(c), vals))
                    except Exception:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
                        samples.append((str(c), ()))
                cell_sample = tuple(samples)
        except Exception:
            cell_sample = ()
        payload = repr((cols, n_rows, dtypes_repr, cell_sample)).encode()
        return hashlib.blake2b(payload, digest_size=12).hexdigest()
    except Exception:
        return f"fp_id{id(X):x}"


def _hashable_params_signature(params: dict) -> tuple:
    """Build a hashable tuple-of-(key, value) signature from ``get_params`` output.

    Non-hashable values (numpy arrays, callables, dicts of unhashables) fall
    back to content-based hashing. Used by ``MRMR._FIT_CACHE`` so two clones
    with the same constructor parameters and (X, y) share fitted state.

    CACHE-Low-2: numpy arrays are content-hashed via
    ``(arr.tobytes(), arr.shape, str(arr.dtype))`` instead of ``repr(v)``;
    ``repr`` is array-summary on numpy >= 2 (e.g. ``array([1, ..., 4])``)
    so two arrays with identical content but different sizes / abbreviation
    behaviour produced different signatures before, defeating the cache.
    """
    items = []
    for k, v in sorted(params.items()):
        try:
            hash(v)
            items.append((k, v))
        except TypeError:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
            # CACHE-Low-2: content-hash numpy arrays so a copy hashes equal
            # to the original. ``np.ndarray.tobytes`` + shape + dtype is the
            # cheapest exact fingerprint and works for all numpy versions.
            if isinstance(v, np.ndarray):
                try:
                    items.append((k, (v.tobytes(), v.shape, str(v.dtype))))
                    continue
                except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                    logger.debug("suppressed in _mrmr_fingerprints.py:268: %s", e)
                    pass
            try:
                items.append((k, repr(v)))
            except Exception:
                items.append((k, id(v)))
    return tuple(items)


def _content_array_signature(arr) -> tuple:
    """Cheap O(1) content-based fingerprint of an array / DataFrame.

    Used as a cache key when ``id()`` is unreliable: the training suite copies X between model iterations,
    so semantically-equal X have distinct ``id(X)`` but identical content. Samples 10 evenly-spaced positions;
    column names (when available) are included so ``df`` vs ``df.rename(...)`` produce different keys (otherwise
    a fit on the renamed frame would replay the prior ``feature_names_in_`` and ``transform()`` would mis-select).

    Returns (shape, dtype_str, sampled_values_bytes, col_names); falls back to ``id(arr)`` on failure.
    """
    try:
        # Capture column names (DataFrames only) BEFORE unwrap so signature distinguishes df vs df.rename(...).
        col_names = None
        if hasattr(arr, "columns"):
            try:
                col_names = tuple(str(c) for c in arr.columns)
            except Exception:
                col_names = None
        # Unwrap to numpy
        if hasattr(arr, "to_numpy"):
            try:
                np_arr = arr.to_numpy()
            except Exception:
                return ("uncached", id(arr))
        elif hasattr(arr, "values"):
            np_arr = arr.values
        else:
            np_arr = arr
        if not hasattr(np_arr, "shape") or not hasattr(np_arr, "dtype"):
            return ("uncached", id(arr))
        shape = np_arr.shape
        dtype_str = str(np_arr.dtype)
        # Strided 1024-position sample. The prior 10-cell sample collided on any two frames whose ten boundary cells happened to agree (e.g. column-wise outlier clip that preserves min/median/max rows). 1024 strided positions keep the fingerprint O(1) in n_rows yet make a content-collision astronomically unlikely while remaining cheaper than a full blake2b on 100GB frames.
        flat = np_arr.ravel()
        n = flat.size
        if n == 0:
            return (shape, dtype_str, b"", col_names)
        _n_samples = 1024
        idx = [int(i * (n - 1) / (_n_samples - 1)) for i in range(_n_samples)] if n >= _n_samples else list(range(n))
        try:
            sampled = bytes(flat[idx].tobytes())
        except Exception:
            return ("uncached", id(arr))
        return (shape, dtype_str, sampled, col_names)
    except Exception:
        return ("uncached", id(arr))


def _target_to_numpy_values(y) -> np.ndarray:
    """Return a numpy view/array for sklearn-style targets.

    Pandas exposes both ``to_numpy`` and ``values``; Polars Series exposes ``to_numpy`` only. MRMR only needs
    the 1-D/2-D target vector for its temporary target columns, so normalize y directly without copying X.
    """
    if isinstance(y, np.ndarray):
        return y
    if hasattr(y, "to_numpy"):
        # Modern path: covers pandas Series/DataFrame and polars Series. Returned array is
        # caller-safe (pandas returns a copy or read-only view; polars converts).
        return np.asarray(y.to_numpy())
    # ``.values`` fallback exists only for legacy duck-typed targets without ``to_numpy``
    # (custom wrappers around older sklearn arrays). Pandas/polars both flow through
    # the ``to_numpy`` branch above; the fallback can be dropped once legacy callers are gone.
    return np.asarray(y)


def _mrmr_y_corr_sample(y, max_sample: int = 1000) -> np.ndarray:
    """Deterministic strided numeric sample of a target for the identity-cache y-correlation gate.

    Returns a 1-D float array of length <= ``max_sample`` (never the full y; cheap on TB frames). Non-numeric
    targets are factorized to integer codes so a categorical y still yields a comparable numeric vector. The
    SAME stride rule is applied at write and read time so two samples align positionally for correlation.
    """
    arr = _target_to_numpy_values(y)
    arr = np.asarray(arr)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    n = arr.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    step = max(1, n // max_sample)
    sample = arr[::step][:max_sample]
    if not np.issubdtype(sample.dtype, np.number):
        # Factorize non-numeric (object/str/categorical) codes so the gate still has a numeric vector.
        _, codes = np.unique(sample.astype(str), return_inverse=True)
        return codes.astype(np.float64)
    return sample.astype(np.float64)


def _mrmr_y_corr(a: np.ndarray, b: np.ndarray):
    """|Pearson correlation| between two equal-length numeric samples; ``None`` when undefined.

    Returns ``None`` (caller treats as "cannot confirm") when the lengths differ, either sample is constant, or
    the correlation is NaN. Used only by the identity-cache y-correlation gate, never on a hot per-row path.
    """
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[0] != b.shape[0] or a.shape[0] < 2:
        return None
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    if a.std() == 0.0 or b.std() == 0.0:
        return None
    c = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(c):
        return None
    return c


def _target_name_signature(y) -> tuple:
    """Return a tuple of column names (or Series ``name``) for ``y`` if available.

    Two distinct targets with statistically-similar 10-cell samples (e.g. both balanced binary) must produce
    different cache keys; the target name (when the caller gave us a named Series / DataFrame) is the cheapest
    discriminator. Falls back to ``()`` when y is a bare ndarray.
    """
    try:
        if hasattr(y, "columns"):
            return tuple(str(c) for c in y.columns)
        name = getattr(y, "name", None)
        if name is not None:
            return (str(name),)
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed in _mrmr_fingerprints.py:401: %s", e)
        pass
    return ()


# iter627 (perf): single-entry "last computed" memo cache for MRMR's
# _full_x_content_hash. Caller sites at _mrmr_fit_impl.py:131 + :158
# pass the SAME X frame in tight succession (line 131 builds the
# signature for the cache lookup; line 158 re-runs the full hash for
# the cache store key). The blake2b hash chain runs on the full X
# frame bytes -- ~50ms on a 200MB frame per the iter59 bench, but
# grows linear with frame size; on c0009-like wide post-FE frames
# the second call wastes seconds re-hashing the same bytes.
#
# Same safety model as the iter625 _pipeline_cache._pre_pipeline_-
# cache_key memo: id() + shape discriminate against GC-recycled-id
# collisions, the two calls happen microseconds apart within MRMR.fit
# so the input is guaranteed unchanged.
_MRMR_LAST_X_HASH_CACHE: dict = {"id_shape": None, "hash": None}


def _full_x_content_hash(X) -> str:
    """Full-content blake2b digest over X for collision-free cache keying.

    The cheap 1024-strided ``_content_array_signature`` is still used as a fast-path discriminator but two
    truly different X frames whose 1024 strided cells happen to coincide (e.g. after a column-wise outlier
    clip that preserves the sampled positions) collide. Folding a full blake2b of X.tobytes() into the
    ``_FIT_CACHE`` key rules this out at O(len(X) bytes hashed) cost -- negligible next to MRMR fit time
    (~50ms on a 200MB frame, vs minutes of actual screening).

    Mirrors ``_full_y_content_hash`` so the y-side and X-side guarantees are symmetric. Returns ``""`` on
    any conversion failure so the caller can choose to skip the cache rather than serve a wrong replay.

    iter627 (perf): single-entry memo on (id(X), X.shape). MRMR.fit
    calls this twice with the same X (signature key + store key);
    the second call returns the cached hash without re-hashing.
    """
    sh = getattr(X, "shape", None)
    try:
        if hasattr(X, "to_numpy"):
            try:
                arr = X.to_numpy()
            except Exception:
                return ""
        elif hasattr(X, "values"):
            arr = X.values
        else:
            arr = X
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        # Object dtype (mixed-type pandas frames) cannot be hashed via tobytes deterministically; skip cache.
        if arr.dtype == object:
            return ""
        # iter627 fast-path memo: a recycled id(X) across fits (freed frame A, new frame B allocated at the
        # same address + same shape) made the old (id, shape)-only key return A's digest for B. Fold the cheap
        # 10-cell strided content signature into the key so a different B is a memo MISS (recompute), while the
        # intra-fit second call on the SAME X (id+shape+content all identical) still hits and skips the full hash.
        id_shape = (id(X), sh if sh is not None else (None,), _content_array_signature(arr))
        if _MRMR_LAST_X_HASH_CACHE["id_shape"] == id_shape:
            return str(_MRMR_LAST_X_HASH_CACHE["hash"])
        # blake2b reads the contiguous array via the buffer protocol directly
        # (no .tobytes() copy); bit-identical to hashing tobytes() bytes.
        h = hashlib.blake2b(np.ascontiguousarray(arr), digest_size=16)  # type: ignore[arg-type]  # ndarray supports the buffer protocol; hashlib's Buffer stub doesn't recognize it
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        # Fold column names (DataFrame-only) so df vs df.rename produce different keys.
        if hasattr(X, "columns"):
            try:
                h.update(",".join(str(c) for c in X.columns).encode())
            except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in _mrmr_fingerprints.py:470: %s", e)
                pass
        result = h.hexdigest()
        # Guard the single-slot memo write with the same lock the sibling identity cache uses; the publish-hash-before-key ordering is the
        # extra belt-and-braces so even a torn read (lock contention aside) can only see an OLD key paired with whatever hash -- a miss that
        # recomputes -- never a NEW id_shape paired with a stale hash from a prior different X.
        with _MRMR_IDENTITY_FP_LOCK:
            _MRMR_LAST_X_HASH_CACHE["hash"] = result
            _MRMR_LAST_X_HASH_CACHE["id_shape"] = id_shape
        return result
    except Exception:
        return ""


def _full_y_content_hash(y) -> str:
    """Return a hex digest covering the full content of ``y`` for cache-key disambiguation.

    The 10-sample ``_content_array_signature`` collides on targets whose sampled cells coincide (common for
    balanced binary classification). A full blake2b over y.tobytes() rules this out; the cost is O(len(y))
    bytes hashed, which is negligible next to the actual MRMR fit. Returns ``""`` on any conversion failure
    so the caller can choose to skip the cache.
    """
    try:
        arr = _target_to_numpy_values(y)
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        # ascontiguousarray covers non-contiguous slices; blake2b then reads its
        # buffer directly (no .tobytes() copy), bit-identical to hashing the bytes.
        h = hashlib.blake2b(np.ascontiguousarray(arr), digest_size=16)  # type: ignore[arg-type]  # ndarray supports the buffer protocol; hashlib's Buffer stub doesn't recognize it
        # Fold shape + dtype so reshape-only changes also bust the key.
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        return h.hexdigest()
    except Exception:
        return ""


# Constructor-parameter names of ``MRMR``. Resolved from the live class on first replay rather than at import time because importing ``MRMR`` here would
# close the ``_mrmr_class -> _mrmr_fingerprints`` import cycle; the one-time computation is guarded by ``_MRMR_IDENTITY_FP_LOCK`` so concurrent first
# replays (joblib threads) cannot race the module-global write.
_MRMR_INIT_PARAM_NAMES: frozenset[str] | None = None


def _replay_fitted_state(target: MRMR, source: MRMR) -> int:
    """Copy fitted attributes from ``source`` onto ``target``, preserving target's constructor params intact.

    A cloned MRMR has its own constructor params + ``signature=None`` + no fitted state; we want to inherit
    ``support_``, ``_engineered_recipes_``, ``_cat_fe_cache_``, ``ranking_``, etc. WITHOUT overwriting any
    constructor params. Returns the number of attributes replayed.
    """
    global _MRMR_INIT_PARAM_NAMES
    if _MRMR_INIT_PARAM_NAMES is None:
        with _MRMR_IDENTITY_FP_LOCK:
            if _MRMR_INIT_PARAM_NAMES is None:
                import inspect
                # Use target.__class__ instead of the (TYPE_CHECKING-only) ``MRMR``
                # symbol; both source and target are MRMR instances at runtime.
                _MRMR_INIT_PARAM_NAMES = frozenset(p for p in inspect.signature(target.__class__.__init__).parameters if p != "self")
    # ``MRMR._FIT_CACHE[key]`` stores the LIVE first-fitted instance, so the cached source's fitted attributes are the canonical copy every future
    # replay reads. Any mutable fitted attribute that is merely shallow-assigned onto a replayed target is then SHARED with the cache entry, so an
    # in-place mutation by any caller of the replayed instance silently corrupts the source and all later replays. The safe contract: deep-copy
    # everything that is not an immutable scalar (bool/int/float/complex), str/bytes/bytearray-of-immutable, None, or a frozenset; keep a fast-path
    # only for ndarrays, where the dense state can be large and the read-only freeze prevents accidental cache corruption without a full copy.
    import copy as _deepcopy_mod
    # Small public learned-index arrays a downstream caller may legitimately mutate in place. They are tiny (O(n_selected)), so a writeable per-replay
    # copy is cheap AND makes a cache-replayed instance behave IDENTICALLY to a cold-fit one (D7: cold fit's ``support_`` is writeable, so the replay's
    # must be too). Larger internal ndarrays keep the read-only freeze-and-share fast-path for density.
    _WRITEABLE_PUBLIC_INDEX_ARRAYS = ("support_", "ranking_")
    _IMMUTABLE_SCALAR_TYPES = (bool, int, float, complex, str, bytes, type(None), frozenset)
    # Per-instance transient runtime state that must NEVER be copied from ``source`` onto ``target``:
    # each instance's OWN lazily-created re-entrancy lock (finding #5, 05_concurrency_and_statistics.md)
    # guards ITS OWN concurrent-fit() detection. Replaying the cached source's lock object here would
    # alias the target's ``fit()`` wrapper (which already acquired the target's own, different, lock) to
    # release a lock it never acquired -- "release unlocked lock" RuntimeError on every cache-hit replay.
    _TRANSIENT_INSTANCE_KEYS = frozenset({"_fit_reentrancy_lock_", "_pre_fit_ctor_params_snapshot_"})
    n_replayed = 0
    for k, v in source.__dict__.items():
        if k in _MRMR_INIT_PARAM_NAMES or k in _TRANSIENT_INSTANCE_KEYS:
            continue
        if isinstance(v, np.ndarray):
            if k in _WRITEABLE_PUBLIC_INDEX_ARRAYS:
                # Hand the target its OWN writeable copy so cold-fit and replayed instances behave identically on public index arrays.
                target.__dict__[k] = v.copy()
            else:
                # Freeze the source ndarray so any in-place write raises instead of silently corrupting the shared cache entry; only freeze arrays
                # we own (skip views/slices) and share the (now read-only) buffer with the target for density.
                if v.flags.owndata and v.flags.writeable:
                    try:
                        v.flags.writeable = False
                    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                        logger.debug("suppressed in _mrmr_fingerprints.py:553: %s", e)
                        pass
                target.__dict__[k] = v
        elif isinstance(v, _IMMUTABLE_SCALAR_TYPES):
            target.__dict__[k] = v
        else:
            # Everything else (dict / list / set, pandas DataFrame/Series, dataclasses like CatFEState, friend_graph_, dict-of-arrays replay state,
            # any other object carrying mutable state) is deep-copied so a mutation on the replayed instance never reaches back to the cached source.
            try:
                target.__dict__[k] = _deepcopy_mod.deepcopy(v)
            except Exception:
                # A non-deepcopyable fitted artefact (e.g. a live framework handle) falls back to the legacy shared assignment rather than failing
                # the whole replay; such objects are rare and treated as read-only by callers.
                target.__dict__[k] = v
        n_replayed += 1
    return n_replayed


def _lazy_chunks(iterable, chunk_size: int):
    """Lazily yield successive ``chunk_size``-sized lists from ``iterable``.

    Peak memory is O(chunk_size), not O(len(iterable)). Used by ``_run_fe_step``
    to avoid materialising all O(k^2) feature pairs at once on wide datasets
    (k=5000 features = 12.5M pairs, ~300 MB tuple-list before chunking).
    """
    if chunk_size < 1:
        raise ValueError(f"_lazy_chunks: chunk_size must be >= 1, got {chunk_size}")
    it = iter(iterable)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk
