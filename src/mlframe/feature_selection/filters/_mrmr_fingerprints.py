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

import copy
import hashlib
import logging
import warnings
from collections import OrderedDict
from itertools import islice
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

try:
    from astropy.stats import histogram as _astropy_histogram
except (ImportError, AttributeError):
    _astropy_histogram = None

if TYPE_CHECKING:
    from .mrmr import MRMR  # noqa: F401 -- forward ref for the type annotation


def histogram(a, bins="auto", **kwargs):
    """Astropy histogram with np.histogram fallback. See
    ``mlframe.feature_engineering.numerical.histogram`` for the rationale.
    """
    if _astropy_histogram is not None:
        return _astropy_histogram(a, bins=bins, **kwargs)
    return np.histogram(a, bins=bins, **kwargs)


from numpy.polynomial.hermite import hermval
from scipy import special as sp
from scipy.stats import mode

from catboost import CatBoostClassifier

from pyutilz.numbalib import (
    generate_combinations_recursive_njit,
    python_dict_2_numba_dict,
    set_numba_random_seed,
)
from pyutilz.parallel import mem_map_array, parallel_run, split_list_into_chunks
from pyutilz.pythonlib import (
    get_parent_func_args,
    sort_dict_by_value,
    store_params_in_object,
)
from pyutilz.system import tqdmu

from mlframe.core.arrays import arrayMinMax
from mlframe.feature_selection.wrappers import RFECV
from mlframe.metrics.core import compute_probabilistic_multiclass_error
from mlframe.utils.misc import set_random_seed

from ._internals import (
    ENSURE_ARROW_DF_SUPPORT,
    GPU_MAX_BLOCK_SIZE,
    LARGE_CONST,
    MAX_CONFIRMATION_CAND_NBINS,
    MAX_ITERATIONS_TO_TRACK,
    MAX_JOBLIB_NBYTES,
    NMAX_NONPARALLEL_ITERS,
    sanitize,
)
from ._numba_utils import arr2str, count_cand_nbins, unpack_and_sort
from .discretization import (
    categorize_dataset,
    discretize_array,
)
from .feature_engineering import (
    FE_DEFAULT_SUBSAMPLE_N,
    check_prospective_fe_pairs,
    compute_pairs_mis,
    create_binary_transformations,
    create_unary_transformations,
    get_existing_feature_name,
    get_new_feature_name,
)
from .gpu import init_kernels, mi_direct_gpu
from .info_theory import (
    compute_mi_from_classes,
    conditional_mi,
    entropy,
    merge_vars,
    mi,
)
from .permutation import distribute_permutations, mi_direct, parallel_mi
from .evaluation import (
    evaluate_candidate,
    evaluate_candidates,
    evaluate_gain,
    find_best_partial_gain,
    get_candidate_name,
    handle_best_candidate,
    should_skip_candidate,
)
from .fleuret import (
    get_fleuret_criteria_confidence,
    get_fleuret_criteria_confidence_parallel,
    parallel_fleuret,
)
from .screen import postprocess_candidates, screen_predictors

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
# Layer 3 pre-batch: thresholds for the dispatch_batch_pair_mi pre-fill path in _run_fe_step.
#
# * _MRMR_BATCH_PRECOMPUTE_MAX_K: cap on numeric_vars_to_consider size. The pre-fill materialises every (a, b)
#   pair tuple, which is O(k^2). At k=200 -> 19_900 pair tuples (~480 KB of Python tuples); at k=500 -> 124_750
#   tuples (~3 MB). 200 is the safe sweet spot: covers the typical fe_ntop_features=30-50 axis with margin and
#   keeps the materialised tuple list under 1 MB. Above the cap the legacy combinations(...) lazy path runs
#   unchanged.
# * _MRMR_BATCH_PRECOMPUTE_MIN_PAIRS: smallest pair count where the dispatcher overhead amortises. Below this
#   the per-pair joblib path is competitive (and avoids a redundant numba.cuda first-call compile when no GPU
#   speedup would materialise).
_MRMR_BATCH_PRECOMPUTE_MAX_K = 200
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
        return "i" + s[len("int"):]
    if s.startswith("uint"):
        return "u" + s[len("uint"):]
    if s.startswith("float"):
        return "f" + s[len("float"):]
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
        _is_lazy_polars = (
            hasattr(X, "collect_schema")
            and type(X).__name__ == "LazyFrame"
        )
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
                dtypes_repr = tuple(
                    (str(c), _canonicalise_dtype_str(_resolved_schema[c]))
                    for c in _resolved_schema.names()
                )
            except Exception:
                dtypes_repr = ()
        elif hasattr(X, "schema") and hasattr(X, "columns"):
            try:
                dtypes_repr = tuple(
                    (str(c), _canonicalise_dtype_str(X.schema[c]))
                    for c in X.columns
                )
            except Exception:
                dtypes_repr = ()
        elif hasattr(X, "dtypes") and hasattr(X, "columns"):
            try:
                dtypes_repr = tuple(
                    (str(c), _canonicalise_dtype_str(X.dtypes[c]))
                    for c in X.columns
                )
            except Exception:
                dtypes_repr = ()
        else:
            dtypes_repr = ()
        # Cell-content sample: 10 evenly-spaced positions per column. Prevents
        # same-schema-different-content X frames from colliding in the cache.
        # Skip cell-sampling for LazyFrame: each column read triggers a full
        # materialisation, the cost dominates the schema-only fingerprint.
        # The schema+dtype repr above is already collision-resistant enough.
        cell_sample = ()
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
                        vals = tuple(repr(arr[p]) for p in positions if p < len(arr))
                        samples.append((str(c), vals))
                    except Exception:
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
        except TypeError:
            # CACHE-Low-2: content-hash numpy arrays so a copy hashes equal
            # to the original. ``np.ndarray.tobytes`` + shape + dtype is the
            # cheapest exact fingerprint and works for all numpy versions.
            if isinstance(v, np.ndarray):
                try:
                    items.append(
                        (k, (v.tobytes(), v.shape, str(v.dtype)))
                    )
                    continue
                except Exception:
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
        return y.to_numpy()
    # ``.values`` fallback exists only for legacy duck-typed targets without ``to_numpy``
    # (custom wrappers around older sklearn arrays). Pandas/polars both flow through
    # the ``to_numpy`` branch above; the fallback can be dropped once legacy callers are gone.
    return np.asarray(y)


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
    except Exception:
        pass
    return ()


def _full_x_content_hash(X) -> str:
    """Full-content blake2b digest over X for collision-free cache keying.

    The cheap 1024-strided ``_content_array_signature`` is still used as a fast-path discriminator but two
    truly different X frames whose 1024 strided cells happen to coincide (e.g. after a column-wise outlier
    clip that preserves the sampled positions) collide. Folding a full blake2b of X.tobytes() into the
    ``_FIT_CACHE`` key rules this out at O(len(X) bytes hashed) cost -- negligible next to MRMR fit time
    (~50ms on a 200MB frame, vs minutes of actual screening).

    Mirrors ``_full_y_content_hash`` so the y-side and X-side guarantees are symmetric. Returns ``""`` on
    any conversion failure so the caller can choose to skip the cache rather than serve a wrong replay.
    """
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
        buf = np.ascontiguousarray(arr).tobytes()
        h = hashlib.blake2b(buf, digest_size=16)
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        # Fold column names (DataFrame-only) so df vs df.rename produce different keys.
        if hasattr(X, "columns"):
            try:
                h.update(",".join(str(c) for c in X.columns).encode())
            except Exception:
                pass
        return h.hexdigest()
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
        # ascontiguousarray covers non-contiguous slices; tobytes itself would copy too but this is explicit.
        buf = np.ascontiguousarray(arr).tobytes()
        h = hashlib.blake2b(buf, digest_size=16)
        # Fold shape + dtype so reshape-only changes also bust the key.
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        return h.hexdigest()
    except Exception:
        return ""


# Constructor-parameter names of ``MRMR``. Populated lazily to avoid importing inspect at module load.
_MRMR_INIT_PARAM_NAMES: frozenset[str] | None = None


def _replay_fitted_state(target: MRMR, source: MRMR) -> int:
    """Copy fitted attributes from ``source`` onto ``target``, preserving target's constructor params intact.

    A cloned MRMR has its own constructor params + ``signature=None`` + no fitted state; we want to inherit
    ``support_``, ``_engineered_recipes_``, ``_cat_fe_cache_``, ``ranking_``, etc. WITHOUT overwriting any
    constructor params. Returns the number of attributes replayed.
    """
    global _MRMR_INIT_PARAM_NAMES
    if _MRMR_INIT_PARAM_NAMES is None:
        import inspect
        # Use target.__class__ instead of the (TYPE_CHECKING-only) ``MRMR``
        # symbol; both source and target are MRMR instances at runtime.
        _MRMR_INIT_PARAM_NAMES = frozenset(
            p for p in inspect.signature(target.__class__.__init__).parameters
            if p != "self"
        )
    n_replayed = 0
    for k, v in source.__dict__.items():
        if k in _MRMR_INIT_PARAM_NAMES:
            continue
        # Shallow assign; deep copy would defeat the cache (we WANT to share cached numpy arrays).
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

