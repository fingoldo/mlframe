"""Constants, small pure-Python helpers, and warnings setup. Each constant must document (a) what bound it prevents, (b) the failure mode on misset value,
and (c) the empirical / algorithmic basis. Magic numbers without docstrings are not allowed here."""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import numba
from numba import njit
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

# Wave 87 (2026-05-21): scoped numba/discretization warnings suppressor.
# Replaces the prior module-level filter mutation which silently poisoned
# the process-global filter for every importer.
from contextlib import contextmanager as _contextmanager


@_contextmanager
def suppress_numba_warnings():
    """Scope-local numba + discretization-internal warning suppression.

    Use inside the numba-heavy entry points (kernel hot loops, hist binning)
    rather than mutating the global filter at import time:

        with suppress_numba_warnings():
            _njit_kernel(...)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module=".*_discretization")
        warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
        warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
        yield


_NUMBA_CUDA_CAN_COMPILE: bool | None = None


def numba_cuda_can_compile() -> bool:
    """True only if numba.cuda can actually COMPILE + LAUNCH a kernel on this host.

    ``numba.cuda.is_available()`` / ``pyutilz.is_cuda_available()`` report device PRESENCE,
    not whether NVVM supports the device's compute capability. On a host with a GPU but a
    cudatoolkit/numba mismatch the device probe returns True yet the first real kernel launch
    raises ``NvvmSupportError`` ("No supported GPU compute capabilities found"). Gating the
    numba.cuda backend on device-presence alone therefore crashes the dispatcher instead of
    falling back to cupy/CPU. This probe compiles + launches a trivial kernel ONCE (result
    cached) so callers can route around a numba.cuda stack that cannot actually run kernels.
    cupy is gated separately (it can work even when numba.cuda's NVVM cannot)."""
    global _NUMBA_CUDA_CAN_COMPILE
    if _NUMBA_CUDA_CAN_COMPILE is not None:
        return _NUMBA_CUDA_CAN_COMPILE
    try:
        from numba import cuda as _cuda

        if not _cuda.is_available():
            _NUMBA_CUDA_CAN_COMPILE = False
            return False

        @_cuda.jit
        def _probe(out):  # pragma: no cover - trivial device kernel
            """Minimal device kernel: writes a sentinel so the host side can confirm compile+launch actually succeeded."""
            out[0] = 1

        out = _cuda.to_device(np.zeros(1, dtype=np.int32))
        _probe[1, 1](out)
        _cuda.synchronize()
        _NUMBA_CUDA_CAN_COMPILE = int(out.copy_to_host()[0]) == 1
    except Exception:
        # NvvmSupportError, missing toolkit, driver mismatch, OOM at probe -- any failure means
        # the numba.cuda path is unusable on this host; route to cupy/CPU.
        _NUMBA_CUDA_CAN_COMPILE = False
    return _NUMBA_CUDA_CAN_COMPILE


# =============================================================================
# Constants
# =============================================================================

#: Max bytes-per-arg before joblib spills worker payload to disk. Small (1e3) to force inline transmission of njit-typed.Dict cache snapshots; raising it
#: incurs memmap creation overhead on every short-lived joblib job.
MAX_JOBLIB_NBYTES: float = 1e3

#: Below this iteration count the inner loop runs sequentially. At < 2 permutations the per-call joblib spawn cost dominates the work.
NMAX_NONPARALLEL_ITERS: int = 2

#: How many recent screening iterations to retain in observability metadata. Used by ``screen_predictors`` for a tail summary; larger values bloat the
#: typed.Dict that crosses joblib worker boundaries.
MAX_ITERATIONS_TO_TRACK: int = 5

#: Sentinel initial best-gain in early-exit comparisons. 1e30 is large enough that no real MI value will ever beat it but still fits in float64 without
#: saturating subsequent arithmetic.
LARGE_CONST: float = 1e30

#: CUDA block size for joint-histogram and MI raw kernels in ``gpu.py``. 1024 is the per-block thread limit on every CUDA compute capability >= 2.0 we
#: target; lowering under-utilises SMs, raising above 1024 fails kernel launch.
GPU_MAX_BLOCK_SIZE: int = 1024

#: Default cutoff above which the per-candidate confirmation step stops (conditioning set too large for permutation testing to converge). Replaced by
#: kwarg ``MRMR(max_confirmation_cand_nbins=...)`` with default ``quantization_nbins ** interactions_max_order * 2``; kept here as deprecation re-export.
MAX_CONFIRMATION_CAND_NBINS: int = 50

#: Access Arrow-backed pandas DataFrames (from polars zero-copy conversion) by column name instead of ``iloc``. Older pandas could silently degrade
#: Arrow-extension columns via positional iloc; by-name access is safe across every supported version.
ENSURE_ARROW_DF_SUPPORT: bool = True


# =============================================================================
# Pure-Python helpers
# =============================================================================


def smart_log(x: np.ndarray) -> np.ndarray:
    """Numerically-safe ``log(x)``: additively shifts non-positive inputs. Preserves input dtype (a prior cast to float32 silently demoted float64).

    When ``x_min == x_max == 0`` (a column of pure zeros emerging from
    Optuna's coefficient sweep at very small parameter magnitudes), the
    additive shift becomes ``1e-5`` exactly, ``log(1e-5)`` is finite, no
    warnings. But when ``x`` contains both zeros and very small positive
    floats, ``x + (1e-5 - x_min)`` can underflow to zero in float32 and
    trigger ``divide by zero encountered in log``. The downstream
    consumer ``nan_to_num`` already sanitises the column; suppress the
    bare-numpy warning emit here so the stderr stream stays clean for
    real problems."""
    x_min = np.nanmin(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        if x_min > 0:
            return np.asarray(np.log(x))
        return np.asarray(np.log(x + (1e-5 - x_min)))


def canonical_group_token(value) -> str:
    """Stable per-value group/category key robust to int<->float dtype drift.

    Fit and predict can see the SAME group/category values arriving in different
    dtypes (polars int->float promotion, a pandas join upcast, a re-read CSV).
    A bare ``str`` makes the integer ``1`` (``'1'``) and the float ``1.0``
    (``'1.0'``) DIFFERENT keys, so a fit-on-int / predict-on-float drift misses
    every per-group lookup and silently routes every row to the global fallback
    -- the engineered column is then computed from the wrong (global) statistic
    with no error. Collapse integral-valued numerics so ``1``, ``1.0``,
    ``np.int64(1)``, ``np.float64(1.0)`` all map to ``'1'``; non-integral floats
    keep their full repr; non-numeric labels pass through ``str`` unchanged.

    Mirrors ``training.composite.transforms._canonical_group_key`` so the whole
    framework keys per-group state identically."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if np.isfinite(f) and f == int(f):
            return str(int(f))
        return repr(f)
    return str(value)


def group_key_strings(col) -> np.ndarray:
    """Object array of per-row canonical group keys (int<->float drift safe).

    Group columns are low-cardinality, so only the distinct values are
    canonicalised and gathered back via ``np.unique(return_inverse)`` --
    ``canonical_group_token`` runs per-unique instead of per-row (6-10x on
    typical group keys; identical result to the per-row map). Integral-valued
    int and float labels collapse to the same key (``1`` and ``1.0`` -> ``'1'``)
    so a fit-int / predict-float dtype drift still hits the per-group entry
    instead of falling through to the global value. NaN -> ``'nan'``,
    None -> ``'None'`` (unchanged); non-integral floats keep their repr."""
    arr = col.to_numpy()
    if arr.dtype.kind in ("i", "u", "b"):
        # Integral / bool: native dtype already collapses 1 and 1.0 to "1" via
        # int casting on the other side; keep the fast per-unique str path.
        uniq, inv = np.unique(arr, return_inverse=True)
        if arr.dtype.kind == "b":
            return uniq.astype(str).astype(object)[inv]
        return np.array([str(int(u)) for u in uniq], dtype=object)[inv]
    if arr.dtype.kind == "f":
        # Float: canonicalise integral values to their int form so they match a
        # fit/predict counterpart that arrived as integer dtype.
        uniq, inv = np.unique(arr, return_inverse=True)
        toks = np.array([canonical_group_token(u) for u in uniq], dtype=object)
        return np.asarray(toks[inv])
    # object / categorical / string: per-value canonical (handles mixed dtypes,
    # python ints/floats embedded in object arrays, NaN sentinels via str()).
    return np.asarray(col.astype(object).map(canonical_group_token).to_numpy())


# Process-global cache of already-njit-compiled dispatchers, keyed by the SOURCE
# IDENTITY of the original Python callable. ``create_unary_transformations`` /
# ``create_binary_transformations`` are rebuilt many times per fit (FE step, recipe
# replay, fingerprinting), and each prior call wrapped every registry entry in a FRESH
# ``njit(func)`` -- a brand-new dispatcher with an empty compile cache -- so the SAME
# signature (e.g. ``_safe_div(ndarray, ndarray)``) was re-LLVM-compiled on every rebuild.
# On the canonical n=100k fit that redundant recompilation cost ~13s of pure llvmlite time.
# Keying by ``__code__`` (not ``id(func)``) keeps the cache effective for the registries'
# lambdas too: those lambda OBJECTS are recreated each call, but their underlying code
# object is interned per source location, so structurally identical lambdas share a single
# compiled dispatcher. Closure-bound lambdas (e.g. ``_order=order`` in the maximal preset)
# carry distinct closure cells; we include ``__defaults__``/``__closure__`` cell ids in the
# key so two captures with different bound constants never collide.
_NJIT_DISPATCHER_CACHE: dict = {}


def _njit_cache_key(func):
    """Build a hashable key identifying ``func``'s compiled form for ``_NJIT_DISPATCHER_CACHE`` (code object + defaults + closure-cell identities); returns ``None`` when the function is not cacheable (no ``__code__``, or an unhashable default)."""
    code = getattr(func, "__code__", None)
    if code is None:
        return None  # builtins / ufuncs have no __code__ -> not cacheable, compile fresh
    closure = getattr(func, "__closure__", None)
    cell_ids = tuple(id(c) for c in closure) if closure else ()
    key = (code, func.__defaults__, cell_ids)
    # ``__defaults__`` carries default-arg VALUES; if one is unhashable (e.g. a numpy array bound as a
    # default), the key can't index the dict. Return None so the caller still njit-compiles the function
    # fresh (correct, just uncached) instead of letting a hash error drop it back to pure-Python.
    try:
        hash(key)
    except TypeError:
        return None
    return key


def njit_functions_dict(
    dict_: dict,
    exceptions: Sequence = ("grad1", "grad2", "sinc", "log", "logn", "greater", "less", "equal"),
) -> None:
    """Replace functions in ``dict_`` with ``@njit`` equivalents, skipping ``exceptions`` (known to fail compilation or that Numba inlines worse than CPython).

    Compiled dispatchers are memoised process-wide (see ``_NJIT_DISPATCHER_CACHE``) so the
    registry rebuilds within a single fit reuse one dispatcher per distinct source callable
    instead of recompiling the same signature repeatedly."""
    for key, func in dict_.items():
        if key not in exceptions:
            try:
                ck = _njit_cache_key(func)
                if ck is not None:
                    cached = _NJIT_DISPATCHER_CACHE.get(ck)
                    if cached is None:
                        cached = njit(func)
                        _NJIT_DISPATCHER_CACHE[ck] = cached
                    dict_[key] = cached
                else:
                    dict_[key] = njit(func)
            except Exception:  # nosec B110 - best-effort path
                pass


def sanitize(obj):
    """Recursively convert ``numba.typed.Dict`` and friends back to plain Python / NumPy so the result can be pickled by joblib / cloudpickle."""
    if isinstance(obj, numba.typed.Dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return np.array(obj, copy=True)
    return obj
