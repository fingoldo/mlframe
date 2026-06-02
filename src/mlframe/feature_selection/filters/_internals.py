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
            return np.log(x)
        return np.log(x + (1e-5 - x_min))


def group_key_strings(col) -> np.ndarray:
    """Object array of per-row string group keys, equivalent to
    ``col.astype(object).map(str).to_numpy()``.

    Integer / unsigned / bool columns can never hold None or NaN, and the
    group columns these FE layers key on are low-cardinality, so only the
    distinct values are stringified and gathered back via
    ``np.unique(return_inverse)`` -- str() runs per-unique instead of per-row
    (6-10x on typical group keys; bit-identical to the per-row map). Any other
    dtype falls back to the exact ``map(str)`` semantics (NaN -> ``"nan"``,
    None -> ``"None"``, floats/objects/categoricals unchanged)."""
    arr = col.to_numpy()
    if arr.dtype.kind in ("i", "u", "b"):
        uniq, inv = np.unique(arr, return_inverse=True)
        return uniq.astype(str).astype(object)[inv]
    return col.astype(object).map(str).to_numpy()


def njit_functions_dict(
    dict_: dict,
    exceptions: Sequence = ("grad1", "grad2", "sinc", "log", "logn", "greater", "less", "equal"),
) -> None:
    """Replace functions in ``dict_`` with ``@njit`` equivalents, skipping ``exceptions`` (known to fail compilation or that Numba inlines worse than CPython)."""
    for key, func in dict_.items():
        if key not in exceptions:
            try:
                dict_[key] = njit(func)
            except Exception:
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
