"""Constants, small pure-Python helpers, and warnings setup.

Each constant has a docstring explaining (a) what bound it prevents,
(b) the failure mode when exceeded or under-set, and (c) the empirical
or algorithmic basis for the chosen value. Magic numbers without
docstrings are not allowed in this module.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import numba
from numba import njit
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning


# =============================================================================
# Warnings setup (mirrors legacy filters.py top-of-file initialisation)
# =============================================================================
warnings.filterwarnings("ignore", module=".*_discretization")
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


# =============================================================================
# Constants
# =============================================================================

#: Maximum bytes-per-arg before joblib spills the worker payload to disk.
#: Keeping this small (1e3 bytes) forces inline transmission of the small
#: njit-typed.Dict snapshots used for cache propagation; raising it would
#: cause memmap creation overhead on every short-lived joblib job.
MAX_JOBLIB_NBYTES: float = 1e3

#: Below this iteration count the inner loop runs sequentially (no joblib).
#: At < 2 permutations the per-call joblib spawn cost dominates the work.
NMAX_NONPARALLEL_ITERS: int = 2

#: How many recent screening iterations to retain in observability metadata.
#: Used by `screen_predictors` to log a tail summary; larger values bloat
#: the typed.Dict that passes through joblib worker boundaries.
MAX_ITERATIONS_TO_TRACK: int = 5

#: Sentinel used as initial best-gain in early-exit comparisons. 1e30 is
#: large enough that no real MI value will ever beat it but still fits in
#: float64 without saturating subsequent arithmetic.
LARGE_CONST: float = 1e30

#: CUDA block size for the joint-histogram and MI raw kernels in `gpu.py`.
#: 1024 is the per-block thread limit on every CUDA compute capability >= 2.0
#: that we target; reducing it would leave SMs under-utilised, raising it
#: above 1024 fails kernel launch.
GPU_MAX_BLOCK_SIZE: int = 1024

#: Default cutoff above which the per-candidate confirmation step stops
#: (because the conditioning set has too many bins for permutation testing
#: to converge in reasonable time). The B13 plan replaces this with a
#: kwarg ``MRMR(max_confirmation_cand_nbins=...)`` whose default is
#: ``quantization_nbins ** interactions_max_order * 2``; this constant
#: stays as a deprecation re-export until that lands at etap 12.
MAX_CONFIRMATION_CAND_NBINS: int = 50

#: Use column names instead of `iloc` for Arrow-backed pandas DataFrames
#: produced by polars zero-copy conversion. Older pandas versions could
#: silently degrade Arrow-extension columns when accessed via positional
#: iloc; using by-name access is safe across every supported version.
ENSURE_ARROW_DF_SUPPORT: bool = True


# =============================================================================
# Pure-Python helpers
# =============================================================================


def smart_log(x: np.ndarray) -> np.ndarray:
    """Numerically-safe ``log(x)`` that handles non-positive inputs by
    additive shifting. Preserves the input's dtype (a previous cast to
    float32 silently demoted float64 inputs)."""
    x_min = np.nanmin(x)
    if x_min > 0:
        return np.log(x)
    return np.log(x + (1e-5 - x_min))


def njit_functions_dict(
    dict_: dict,
    exceptions: Sequence = ("grad1", "grad2", "sinc", "log", "logn", "greater", "less", "equal"),
) -> None:
    """Try replacing functions in ``dict_`` with their ``@njit`` equivalents,
    skipping the named ``exceptions`` (functions known to fail compilation
    or that Numba inlines worse than CPython)."""
    for key, func in dict_.items():
        if key not in exceptions:
            try:
                dict_[key] = njit(func)
            except Exception:
                pass


def sanitize(obj):
    """Recursively convert ``numba.typed.Dict`` and friends back to plain
    Python / NumPy so the result can be pickled by joblib / cloudpickle."""
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
