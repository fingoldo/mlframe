"""Constants, small pure-Python helpers, and warnings setup. Each constant must document (a) what bound it prevents, (b) the failure mode on misset value,
and (c) the empirical / algorithmic basis. Magic numbers without docstrings are not allowed here."""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import numba
from numba import njit
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning


# Warnings setup (mirrors legacy filters.py top-of-file initialisation).
warnings.filterwarnings("ignore", module=".*_discretization")
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


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
    """Numerically-safe ``log(x)``: additively shifts non-positive inputs. Preserves input dtype (a prior cast to float32 silently demoted float64)."""
    x_min = np.nanmin(x)
    if x_min > 0:
        return np.log(x)
    return np.log(x + (1e-5 - x_min))


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
