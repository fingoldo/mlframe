"""Optional Intel-acceleration patch for sklearn-based FE primitives.

sklearn-intelex (https://pypi.org/project/scikit-learn-intelex/) provides drop-in C++ kernels
for several sklearn estimators we use in the kNN-bottlenecked FE stack:
    - NearestNeighbors (when used outside _knn_helper)
    - RobustScaler
    - BayesianGaussianMixture / GaussianMixture
    - KMeans

On Intel CPUs the speedup is 5-50x for these primitives. On AMD CPUs sklearn-intelex still
provides some speedup via MKL, smaller (1.5-3x) but free; on non-x86 it's a no-op.

The patch is opt-out via ``MLFRAME_USE_SKLEARNEX=0`` and called idempotently: applying the
patch repeatedly is safe (sklearn-intelex internally guards), so each FE function can call
``try_patch_sklearn()`` at entry without performance impact after the first call.

If sklearn-intelex is not installed the patch is a no-op (returns False). Install via
``pip install scikit-learn-intelex`` to opt in.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_PATCH_APPLIED: bool = False
_PATCH_ATTEMPTED: bool = False


def try_patch_sklearn() -> bool:
    """Apply sklearn-intelex's patch_sklearn() once per process, if available.

    Returns True if the patch is active (was applied this call or in a prior call), False
    if sklearn-intelex isn't installed or env var ``MLFRAME_USE_SKLEARNEX=0`` was set.

    Idempotent: subsequent calls after the first successful patch are O(1) (just return True).
    """
    global _PATCH_APPLIED, _PATCH_ATTEMPTED
    if _PATCH_APPLIED:
        return True
    if _PATCH_ATTEMPTED:
        return False
    _PATCH_ATTEMPTED = True
    if os.environ.get("MLFRAME_USE_SKLEARNEX", "1") == "0":
        logger.info("[_intel_patch] MLFRAME_USE_SKLEARNEX=0; skipping sklearn-intelex patch.")
        return False
    try:
        from sklearnex import patch_sklearn
        patch_sklearn()
        _PATCH_APPLIED = True
        logger.info("[_intel_patch] sklearn-intelex patch applied (5-50x speedup on Intel CPUs).")
        return True
    except ImportError:
        logger.info(
            "[_intel_patch] sklearn-intelex not installed; running on stock sklearn. "
            "Install (pip install scikit-learn-intelex) for 5-50x speedup on Intel CPUs."
        )
        return False
    except Exception as exc:  # pragma: no cover
        logger.warning("[_intel_patch] sklearn-intelex patch failed: %s; running on stock sklearn.", exc)
        return False
