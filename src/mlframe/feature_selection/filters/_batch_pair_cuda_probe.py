"""CUDA availability probe for the GPU pair-usability kernels.

``_CUDA_AVAIL`` and ``_nb_cuda`` are resolved once at import. ``batch_pair_usability_corr_gpu`` imports them, so the
kernels and every caller see the same module globals. Logging keeps the parent module's logger name.
"""

import logging

logger = logging.getLogger("mlframe.feature_selection.filters.batch_pair_usability_corr_gpu")

from numba import cuda as _nb_cuda

try:
    from pyutilz.core.pythonlib import is_cuda_available as _pyutilz_is_cuda_available

    _CUDA_AVAIL = _pyutilz_is_cuda_available()
except ImportError as e:
    # pyutilz genuinely absent: expected on a lean install, permanent, quiet.
    logger.debug("pyutilz.core.pythonlib unavailable (%s); falling back to numba.cuda.is_available().", e)
except Exception as e:
    # This is the `_select_mi_backend` mechanism exactly, and it is worth a WARNING rather than a debug line:
    # the probe module imports code that touches CUDA at import time, so a TRANSIENT device fault raised here
    # is indistinguishable from "no CUDA" -- and because `_CUDA_AVAIL` is resolved once at MODULE IMPORT, that
    # one hiccup disables this module's CUDA kernels for the entire process, invisibly.
    logger.warning(
        "is_cuda_available() probe raised %s (%s) during module import, which is NOT 'pyutilz absent'. "
        "_CUDA_AVAIL is resolved ONCE at import, so if this was a transient device fault this module's CUDA "
        "kernels are disabled for the whole process. Falling back to numba.cuda.is_available().",
        type(e).__name__,
        e,
    )
    try:
        _CUDA_AVAIL = bool(getattr(_nb_cuda, "is_available", lambda: False)()) if _nb_cuda is not None else False
    except Exception as e2:
        logger.warning("numba.cuda.is_available() probe failed, assuming CUDA unavailable: %s", e2)
        _CUDA_AVAIL = False

# Require numba.cuda to actually compile+launch a kernel (not just device presence) so a cudatoolkit/NVVM
# mismatch routes to CPU instead of raising NvvmSupportError mid-dispatch (mirrors batch_pair_mi_gpu.py).
try:
    from ._internals import numba_cuda_can_compile as _numba_cuda_can_compile

    _CUDA_AVAIL = _CUDA_AVAIL and _numba_cuda_can_compile()
except Exception as e:
    logger.warning("numba_cuda_can_compile() check failed, assuming CUDA unavailable: %s", e)
    _CUDA_AVAIL = False
