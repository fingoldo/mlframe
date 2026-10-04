"""Fit-end release of the process-global caches that only help within one selector fit (device operand tables, F-order CMI copy, memmap dumps)."""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_PKG = "mlframe.feature_selection.filters"

# (module, releaser) pairs: a module that was never imported holds nothing, so it is looked up in sys.modules instead of being imported.
_RELEASERS = (
    (f"{_PKG}._gpu_resident_materialise", "clear_gpu_operand_table_caches"),
    (f"{_PKG}.info_theory._cmi_cuda", "reset_cmi_forder_cache"),
    (f"{_PKG}._joblib_safe", "release_fit_constant_memmaps"),
)


def release_fit_scoped_caches() -> None:
    """Free the fit-scoped caches (GPU operand tables, F-order CMI matrix copy, memmap dumps); each rebuilds on the next miss, so this is always safe."""
    for mod_name, fn_name in _RELEASERS:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        try:
            getattr(mod, fn_name)()
        except Exception as exc:
            logger.warning("fit-end release %s.%s failed (%s: %s); the cache stays resident", mod_name, fn_name, type(exc).__name__, exc)
