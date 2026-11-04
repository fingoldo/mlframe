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
    (f"{_PKG}.info_theory._cmi_resident_cache", "clear_cmi_resident_cache"),
    (f"{_PKG}.info_theory._cmi_resident_cache", "clear_cmi_xc_resident_cache"),
)


def _free_cached_device_blocks() -> None:
    """Return the cached free blocks of the default cupy pool and of every FE stage slot pool to the device.

    A slot frees its pool's blocks when its stage ends, while the stage's arrays are still alive in the caches released above; once those die,
    their blocks sit cached in the persistent slot pools (and the default pool) for the rest of the process. Measured after a warm 100k fit
    on the 4 GB card: 270 MB cached in the default pool and 530 MB in the three slot pools, which the next fit's planner saw as used memory, so
    its slots got small budgets and fell into the OOM fallbacks (80 s first fit, 258-338 s for the following ones)."""
    cp = sys.modules.get("cupy")
    if cp is None:
        return
    cp.get_default_memory_pool().free_all_blocks()
    sched = sys.modules.get(f"{_PKG}._fe_device_scheduler")
    if sched is not None:
        for pool in list(sched._SLOT_POOLS.values()):
            pool.free_all_blocks()


def release_fit_scoped_caches() -> None:
    """Free the fit-scoped caches (GPU operand tables, F-order CMI matrix copy, memmap dumps); each rebuilds on the next miss, so this is always safe."""
    failures = []
    for mod_name, fn_name in _RELEASERS:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        try:
            getattr(mod, fn_name)()
        except Exception as exc:
            logger.debug("fit-end release %s.%s failed", mod_name, fn_name, exc_info=True)
            failures.append(f"{mod_name}.{fn_name} ({type(exc).__name__}: {exc})")
    try:
        _free_cached_device_blocks()
    except Exception as exc:
        logger.debug("fit-end release of cached device blocks failed", exc_info=True)
        failures.append(f"cached device blocks ({type(exc).__name__}: {exc})")
    if failures:
        logger.warning("fit-end release failed for %s; those caches stay resident", "; ".join(failures))
