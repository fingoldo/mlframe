"""The offline NVRTC -> ptxas engine lives in py-ci-shared (``py_ci_shared.cuda_offline_report``); this module re-exports the names the offline suite of this package uses, so its tests import from one place.

What stays in this package is the data the engine is run on: the kernel sources and their recipes (``_offline_kernel_sources.py``), the spill exemptions with their numbers and the architectures to check
(``test_cuda_kernels_offline_per_arch.py``).
"""

from __future__ import annotations

from py_ci_shared.cuda_offline_report import (  # noqa: F401
    ARCH_LIMITS,
    REGISTER_FILE,
    EntryReport,
    blocks_per_sm,
    build_ptx_and_report,
    entry_reports,
    find_ptxas,
    launch_bounds_of,
    launch_bounds_violations,
    local_device_limits,
    markdown,
    occupancy,
    required_arch,
)
