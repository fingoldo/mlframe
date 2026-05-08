"""
mlframe feature-handling subsystem.

This package houses the abstractions that drive per-model categorical /
text / numeric handling, embedding providers, the multi-tier cache, and
the polars-ds vs sklearn dispatcher. The full design rationale lives in
``docs/feature_handling_architecture.md``; the comments below give the
two-line tour.

Layered structure
-----------------
- ``axis``        -- ``Axis`` enum, ``HandlerSpec`` ABC, axis registry.
                     The extension seam for image / audio / sequence
                     axes that are not part of v1.
- ``locking``     -- PID-aware filelock helper that detects and reclaims
                     stale locks left by SIGKILL'd peer processes
                     (chaos-audit C5 + C16, 2026 round 3).
- ``system``      -- cgroup-aware memory probe, CUDA error classifier,
                     Windows long-path helper.
- ``cache_backend`` -- ``CacheBackend`` Protocol + ``LocalDiskBackend``
                     impl (the only backend in v1; S3/GCS/NFS slot in
                     here later).

Phase M deliverable. The bigger features (FeatureHandlingConfig top-level
and most sub-configs, providers, multi-handler concat, cache fingerprint
logic) land in subsequent phases per the plan in
``docs/feature_handling_architecture.md`` (forthcoming).
"""

from mlframe.training.feature_handling.axis import (
    Axis,
    HandlerSpec,
    register_handler_spec,
    get_handler_spec_for_axis,
)
from mlframe.training.feature_handling.cache_backend import (
    CacheBackend,
    LocalDiskBackend,
)
from mlframe.training.feature_handling.locking import (
    PIDAwareFileLock,
    StaleLockReclaimed,
)
from mlframe.training.feature_handling.system import (
    CudaErrorClass,
    classify_cuda_error,
    detect_memory_limit_bytes,
    long_path_safe,
)

__all__ = [
    # axis
    "Axis",
    "HandlerSpec",
    "register_handler_spec",
    "get_handler_spec_for_axis",
    # cache_backend
    "CacheBackend",
    "LocalDiskBackend",
    # locking
    "PIDAwareFileLock",
    "StaleLockReclaimed",
    # system
    "CudaErrorClass",
    "classify_cuda_error",
    "detect_memory_limit_bytes",
    "long_path_safe",
]
