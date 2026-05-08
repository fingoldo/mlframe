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
from mlframe.training.feature_handling.compat import (
    register_model_axis_support,
    validate_handler_for_model,
    validate_fhc_handlers,
)
from mlframe.training.feature_handling.config import (
    AutoDeriveConfig,
    CacheConfig,
    FeatureHandlingConfig,
    LoggingConfig,
    MemoryConfig,
    ModelHandlingOverride,
    PricingConfig,
    ReproConfig,
    TextDetectionConfig,
)
from mlframe.training.feature_handling.handlers import (
    CatHandlerSpec,
    CustomParams,
    FrozenEmbeddingParams,
    HashingParams,
    LearnableEmbeddingParams,
    NoParams,
    TargetEncodeParams,
    TextHandlerSpec,
    TfidfParams,
)
from mlframe.training.feature_handling.locking import (
    PIDAwareFileLock,
    StaleLockReclaimed,
)
from mlframe.training.feature_handling.presets import (
    cb_native_only,
    embedding_only,
    tfidf_only,
)
from mlframe.training.feature_handling.providers import (
    EmbeddingProvider,
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
    # compat
    "register_model_axis_support",
    "validate_handler_for_model",
    "validate_fhc_handlers",
    # config (sub + top-level)
    "AutoDeriveConfig",
    "CacheConfig",
    "FeatureHandlingConfig",
    "LoggingConfig",
    "MemoryConfig",
    "ModelHandlingOverride",
    "PricingConfig",
    "ReproConfig",
    "TextDetectionConfig",
    # handlers
    "CatHandlerSpec",
    "CustomParams",
    "FrozenEmbeddingParams",
    "HashingParams",
    "LearnableEmbeddingParams",
    "NoParams",
    "TargetEncodeParams",
    "TextHandlerSpec",
    "TfidfParams",
    # locking
    "PIDAwareFileLock",
    "StaleLockReclaimed",
    # presets
    "cb_native_only",
    "embedding_only",
    "tfidf_only",
    # providers
    "EmbeddingProvider",
    # system
    "CudaErrorClass",
    "classify_cuda_error",
    "detect_memory_limit_bytes",
    "long_path_safe",
]
