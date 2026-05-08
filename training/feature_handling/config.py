"""
Sub-configs + top-level :class:`FeatureHandlingConfig`.

Final structure after the round-2 UX audit nested the previously-flat
25-field surface into 5 sub-configs (round-2 U-R2-1: cognitive load
drops drastically; user readily-confirmed). Top-level FHC now exposes
~8 fields; the rest live under ``cache``, ``memory``, ``pricing``,
``logging``, ``repro``, ``text_detection``.

Phase A scope. The fields are wired structurally (validated, default-
factories, auto-derive logic for memory budgets, describe stub) but
the deep consumers in phases B-G read them by attribute -- no
behaviour change in the suite call until phase D wires the cache
layer.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from pydantic import ConfigDict, Field, model_validator

from mlframe.training.configs import BaseConfig
from mlframe.training.feature_handling.compat import (
    validate_fhc_handlers,
)
from mlframe.training.feature_handling.handlers import (
    CatHandlerSpec,
    TextHandlerSpec,
)
from mlframe.training.feature_handling.system import detect_memory_limit_bytes

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd  # noqa: F401
    import polars as pl  # noqa: F401


# =====================================================================
# Sub-configs
# =====================================================================


class AutoDeriveConfig(BaseConfig):
    """Knobs for the auto-derived memory / cache budgets in
    :class:`MemoryConfig`. Defaults give sane budgets across machines
    from 8 GB laptops to 128 GB workstations.

    Examples (with default 0.7 / 0.3 / 0.1 fractions, 2 GB keep-free min):
        8 GB total  -> budget=5.6 cache=1.7 keep_free=2.0 (capped)
        16 GB total -> budget=11.2 cache=3.4 keep_free=2.0 (capped)
        32 GB total -> budget=22.4 cache=6.7 keep_free=3.2
        64 GB total -> budget=44.8 cache=13.4 keep_free=6.4
    """
    model_config = ConfigDict(extra="forbid")
    memory_budget_fraction: float = Field(default=0.7, gt=0.0, le=1.0)
    cache_ram_fraction: float = Field(default=0.3, gt=0.0, le=1.0)
    cache_keep_free_fraction: float = Field(default=0.1, gt=0.0, le=1.0)
    cache_keep_free_min_gb: float = Field(default=2.0, ge=0.0)


class MemoryConfig(BaseConfig):
    """Memory budget + pressure watermark.

    All ``Optional[float]`` budgets are auto-derived from
    :func:`mlframe.training.feature_handling.system.detect_memory_limit_bytes`
    at FHC construction (cgroup-aware: container-safe). Pass an
    explicit float to override.
    """
    model_config = ConfigDict(extra="forbid")
    budget_gb: Optional[float] = None
    pressure_watermark_pct: float = Field(default=85.0, ge=0.0, le=100.0)
    auto_derive: AutoDeriveConfig = Field(default_factory=AutoDeriveConfig)


class CacheConfig(BaseConfig):
    """All cache mechanics: persistence, RAM/disk budgets, eviction,
    prefetching, namespace.

    ``persistence="off"`` is the default for solo greenfield (RAM-only,
    no disk roundtrip cost). Opt in to ``"auto"`` / ``"read_write"``
    for long-running prod suites where the 10+ minute transformer
    inference cost amortises over re-runs.
    """
    model_config = ConfigDict(extra="forbid")
    persistence: Literal["off", "auto", "read_only", "read_write"] = "off"
    namespace: str = "default"  # round-3 F4: A/B testing seam
    dir: Optional[str] = None
    dataset_id: Optional[str] = None  # bypass content fingerprinting (round-3 user feedback)

    # In-memory tier (auto-derived if None)
    ram_max_gb: Optional[float] = None
    ram_reserve_gb: Optional[float] = None

    # Disk tier (only relevant when persistence != "off")
    disk_evict_when_free_below_gb: float = Field(default=50.0, ge=0.0)
    disk_min_free_gb: float = Field(default=5.0, ge=0.0)
    eviction_strategy: Literal["lru", "lfu", "size_weighted"] = "size_weighted"
    eviction_async: bool = True

    # Prefetching
    prefetch_enabled: bool = True
    prefetch_device: Literal["auto", "cpu", "cuda"] = "auto"
    prefetch_vram_safety_factor: float = Field(default=2.0, ge=1.0)

    # Provider lifecycle
    keep_n_providers: Union[int, Literal["auto"]] = "auto"

    # Per-fold scaling
    max_per_column_entries: int = Field(default=1000, gt=0)

    # Backend abstraction
    backend: Literal["local_disk"] = "local_disk"


class PricingConfig(BaseConfig):
    """Cost gates for paid embedding providers (OpenAI / Cohere /
    Voyage / Jina). ``cap_usd=None`` means no gate; for self-hosted
    HF / ONNX / FastText this never activates.
    """
    model_config = ConfigDict(extra="forbid")
    cap_usd: Optional[float] = None
    warn_above_usd: float = Field(default=1.0, ge=0.0)


class LoggingConfig(BaseConfig):
    """Logging discipline for the feature-handling subsystem."""
    model_config = ConfigDict(extra="forbid")
    verbose: bool = False  # default log.debug; True -> log.info
    redact_column_names: List[str] = Field(default_factory=list)


class ReproConfig(BaseConfig):
    """Reproducibility safeguards. All False/None defaults -- this is
    opt-in; turning on incurs a perf cost (deterministic_torch is
    10-25%) so default off for dev workflow, opt-in for CI gate
    parity tests.
    """
    model_config = ConfigDict(extra="forbid")
    deterministic_torch: bool = False
    pinned_revision: bool = True  # zero-cost; pinned SHA in default provider
    langdetect_seed: int = 0
    pinned_svd_solver_params: bool = True
    forbid_nonatomic_fs: bool = False
    deterministic_eviction: bool = False  # CI/audit mode only


class TextDetectionConfig(BaseConfig):
    """Multi-criteria text-vs-categorical heuristic with anti-UUID
    guards. Replaces the pre-2026 single-trigger
    ``cat_text_cardinality_threshold > 300`` (round-2 A10).
    """
    model_config = ConfigDict(extra="forbid")

    # Triggers (any -> text)
    definite_text_mean_chars: int = Field(default=100, ge=1)
    text_min_mean_chars: int = Field(default=30, ge=1)
    text_min_mean_tokens: float = Field(default=4.0, gt=0.0)
    text_min_unique_ratio: float = Field(default=0.95, ge=0.0, le=1.0)
    text_min_cardinality: int = Field(default=300, ge=1)

    # Anti-UUID guards (round-3 R2-21: tightened to 4.5 to keep UUID-v4
    # at 4.04 below the threshold).
    min_alphabet_entropy: float = Field(default=4.5, ge=0.0)
    min_mean_tokens_for_text: float = Field(default=2.0, ge=0.0)

    # User overrides
    skip_columns: List[str] = Field(default_factory=list)
    explicit_text_columns: List[str] = Field(default_factory=list)
    explicit_categorical_columns: List[str] = Field(default_factory=list)

    sample_size_for_stats: int = Field(default=50_000, gt=0)

    # Round-3 user-confirmation: respect_explicit_categorical_dtype
    # default flipped to True for greenfield -- categorical/Enum dtype
    # signals user intent and beats the cardinality heuristic.
    respect_explicit_categorical_dtype: bool = True


# =====================================================================
# ModelHandlingOverride
# =====================================================================


class ModelHandlingOverride(BaseConfig):
    """Per-model override for a slice of the default cat/text handler
    chains. Phase A: scalar shorthand acceptance lives at the FHC
    level; this dataclass is the canonical-form structure.

    `text_append` / `cat_append` semantics (round-3 U-R2-4): when set,
    APPEND to the FHC-level default rather than replacing. Common
    pattern: "all models get tfidf, plus mlp gets a learnable_text
    block on top".
    """
    model_config = ConfigDict(extra="forbid")
    cat: Optional[List[CatHandlerSpec]] = None
    text: Optional[List[TextHandlerSpec]] = None
    cat_append: Optional[List[CatHandlerSpec]] = None
    text_append: Optional[List[TextHandlerSpec]] = None


# =====================================================================
# FeatureHandlingConfig (top-level)
# =====================================================================


class FeatureHandlingConfig(BaseConfig):
    """Single source of truth for per-model categorical / text /
    numeric feature handling.

    Top-level shape (~8 fields). Tuning lives in nested sub-configs:

      * :attr:`text_detection` -- auto-detection heuristic.
      * :attr:`cache` -- persistence / RAM / disk / prefetching.
      * :attr:`memory` -- budget + watermark + auto-derive knobs.
      * :attr:`pricing` -- paid-provider cost gates.
      * :attr:`logging` -- verbosity + PII redaction.
      * :attr:`repro` -- reproducibility safeguards (opt-in).

    Greenfield -- no aliases, no deprecation. Phase A wires the
    structure; phases B-G consume it.

    Examples
    --------
    >>> # zero-config: defaults across the board
    >>> fhc = FeatureHandlingConfig()
    >>> # one-line preset for the 80% case
    >>> from mlframe.training.feature_handling.presets import tfidf_only
    >>> fhc2 = tfidf_only(max_features=10000)
    >>> # power-user fine tuning
    >>> fhc3 = FeatureHandlingConfig(
    ...     per_model={"mlp": ModelHandlingOverride(text=[...])},
    ...     cache=CacheConfig(persistence="auto"),
    ... )
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # === Defaults: List | Spec | str shorthand
    default_cat: Union[List[CatHandlerSpec], CatHandlerSpec, str, None] = None
    default_text: Union[List[TextHandlerSpec], TextHandlerSpec, str, None] = None

    # === Per-model / per-target overrides
    per_model: Dict[str, ModelHandlingOverride] = Field(default_factory=dict)
    per_target: Dict[str, "FeatureHandlingConfig"] = Field(default_factory=dict)  # round-3 F10 reserved

    # === Mode (round-3 F13)
    mode: Literal["fit", "predict"] = "fit"

    # === Provider config
    default_text_provider: Optional[Any] = None  # EmbeddingProvider -- phase A2

    # auto-locale: when "off" the default_text_provider falls back to
    # ``intfloat/multilingual-e5-small`` (more general, per user
    # confirmation 2026-05-09). "always" probes via langdetect on
    # every FHC; "fallback_only" probes only when default unset.
    auto_locale_detection: Literal["off", "always", "fallback_only"] = "fallback_only"
    auto_locale_sample_size: int = Field(default=1000, gt=0)
    auto_locale_english_threshold: float = Field(default=0.95, ge=0.0, le=1.0)

    # === Sub-configs
    text_detection: TextDetectionConfig = Field(default_factory=TextDetectionConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)  # noqa: A003
    repro: ReproConfig = Field(default_factory=ReproConfig)

    # === Auto-derived (filled by validator at construction)
    _resolved_memory_budget_gb: Optional[float] = None
    _resolved_cache_ram_max_gb: Optional[float] = None
    _resolved_cache_ram_reserve_gb: Optional[float] = None

    # ------------------------------------------------------------------
    # Auto-derivation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _derive_memory_budgets(self) -> "FeatureHandlingConfig":
        """Resolve ``memory.budget_gb`` / ``cache.ram_max_gb`` /
        ``cache.ram_reserve_gb`` from system probe when None.

        cgroup-aware via ``detect_memory_limit_bytes`` -- containers
        get container limits, not host RAM (round-3 R2-1).
        """
        total_bytes = detect_memory_limit_bytes()
        total_gb = total_bytes / 1e9
        ad = self.memory.auto_derive

        # memory_budget_gb
        if self.memory.budget_gb is None:
            self.memory.budget_gb = total_gb * ad.memory_budget_fraction
        budget_gb = self.memory.budget_gb

        # cache.ram_max_gb derives from memory_budget
        cache_ram_was_explicit = self.cache.ram_max_gb is not None
        if self.cache.ram_max_gb is None:
            self.cache.ram_max_gb = budget_gb * ad.cache_ram_fraction

        # cache.ram_reserve_gb -- absolute floor, not fraction-of-budget
        if self.cache.ram_reserve_gb is None:
            self.cache.ram_reserve_gb = max(
                ad.cache_keep_free_min_gb,
                total_gb * ad.cache_keep_free_fraction,
            )

        # Sanity invariant. Two policies:
        #   * If both ram_max and ram_reserve are auto-derived, cap
        #     ram_max so cap+reserve fits in total RAM. Common case
        #     when user sets ``memory.budget_gb`` larger than total
        #     (legitimate -- they may want headroom for swap/etc).
        #   * If ram_max was EXPLICIT and exceeds total - reserve,
        #     raise. The user told us a number we can't honour.
        max_allowed = total_gb - self.cache.ram_reserve_gb
        if self.cache.ram_max_gb > max_allowed:
            if cache_ram_was_explicit:
                raise ValueError(
                    f"FeatureHandlingConfig: cache.ram_max_gb={self.cache.ram_max_gb:.2f} "
                    f"exceeds total-{self.cache.ram_reserve_gb:.2f} "
                    f"= {max_allowed:.2f} GB on a {total_gb:.1f} GB system. "
                    f"Lower cache.ram_max_gb or cache.ram_reserve_gb."
                )
            # Auto-derived: cap silently with a log line so tuning is visible.
            new_cap = max(0.0, max_allowed)
            logger.info(
                "[fhc] auto-derived cache.ram_max_gb=%.2f exceeds total-%.2f=%.2f; "
                "capping to %.2f to fit %.1f GB system",
                self.cache.ram_max_gb, self.cache.ram_reserve_gb, max_allowed,
                new_cap, total_gb,
            )
            self.cache.ram_max_gb = new_cap

        # Hard invariant after cap: cache_ram must be > 0 on any
        # reasonable machine. Below this we genuinely can't run --
        # tiny 1 GB box + 2 GB reserve_min would have ram_max < 0,
        # caught here.
        if self.cache.ram_max_gb <= 0.0:
            raise ValueError(
                f"FeatureHandlingConfig: invalid memory budget on a "
                f"{total_gb:.1f} GB system. After auto-derive, "
                f"cache.ram_max_gb={self.cache.ram_max_gb:.2f} (must be > 0). "
                f"Lower cache.ram_reserve_gb or auto_cache_keep_free_min_gb."
            )

        # Stash resolved values for describe() / repr().
        self._resolved_memory_budget_gb = budget_gb
        self._resolved_cache_ram_max_gb = self.cache.ram_max_gb
        self._resolved_cache_ram_reserve_gb = self.cache.ram_reserve_gb

        return self

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def resolved(self) -> Dict[str, Any]:
        """Auto-derived numeric values, exposed for log/repr/describe."""
        return {
            "memory_budget_gb": self._resolved_memory_budget_gb,
            "cache_ram_max_gb": self._resolved_cache_ram_max_gb,
            "cache_ram_reserve_gb": self._resolved_cache_ram_reserve_gb,
        }

    def describe(
        self,
        df: Optional[Any] = None,
        *,
        short: bool = True,
    ) -> Dict[str, Any]:
        """Return a printable resolution plan.

        Phase A returns a structural summary only -- the per-column
        auto-text-detection decisions land in phase K when
        ``_detect_text_columns_polars`` is wired through.
        """
        base = {
            "mode": self.mode,
            "default_text_provider": self.default_text_provider,
            "auto_locale_detection": self.auto_locale_detection,
            "per_model_keys": list(self.per_model.keys()),
            "per_target_keys": list(self.per_target.keys()),
            "resolved_memory_gb": self.resolved,
            "cache_persistence": self.cache.persistence,
            "cache_namespace": self.cache.namespace,
        }
        if not short:
            base["text_detection"] = self.text_detection.model_dump()
            base["cache"] = self.cache.model_dump()
            base["memory"] = self.memory.model_dump()
            base["repro"] = self.repro.model_dump()
        return base

    def __repr__(self) -> str:
        # Surface auto-derived values in repr so they're visible at the
        # debugger / log line, not buried (round-3 U-R2-5).
        return (
            f"FeatureHandlingConfig(mode={self.mode!r}, "
            f"per_model={list(self.per_model)}, "
            f"cache.persistence={self.cache.persistence!r}, "
            f"resolved={self.resolved})"
        )

    # ------------------------------------------------------------------
    # Validation against an active model list
    # ------------------------------------------------------------------

    def validate_against_models(self, mlframe_models: List[str]) -> None:
        """Walk per_model + defaults, validate each (model, axis,
        method) tuple against the compat matrix. Raises a single
        combined ``ValueError`` listing every mismatch (round-3
        U-R2-29).

        Called by ``train_mlframe_models_suite`` immediately after
        FHC construction with the active ``mlframe_models`` list.
        FHC alone can't run this -- the list isn't visible at
        construction time.
        """
        # Build per-model spec lists, applying overrides + defaults.
        text_per_model = {m: self._effective_text_specs(m) for m in mlframe_models}
        cat_per_model = {m: self._effective_cat_specs(m) for m in mlframe_models}
        validate_fhc_handlers(
            text_specs_per_model=text_per_model,
            cat_specs_per_model=cat_per_model,
            active_models=mlframe_models,
        )

    # ------------------------------------------------------------------
    # Effective handler chain assembly (defaults + overrides)
    # ------------------------------------------------------------------

    def _coerce_to_spec_list(
        self,
        value: Any,
        spec_cls: type,
    ) -> List[Any]:
        """Normalise the ``Union[List, Spec, str, None]`` shorthand
        forms to a List[spec_cls]."""
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        if isinstance(value, spec_cls):
            return [value]
        # String shorthand: treat as method name with NoParams default.
        if isinstance(value, str):
            from mlframe.training.feature_handling.handlers import NoParams
            return [spec_cls(method=value, params=NoParams(kind=value))]
        raise TypeError(
            f"unsupported value type for {spec_cls.__name__}: {type(value).__name__}"
        )

    def _effective_text_specs(self, model_kind: str) -> List[TextHandlerSpec]:
        """Resolve final text-handler chain for a model: override
        replaces, override_append extends defaults."""
        defaults = self._coerce_to_spec_list(self.default_text, TextHandlerSpec)
        override = self.per_model.get(model_kind)
        if override is None:
            return defaults
        if override.text is not None:
            base = list(override.text)
        else:
            base = defaults
        if override.text_append:
            base = base + list(override.text_append)
        return base

    def _effective_cat_specs(self, model_kind: str) -> List[CatHandlerSpec]:
        """Resolve final cat-handler chain for a model."""
        defaults = self._coerce_to_spec_list(self.default_cat, CatHandlerSpec)
        override = self.per_model.get(model_kind)
        if override is None:
            return defaults
        if override.cat is not None:
            base = list(override.cat)
        else:
            base = defaults
        if override.cat_append:
            base = base + list(override.cat_append)
        return base


# Resolve forward ref for per_target self-recursion.
FeatureHandlingConfig.model_rebuild()


__all__ = [
    "AutoDeriveConfig",
    "MemoryConfig",
    "CacheConfig",
    "PricingConfig",
    "LoggingConfig",
    "ReproConfig",
    "TextDetectionConfig",
    "ModelHandlingOverride",
    "FeatureHandlingConfig",
]
