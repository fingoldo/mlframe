"""Ensembling config carved out of ``_model_configs.py`` to keep the parent below the 1k LOC monolith threshold.

Behaviour preserved bit-for-bit; the parent re-exports ``EnsemblingConfig`` from its bottom so existing imports (``from mlframe.training._model_configs import EnsemblingConfig`` and the canonical ``from mlframe.training.configs import EnsemblingConfig``) keep resolving.
"""
from __future__ import annotations

from ._configs_base import BaseConfig


class EnsemblingConfig(BaseConfig):
    """Configuration for ensembling behaviour, including streaming-vs-legacy
    aggregation choice and quantile-fallback budget.

    Replaces the env-var ``ENSEMBLE_FORCE_LEGACY_MATERIALISATION=1`` knob
    (which is invisible in function signatures, untestable, and global)
    with a structured config. Env var is still honoured as the default
    for one release for back-compat.
    """

    force_legacy: bool = False
    """If True, use the pre-streaming materialised-aggregation path
    (allocates ``(M, N, K)`` tensors). Default False uses streaming Welford."""

    quantile_budget_bytes: int = 500 * 1024 * 1024
    """Skip quantile-bucket aggregation with warn when ``M*N*K*8 > budget``.
    500 MB default. Override per environment to taste."""

    accumulator: str = "welford"
    """Streaming accumulator implementation. ``welford`` is the only
    impl today; ``tdigest`` / ``p2_quantile`` planned (registered via
    ``StreamingAccumulator`` Protocol)."""

    flag_degenerate_conf_subset: bool = True
    """If True, prepend ``[DEGENERATE]`` to the Conf Ensemble model_name
    when the confidence-filtered subset's class balance collapses
    (``min(class_support) / max(class_support) < degenerate_class_ratio``).

    Why: a uniform-quantile confidence filter often keeps only the
    rows the ensemble is most confident about, which on imbalanced data
    means "almost-all-positive" or "almost-all-negative" subsets.
    Metrics computed on that subset are deceptively pristine
    (BR=0.026 %, LL=0.002 - observed in one prod log) and easy to
    misread as headlines. The marker is a one-glance hint that the
    block is reporting on a degenerate slice.

    Binary classification only - for regression there is no class
    balance to check; the flag has no effect."""

    degenerate_class_ratio: float = 0.01
    """Threshold below which a confidence-filtered subset is flagged
    as degenerate. ``0.01`` means a class balance worse than 1:100
    (e.g. 21 negatives vs 81 815 positives, observed in one prod log)
    triggers the marker. Has no effect when
    ``flag_degenerate_conf_subset=False``."""
