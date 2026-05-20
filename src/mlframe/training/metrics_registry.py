"""Pluggable metrics registry for target-type-aware reporting.

Replaces hardcoded metric calls in ``report_probabilistic_model_perf`` with
a registry indexed by ``TargetTypes``. Built-in registrations for
multilabel (``hamming_loss``, ``subset_accuracy``, ``jaccard_score_multilabel``)
land at import time.

Extensibility
-------------
External callers can register domain-specific metrics without touching
``evaluation.py``:

    from mlframe.training.metrics_registry import register_metric
    from mlframe.training.configs import TargetTypes

    def my_custom_multilabel_metric(y_true, probs_NK, preds_NK):
        return some_score

    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION,
        "my_custom_metric",
        my_custom_multilabel_metric,
    )

``register_metric`` idempotent — re-registering a name overwrites the
previous impl (useful for A/B comparison or test stubs).

The metric callable receives ``(y_true, probs_NK, preds_NK)`` where:
- y_true   : 1-D labels (binary/multiclass) OR 2-D indicator (multilabel)
- probs_NK : canonicalised (N, K) probability matrix
- preds_NK : decision-rule output (1-D argmax or 2-D binary threshold)

Returns any value with a ``__format__`` method (typically float).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, Optional, Tuple, Any

from .configs import TargetTypes


@dataclass(frozen=True)
class MetricSpec:
    """Structured metadata for a registered metric.

    Fields:
    - fn: the metric callable ``(y_true, probs_NK, preds_NK) -> value``.
    - higher_is_better: True when larger values mean better performance
      (e.g. accuracy, AUC), False for losses (e.g. log_loss, hamming).
    - description: optional human-readable blurb surfaced in introspection.
    """
    fn: Callable
    higher_is_better: bool = True
    description: str = ""


_REGISTRY: dict[TargetTypes, dict[str, MetricSpec]] = {}


def register_metric(
    target_type: TargetTypes,
    name: str,
    fn: Callable,
    *,
    higher_is_better: bool = True,
    description: str = "",
) -> None:
    """Register a metric function for a target type.

    The optional ``higher_is_better`` flag lets downstream callers (lift-pct,
    strongest-pick, leaderboards) interpret the metric direction without
    hard-coding string lookups against the metric name. The ``description``
    surfaces in :func:`list_registered_specs` for help-text rendering.

    Idempotent: re-registering the same name overwrites.
    """
    _REGISTRY.setdefault(target_type, {})[name] = MetricSpec(
        fn=fn, higher_is_better=bool(higher_is_better), description=description,
    )


def unregister_metric(target_type: TargetTypes, name: str) -> None:
    """Remove a registered metric. No-op if not registered."""
    if target_type in _REGISTRY:
        _REGISTRY[target_type].pop(name, None)


def iter_extra_metrics(
    target_type: TargetTypes, y_true, probs_NK, preds_NK
) -> Iterator[tuple[str, Any]]:
    """Yield (name, value) for every registered metric on this target type.

    Narrow exception catch: only the documented failure modes for sklearn
    metric callables propagate as recoverable (ValueError on degenerate
    inputs, ZeroDivisionError on empty groups, TypeError on shape
    mismatches). Anything else (KeyboardInterrupt, MemoryError, programming
    bugs in caller-supplied metrics) bubbles up so a real bug is not
    masquerading as "metric not applicable".
    """
    import logging
    logger = logging.getLogger(__name__)

    for name, spec in _REGISTRY.get(target_type, {}).items():
        try:
            value = spec.fn(y_true, probs_NK, preds_NK)
            yield name, value
        except (ValueError, ZeroDivisionError, TypeError, FloatingPointError) as e:
            # WARNING not DEBUG -- silently omitting a metric from the report is a
            # substantive event the operator needs to see. Pre-fix the operator saw
            # the report missing the metric row entirely and concluded the metric
            # was never configured (or the model was fine), not "the metric crashed
            # on degenerate data". Common upstream causes: roc_auc on a single-class
            # slice (val/test became class-degenerate due to outlier_detection +
            # tight aging window), pinball on shape mismatch (quantile preds vs
            # scalar y_true).
            try:
                _n = int(len(y_true)) if y_true is not None else 0
            except TypeError:
                _n = -1  # un-sized iterator etc.
            logger.warning(
                "metric %r failed on target_type=%s n=%d: %s: %s; omitted from report",
                name, target_type, _n, type(e).__name__, e,
            )


def list_registered(target_type: TargetTypes) -> list:
    """Introspection: list registered metric names for a target type."""
    return list(_REGISTRY.get(target_type, {}).keys())


def list_registered_specs(target_type: TargetTypes) -> dict[str, MetricSpec]:
    """Introspection: full {name: MetricSpec} map (direction + description)."""
    return dict(_REGISTRY.get(target_type, {}))


def get_metric_direction(
    target_type: TargetTypes, name: str,
) -> Optional[bool]:
    """Return ``higher_is_better`` for a registered metric, or None if absent."""
    spec = _REGISTRY.get(target_type, {}).get(name)
    return None if spec is None else spec.higher_is_better


# ----------------------------------------------------------------------------
# Built-in registrations — land at import time
# ----------------------------------------------------------------------------


def _register_builtin_multilabel():
    from mlframe.metrics.core import (
        hamming_loss, subset_accuracy, jaccard_score_multilabel,
    )

    def _ham(y_true, probs_NK, preds_NK):
        return hamming_loss(y_true, preds_NK)

    def _sub(y_true, probs_NK, preds_NK):
        return subset_accuracy(y_true, preds_NK)

    def _jac(y_true, probs_NK, preds_NK):
        return jaccard_score_multilabel(y_true, preds_NK)

    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION, "hamming_loss", _ham,
        higher_is_better=False,
        description="Fraction of labels predicted incorrectly per sample (lower is better).",
    )
    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION, "subset_accuracy", _sub,
        higher_is_better=True,
        description="Exact-match accuracy: 1 only when every label is correct.",
    )
    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION, "jaccard_samples", _jac,
        higher_is_better=True,
        description="Per-sample Jaccard (intersection over union) averaged across rows.",
    )


_register_builtin_multilabel()
