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

from typing import Callable, Dict, Iterator, Tuple, Any

from .configs import TargetTypes


_REGISTRY: Dict[TargetTypes, Dict[str, Callable]] = {}


def register_metric(target_type: TargetTypes, name: str, fn: Callable) -> None:
    """Register a metric function for a target type.

    Idempotent — re-registering the same name overwrites.
    """
    _REGISTRY.setdefault(target_type, {})[name] = fn


def unregister_metric(target_type: TargetTypes, name: str) -> None:
    """Remove a registered metric. No-op if not registered."""
    if target_type in _REGISTRY:
        _REGISTRY[target_type].pop(name, None)


def iter_extra_metrics(
    target_type: TargetTypes, y_true, probs_NK, preds_NK
) -> Iterator[Tuple[str, Any]]:
    """Yield (name, value) for every registered metric on this target type.

    Metrics that fail (raise) are silently skipped — report keeps going.
    """
    import logging
    logger = logging.getLogger(__name__)

    for name, fn in _REGISTRY.get(target_type, {}).items():
        try:
            value = fn(y_true, probs_NK, preds_NK)
            yield name, value
        except Exception as e:
            logger.debug(f"metric {name!r} failed: {e}")


def list_registered(target_type: TargetTypes) -> list:
    """Introspection: list registered metric names for a target type."""
    return list(_REGISTRY.get(target_type, {}).keys())


# ----------------------------------------------------------------------------
# Built-in registrations — land at import time
# ----------------------------------------------------------------------------


def _register_builtin_multilabel():
    from mlframe.metrics import (
        hamming_loss, subset_accuracy, jaccard_score_multilabel,
    )

    def _ham(y_true, probs_NK, preds_NK):
        return hamming_loss(y_true, preds_NK)

    def _sub(y_true, probs_NK, preds_NK):
        return subset_accuracy(y_true, preds_NK)

    def _jac(y_true, probs_NK, preds_NK):
        return jaccard_score_multilabel(y_true, preds_NK)

    register_metric(TargetTypes.MULTILABEL_CLASSIFICATION, "hamming_loss", _ham)
    register_metric(TargetTypes.MULTILABEL_CLASSIFICATION, "subset_accuracy", _sub)
    register_metric(TargetTypes.MULTILABEL_CLASSIFICATION, "jaccard_samples", _jac)


_register_builtin_multilabel()
