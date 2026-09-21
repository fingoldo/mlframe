"""Target-type policy for which ensemble flavours may be built.

Rank-fusion flavours (``RANK_FUSION_METHODS``: ``rrf``, ``rank_average``) emit a normalised rank score in [0, 1], not a value on the
target's scale:

- regression / multi-target / quantile: the blend's output has mean ~0.5 whatever the target mean is, so every error metric is garbage
  (R2 in the hundreds negative) and the flavour can only look "best" by accident on targets whose labels happen to sit near 0..1;
- binary / multiclass / multilabel classification: a rank blend has a uniform marginal, it is not a probability, and nothing downstream maps
  it back to one (``combine_probs`` returns the raw rank score), so log-loss, Brier, calibration and decision thresholds are all invalid.

Only learning-to-rank targets consume per-row scores whose ordering is the product, so rank fusion is kept there alone. ``score_ensemble``
applies this filter in one place, which means a dropped flavour is never built and therefore can never be picked by the ensemble chooser.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .base import RANK_FUSION_METHODS, SIMPLE_ENSEMBLING_METHODS

logger = logging.getLogger("mlframe.models.ensembling")


def _target_type_key(target_type: Any) -> Optional[str]:
    """Normalise a ``TargetTypes`` member / plain string to its lower-case value (None when unknown)."""
    if target_type is None:
        return None
    _v = getattr(target_type, "value", target_type)
    return str(_v).strip().lower() or None


def rank_fusion_allowed(target_type: Any) -> bool:
    """True only for learning-to-rank targets, where a rank score is a valid prediction."""
    return _target_type_key(target_type) == "learning_to_rank"


def filter_flavours_for_target_type(ensembling_methods: Any, target_type: Any, is_regression: bool, verbose: bool = True) -> Any:
    """Drop ensemble flavours whose output is not a valid prediction for ``target_type``.

    With a known ``target_type`` the rank-fusion flavours are dropped for everything except learning-to-rank. Without one (direct
    ``score_ensemble`` callers) the members' kind is the only signal: rank fusion is dropped for regression-like members (no probs) and an
    explicit request is honoured for classifier-like members, matching the historical contract of direct callers. Logs once at INFO.
    """
    if not ensembling_methods:
        return ensembling_methods
    _methods = list(ensembling_methods)
    _key = _target_type_key(target_type)
    if _key is not None:
        if rank_fusion_allowed(_key):
            return _methods
        _reason = f"target_type={_key}: rank-fusion output is a normalised rank in [0, 1], not a prediction on the target's scale / a probability"
    elif is_regression:
        _reason = "regression-like members: rank-fusion output is a normalised rank in [0, 1], not a prediction on the target's scale"
    else:
        return _methods
    _kept = [m for m in _methods if m not in RANK_FUSION_METHODS]
    _dropped = [m for m in _methods if m in RANK_FUSION_METHODS]
    if _dropped and verbose:
        logger.info("[ensemble] dropping ensemble flavour(s) %s (%s); kept %s.", _dropped, _reason, _kept)
    return _kept


def resolve_flavours_for_target_type(ensembling_methods: Any, target_type: Any, is_regression: bool, verbose: bool = True) -> Any:
    """:func:`filter_flavours_for_target_type`, falling back to the simple flavours only when it removed EVERY requested one.

    A request made only of rank-fusion flavours for a non-ranking target leaves nothing valid, so the simple flavours stand in. An
    explicitly empty request is kept empty: it means "no ensembles", and must not be rewritten into the full simple set.
    """
    kept = filter_flavours_for_target_type(ensembling_methods, target_type, is_regression, verbose)
    if ensembling_methods and not kept:
        return list(SIMPLE_ENSEMBLING_METHODS)
    return kept
