"""Second classification report per binary model, at the decision threshold the suite tuned for it.

The per-model report printed while training is at the default 0.5, but the deployed hard labels use the threshold tuned on val
(``_tune_decision_thresholds``), which only exists once every member has its val probabilities. The tuned-threshold block is therefore
emitted at the moment the threshold is stamped, from the probabilities and targets the model already carries -- nothing is refit or re-tuned.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("mlframe.training.core._phase_train_one_target")

_SPLIT_NOTES = {
    "val": "optimistic: the threshold was tuned on this split",
    "test": "honest: the threshold was tuned on val only",
}


def _positive_scores(probs: Any) -> np.ndarray:
    """Positive-class score vector of a binary probability output, (n,) or (n, 2+)."""
    arr = np.asarray(probs)
    return arr[:, 1] if arr.ndim == 2 and arr.shape[1] >= 2 else arr.ravel()


def log_tuned_threshold_report(
    *, label: str, owner: Any, threshold: float, metric: str, fallback_targets: Optional[dict] = None, digits: int = 4,
) -> None:
    """Log one classification report per available split (val, test) with hard labels ``p >= threshold``.

    ``owner`` carries ``{split}_probs`` / ``{split}_target``; ``fallback_targets`` supplies a split's target when the owner has none (an
    ensemble blend holds probabilities only). A split with missing or length-mismatched data is skipped, never guessed.
    """
    from mlframe.metrics.core import format_classification_report

    for split in ("val", "test"):
        probs = getattr(owner, f"{split}_probs", None)
        target = getattr(owner, f"{split}_target", None)
        if target is None and fallback_targets:
            target = fallback_targets.get(split)
        if probs is None or target is None:
            continue
        try:
            y = np.asarray(target).ravel().astype(np.int64)
            score = _positive_scores(probs)
            if score.shape[0] != y.shape[0] or y.shape[0] == 0:
                continue
            pred = (score >= threshold).astype(np.int64)
            text = format_classification_report(y, pred, nclasses=2, digits=digits, zero_division=0)
        except (ValueError, TypeError) as exc:
            logger.debug("tuned-threshold report skipped for %s/%s: %s", label, split, exc)
            continue
        logger.info("CLASSIFICATION REPORT at tuned threshold %.4f (metric=%s; %s) [%s] %s\n%s", threshold, metric, _SPLIT_NOTES[split], split.upper(), label, text)
