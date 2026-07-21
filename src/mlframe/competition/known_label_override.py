"""Prediction post-processing overrides using external/recovered ground truth.

COMPETITION/EXPLORATORY USE ONLY -- NOT FOR PRODUCTION.

This module implements two closely related "override predictions using knowledge
that isn't causally available at scoring time" patterns documented in
``MLFRAME_IDEAS_competitions.md``:

- ``monotonic_entity_override`` -- "Causal 'future-fraud implies past-fraud'
  post-processing override": if an entity (e.g. a card/account) is known to be
  positive (fraud) at ANY point, override ALL of that entity's predictions
  (past and future rows alike) to reflect the known-positive label, exploiting
  domain monotonicity ("once true, always true" for that entity). This is only
  valid in offline batch-scoring settings where the full history of an entity
  is available at scoring time (e.g. a Kaggle test set scored all at once) --
  in real production, predictions already made in the past have typically
  already been acted upon and cannot be retroactively rewritten, and even
  where a backfill process exists it is a separate, explicitly-audited
  process, never a silent inference-time override.

- ``known_label_override`` -- "Correct-known-label override for high-confidence
  recovered ground truth": given a partial map of high-confidence recovered/
  known labels (e.g. from record-linkage/dedup/leaderboard-probing), override
  predictions only in the direction that reduces asymmetric loss risk (only
  push toward the specified rarer/positive class direction, never toward the
  majority/negative direction). The one-sided-safety rationale: labels
  recovered via exact-identity matching are typically near-certain when they
  assert "this IS the rare/positive class" but much noisier (or entirely
  absent evidence) when they would assert "this is NOT the rare/positive
  class" -- silently trusting a recovered label to flip a prediction toward
  the negative/majority class risks introducing false negatives that are
  costly under class-imbalanced asymmetric loss, whereas flipping toward the
  positive/minority class on high-confidence recovered evidence only ever
  recovers true positives the model missed.

Neither function is safe or meaningful in a real production system: both rely
on knowledge (future entity outcomes, externally recovered "ground truth")
that is either causally unavailable at prediction time or obtained through
competition-specific exploits (LB probing, cross-set record linkage) with no
production analog. See ``mlframe.competition`` package docstring.
"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np

__all__ = [
    "monotonic_entity_override",
    "known_label_override",
]


def monotonic_entity_override(
    preds: np.ndarray,
    entity_ids: np.ndarray,
    known_positive_entity_ids: set,
    *,
    positive_value: float = 1.0,
) -> np.ndarray:
    """Override ALL rows of any entity known to be positive to the positive value.

    COMPETITION/EXPLORATORY USE ONLY -- see module docstring. Implements the
    "future-fraud implies past-fraud" domain-monotonicity post-processing rule:
    once an entity is known positive (at any row, past or future relative to
    any other row of that entity), every row belonging to that entity is
    overridden to ``positive_value``. This is a pure post-processing step
    applied to already-computed predictions -- it must never be turned into a
    training-time FEATURE (the source writeup explicitly notes a
    stacking-with-lag-feature version of this idea caused CV/LB divergence and
    was abandoned), and it is only valid when the full set of an entity's rows
    (including "future" ones) is available at scoring time, e.g. offline
    batch-scoring competition test sets -- never in real-time production
    inference, where future rows for an entity do not exist yet.

    Args:
        preds: 1-D array of model predictions (probabilities or scores), one per row.
        entity_ids: 1-D array of entity identifiers, one per row, aligned with ``preds``.
        known_positive_entity_ids: set of entity ids known to be positive at least once.
        positive_value: the value written into all rows of a known-positive entity.

    Returns:
        A new 1-D array (input ``preds`` is not mutated) with every row belonging
        to a known-positive entity set to ``positive_value``, all other rows unchanged.
    """
    preds_arr: np.ndarray = np.asarray(preds, dtype=float)
    entity_ids = np.asarray(entity_ids)
    if entity_ids.shape[0] != preds_arr.shape[0]:
        raise ValueError("preds and entity_ids must have the same length")

    out: np.ndarray = preds_arr.copy()
    if not known_positive_entity_ids:
        return out

    known_positive_arr = np.asarray(list(known_positive_entity_ids))
    mask = np.isin(entity_ids, known_positive_arr)
    out[mask] = positive_value
    return out


def known_label_override(
    preds: np.ndarray,
    known_label_map: Mapping[int, float],
    *,
    asymmetric_safe_direction: Literal["positive", "negative"] = "positive",
    positive_value: float = 1.0,
    negative_value: float = 0.0,
) -> np.ndarray:
    """Override predictions with recovered known labels, in the safe direction only.

    COMPETITION/EXPLORATORY USE ONLY -- see module docstring. Implements the
    "correct-known-label override for high-confidence recovered ground truth"
    pattern: given a partial map ``{row_index: recovered_label}`` of high-
    confidence recovered/known labels, apply the override ONLY when it pushes
    the prediction toward ``asymmetric_safe_direction`` (default ``"positive"``,
    i.e. the rarer/minority class in the source fraud-detection writeup). Rows
    whose recovered label points toward the OTHER (majority/negative)
    direction are left untouched, because recovered labels are typically
    near-certain evidence for "this IS the rare class" (derived from exact-
    identity matches) but much weaker/absent evidence for "this is NOT the
    rare class" -- asymmetric loss (false negatives on the rare class being
    far costlier than false positives) makes one-directional overriding the
    only safe use of such recovered labels. Applying the override
    bidirectionally (also flipping toward the majority/negative class) can
    actively HURT the target metric whenever some of the "negative-direction"
    recovered labels are wrong -- see the companion biz_value test for a
    concrete demonstration.

    Args:
        preds: 1-D array of model predictions (probabilities or scores), one per row.
        known_label_map: mapping from row index (into ``preds``) to a recovered
            binary label (``positive_value``-like or ``negative_value``-like; any
            value closer to ``positive_value`` is treated as recovered-positive).
        asymmetric_safe_direction: which direction is safe to override toward.
            ``"positive"`` only overrides rows where the recovered label is
            positive (and current pred isn't already >= positive threshold);
            ``"negative"`` is the mirror case for domains where the negative
            class is the asymmetrically-costly/rare one.
        positive_value: value written when overriding toward the positive direction.
        negative_value: value written when overriding toward the negative direction.

    Returns:
        A new 1-D array (input ``preds`` is not mutated) with only the
        safe-direction recovered labels applied.
    """
    preds_arr: np.ndarray = np.asarray(preds, dtype=float)
    out: np.ndarray = preds_arr.copy()

    for idx, recovered_label in known_label_map.items():
        if idx < 0 or idx >= out.shape[0]:
            raise IndexError(f"known_label_map row index {idx} out of bounds for preds of length {out.shape[0]}")
        # "Closer to positive_value" (per this docstring's own contract), NOT `>= midpoint` -- the
        # midpoint comparison silently assumed an ascending negative_value < positive_value convention
        # and applied every override in the OPPOSITE direction for a caller using a reversed scale
        # (e.g. positive_value=0.0, negative_value=1.0). `<=` (not `<`) preserves the exact
        # tie-breaking-toward-positive behavior the old `>= midpoint` formula had at the midpoint itself.
        is_recovered_positive = abs(recovered_label - positive_value) <= abs(recovered_label - negative_value)
        if asymmetric_safe_direction == "positive" and is_recovered_positive:
            out[idx] = positive_value
        elif asymmetric_safe_direction == "negative" and not is_recovered_positive:
            out[idx] = negative_value
        # Otherwise: recovered label points toward the unsafe direction -- skip it.

    return out
