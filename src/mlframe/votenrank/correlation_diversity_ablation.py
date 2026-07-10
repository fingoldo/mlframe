"""``diversity_ablation_report``: flag low-correlation-but-lower-accuracy models for a blend ablation.

Source: 1st_mechanisms-of-action-moa-prediction.md -- deliberately kept architecturally diverse models (e.g.
DeepInsight image-CNN on raw features vs tabular NN on engineered features) with unusually low pairwise OOF
correlation (0.52-0.73) vs the "typically good" 0.85-0.95 range, because the extra diversity was more
blend-additive than the accuracy gap would suggest. A naive "keep only the top-K models by individual CV
score" filter would have DROPPED exactly the models that mattered most for the blend.

Reuses ``training.composite.ensemble.stacking.residual_correlation_matrix`` (pure Pearson correlation, no
composite-internal dependency) for the pairwise correlation computation -- the genuinely missing piece is the
ablation itself: for each candidate flagged as "low-correlation-but-lower-accuracy" relative to the current
best single model, measure the actual with-vs-without blend score change, rather than excluding on
single-model rank alone.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from mlframe.training.composite.ensemble.stacking import residual_correlation_matrix


def diversity_ablation_report(
    oof_preds: Dict[str, np.ndarray],
    individual_scores: Dict[str, float],
    y_true: np.ndarray,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    correlation_threshold: float = 0.85,
    higher_score_is_better: bool = True,
) -> list:
    """Flag low-correlation-but-lower-accuracy candidates and measure their actual with-vs-without blend impact.

    Parameters
    ----------
    oof_preds
        ``{model_name: (n_samples,) OOF prediction array}``.
    individual_scores
        ``{model_name: single-model CV score}`` (same keys as ``oof_preds``).
    y_true
        ``(n_samples,)`` ground truth.
    loss_fn
        ``loss_fn(y_true, y_pred) -> float``, LOWER is better (used for the ablation's blend metric, e.g. RMSE
        or log-loss -- independent of whatever metric ``individual_scores`` uses).
    correlation_threshold
        A candidate's max pairwise correlation with any OTHER model must be BELOW this to be flagged
        "low-correlation" (i.e. genuinely diverse, not just a near-duplicate of an existing model).
    higher_score_is_better
        Whether ``individual_scores`` is a higher-is-better metric (AUC, accuracy, ...) or lower-is-better
        (RMSE, log-loss, ...) -- controls which model is treated as the current "best single model" baseline.

    Returns
    -------
    list of dict
        One entry per flagged low-correlation-but-lower-accuracy candidate: ``{"model", "max_correlation",
        "individual_score", "blend_without_candidate_loss", "blend_with_candidate_loss",
        "ablation_improvement"}`` (``ablation_improvement > 0`` means adding the candidate to an equal-weight
        blend of all OTHER models improved the loss -- i.e. the diversity WAS worth it despite the lower
        individual score).
    """
    names = list(oof_preds.keys())
    if len(names) < 2:
        return []

    corr_matrix, corr_names = residual_correlation_matrix(oof_preds)
    assert corr_names == names

    best_model = max(individual_scores, key=lambda k: individual_scores[k]) if higher_score_is_better else min(individual_scores, key=lambda k: individual_scores[k])
    best_score = individual_scores[best_model]

    # Precompute the total sum ONCE: "without candidate" is then total_sum - candidate (O(n_samples)) and
    # "with candidate" (= mean of ALL models, identical for every flagged candidate by construction) is
    # total_sum / n_models -- also computed once. The naive per-candidate list-rebuild-and-mean was O(n_flagged
    # * n_models * n_samples); this is O(n_models * n_samples) total regardless of how many candidates flag.
    n_models = len(names)
    total_sum = np.sum([oof_preds[n] for n in names], axis=0)
    with_pred = total_sum / n_models
    loss_with = float(loss_fn(y_true, with_pred))

    report = []
    for i, name in enumerate(names):
        if name == best_model:
            continue
        is_lower_accuracy = (individual_scores[name] < best_score) if higher_score_is_better else (individual_scores[name] > best_score)
        off_diag = np.delete(corr_matrix[i], i)
        max_corr = float(np.nanmax(np.abs(off_diag))) if off_diag.size > 0 else 0.0
        is_low_correlation = max_corr < correlation_threshold

        if is_lower_accuracy and is_low_correlation:
            without_pred = (total_sum - oof_preds[name]) / (n_models - 1)
            loss_without = float(loss_fn(y_true, without_pred))
            report.append(
                {
                    "model": name,
                    "max_correlation": max_corr,
                    "individual_score": individual_scores[name],
                    "blend_without_candidate_loss": loss_without,
                    "blend_with_candidate_loss": loss_with,
                    "ablation_improvement": loss_without - loss_with,
                }
            )

    return report


__all__ = ["diversity_ablation_report"]
