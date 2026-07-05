"""Split-metrics emitters carved out of ``_trainer_train_and_evaluate``.

These two thin wrappers were previously nested closures inside
``train_and_evaluate_model``; lifting them to module-level helpers (taking the
needed locals explicitly) keeps the orchestration module under the 1k-LOC budget
without changing behaviour. Imported + called from the parent.
"""
from __future__ import annotations

from ._eval_helpers import _compute_split_metrics


def _run_val_split_metrics(_val_cfg, metrics, has_test, common_metrics_params):
    """Compute the val-split metrics. No-op (None) when no val split was configured."""
    if _val_cfg is None:
        return None
    _, sdf, starg, sidx, spreds, sprobs, sdet, _sc = _val_cfg
    return _compute_split_metrics(
        split_name="val",
        df=sdf,
        target=starg,
        idx=sidx,
        metrics_dict=metrics["val"],
        preds=spreds,
        probs=sprobs,
        details=sdet,
        has_other_splits=has_test,
        **common_metrics_params,
    )


def _run_test_split_metrics(_run_test, metrics, test_df, test_target, test_idx, test_preds, test_probs, test_details, common_metrics_params):
    """Compute the test-split metrics. No-op (None) when the test split is not being run."""
    if not _run_test:
        return None
    return _compute_split_metrics(
        split_name="test",
        df=test_df,
        target=test_target,
        idx=test_idx,
        metrics_dict=metrics["test"],
        preds=test_preds,
        probs=test_probs,
        details=test_details,
        has_other_splits=False,
        **common_metrics_params,
    )
