"""Learning-to-Rank subsystem.

Groups the LTR fit/predict primitives and the top-level ranker suite:

- ``ranking`` -- ``fit_ranker`` / ``predict_ranker_scores`` /
  ``ensemble_ranker_scores`` + the per-strategy input-prep helpers
  (CatBoostRanker / XGBRanker / LGBMRanker) under a uniform contract.
- ``ranker_suite`` -- ``train_mlframe_ranker_suite`` (LTR analogue of
  ``train_mlframe_models_suite``) + rank-fusion helpers (``rrf_fuse`` /
  ``borda_fuse``).

The public surface is re-exported here so existing
``from mlframe.training.ranking import X`` and
``from mlframe.training.ranker_suite import X`` import sites resolve from
the documented package path.
"""
from __future__ import annotations

from .ranking import (
    fit_ranker,
    predict_ranker_scores,
    ensemble_ranker_scores,
    qid_to_group_sizes,
    prepare_cb_inputs,
    prepare_xgb_inputs,
    prepare_lgb_inputs,
)
from .ranker_suite import (
    train_mlframe_ranker_suite,
    rrf_fuse,
    borda_fuse,
)

__all__ = [
    "fit_ranker",
    "predict_ranker_scores",
    "ensemble_ranker_scores",
    "train_mlframe_ranker_suite",
    "rrf_fuse",
    "borda_fuse",
]
