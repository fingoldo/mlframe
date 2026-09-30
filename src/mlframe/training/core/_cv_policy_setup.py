"""Suite-side construction of the shared feature-selector split policy (see ``mlframe.feature_selection.cv_policy``)."""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from mlframe.feature_selection.cv_policy import CVPolicy, decide_cv_policy

logger = logging.getLogger(__name__)


def _decide_suite_cv_policy(
    *,
    feature_selection_config: Any,
    timestamps: Any,
    train_idx: Optional[np.ndarray],
    group_ids: Any,
    split_config: Any,
    hyperparams_config: Any,
    verbose: Any = 0,
) -> Optional[CVPolicy]:
    """The per-target ``CVPolicy`` handed to every selector, or None when ``FeatureSelectionConfig.unified_cv_policy`` is off."""
    if not bool(getattr(feature_selection_config, "unified_cv_policy", True)):
        return None
    policy = decide_cv_policy(
        timestamps=timestamps, train_idx=train_idx, groups=group_ids, split_config=split_config, hyperparams_config=hyperparams_config,
    )
    if verbose:
        logger.info("Feature-selector split policy: %s (%s).", policy.kind, policy.reason)
    return policy


def _publish_suite_cv_policy(*, rfecv_models_params: dict, common_params: dict, **decide_kwargs: Any) -> Optional[CVPolicy]:
    """Decide the per-target policy, hand it to the suite's RFECV instances, and default the OOF pass's ``oof_has_time`` to the same decision.

    An explicit ``oof_has_time`` on the behavior config (already in ``common_params``) wins. Returns the policy for ``_build_pre_pipelines``.
    """
    from ._rfecv_temporal_cv import apply_temporal_cv_to_rfecv

    apply_temporal_cv_to_rfecv(
        rfecv_models_params, timestamps=decide_kwargs["timestamps"], train_idx=decide_kwargs["train_idx"], split_config=decide_kwargs["split_config"],
        hyperparams_config=decide_kwargs["hyperparams_config"], verbose=decide_kwargs["verbose"],
    )
    policy = _decide_suite_cv_policy(**decide_kwargs)
    if policy is not None and common_params.get("oof_has_time") is None:
        common_params["oof_has_time"] = policy.temporal
    return policy
