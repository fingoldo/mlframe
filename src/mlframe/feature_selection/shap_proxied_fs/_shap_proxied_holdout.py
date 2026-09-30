"""Search / holdout row split of ``ShapProxiedFS.fit``, following the suite's shared split policy."""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.cv_policy import get_cv_policy, holdout_indices


def split_search_and_holdout(selector: Any, idx_all: np.ndarray, n_rows: int, stratify: Any) -> Tuple[np.ndarray, np.ndarray]:
    """``(idx_search, idx_holdout)`` for the honest re-validation / trust guard.

    A temporal / grouped policy stamped on ``selector`` makes the holdout the newest rows / whole groups, so the guard is scored on the rows a real
    deployment would see, and records the search rows' policy as ``selector._search_cv_policy`` for the OOF SHAP fold partition. An i.i.d. policy (or none)
    keeps the stratified shuffle.
    """
    policy = get_cv_policy(selector)
    policy_split = holdout_indices(policy, n_rows, selector.holdout_size, random_state=int(selector.random_state))
    if policy_split is not None and policy is not None:
        idx_search, idx_hold = policy_split
        selector._search_cv_policy = policy.subset(idx_search)
        return idx_search, idx_hold
    selector._search_cv_policy = None
    idx_search, idx_hold = train_test_split(idx_all, test_size=selector.holdout_size, random_state=int(selector.random_state), shuffle=True, stratify=stratify)
    return idx_search, idx_hold
