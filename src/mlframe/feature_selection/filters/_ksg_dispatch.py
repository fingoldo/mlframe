"""KSG mutual information of every column of a matrix against a target, with the repository's own kernels in place of ``sklearn.feature_selection.mutual_info_*``.

``ksg_mi_discrete_target`` (``_ksg_cd_fast``) and ``ksg_mi_continuous_target`` (``_ksg_cc_fast``) reproduce sklearn's estimate for continuous columns (scaling, the Kraskov noise drawn from the same
``RandomState`` stream, the ``nextafter`` radius), so a call site changes only its import. The classification form takes the labels as classes of any dtype; the regression form takes them as numbers.
"""

from __future__ import annotations

import numpy as np

from ._ksg_cc_fast import ksg_mi_continuous_target
from ._ksg_cd_fast import ksg_mi_discrete_target


def ksg_mi_columns(X: np.ndarray, y: np.ndarray, *, discrete_target: bool, n_neighbors: int, random_state: int) -> np.ndarray:
    """``mutual_info_classif`` (``discrete_target``) or ``mutual_info_regression`` of every column of ``X`` against ``y``, ``discrete_features=False``: MI per column in nats."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if discrete_target:
        return ksg_mi_discrete_target(X, np.asarray(y).ravel(), n_neighbors=int(n_neighbors), random_state=int(random_state))
    return ksg_mi_continuous_target(X, np.asarray(y).ravel(), n_neighbors=int(n_neighbors), random_state=int(random_state))
