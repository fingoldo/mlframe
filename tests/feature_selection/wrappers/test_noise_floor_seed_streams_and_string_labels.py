"""Noise-floor permutation streams must not overlap across random_state values; string class labels must be accepted."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from mlframe.feature_selection.wrappers._noise_floor import _infer_task_scoring, cv_curve


def test_noise_floor_permutation_streams_do_not_overlap_between_random_states():
    """base_seed 100 permutation 1 and base_seed 101 permutation 0 must not be the same shuffle of y."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y = pd.Series(X["a"] * 2.0 + rng.normal(size=80))
    ranking = list("abcd")
    kw = dict(permute=True, n_perm=2, scoring="r2")
    cv = KFold(3, shuffle=True, random_state=0)
    _, curves_a = cv_curve(lambda: Ridge(), X, y, ranking, [1, 2, 3], cv, base_seed=100, **kw)
    _, curves_b = cv_curve(lambda: Ridge(), X, y, ranking, [1, 2, 3], cv, base_seed=101, **kw)
    assert not np.allclose(curves_a[1], curves_b[0])


def test_infer_task_scoring_accepts_string_class_labels():
    """Binary and multiclass string labels are classification, not a crash in np.round."""
    assert _infer_task_scoring(np.array(["a", "b"] * 10)) == ("binary", "roc_auc")
    assert _infer_task_scoring(np.array(["a", "b", "c"] * 10)) == ("multiclass", "roc_auc_ovr")
