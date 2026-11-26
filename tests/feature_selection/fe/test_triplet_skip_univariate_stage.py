"""The triplet hybrid without its univariate stage gives the same triplet seeds, scores and columns: the seeds take the raw baseline MI from the
same plug-in estimator stage 1 uses. MRMR keeps only the triplet columns of this family, so it skips the stage."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._orthogonal_triplet_fe import hybrid_orth_mi_triplet_fe


def _frame(n, p, seed):
    """``p`` noise columns with a three-way product in the target, a strong marginal column and a weak one."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)).astype(np.float32), columns=[f"x{i}" for i in range(p)])
    signal = X["x3"] * X["x7"] * X["x11"] + 0.8 * X["x0"] + 0.2 * X["x5"] + 0.5 * rng.standard_normal(n)
    return X, (signal > np.median(signal)).astype(np.int64).to_numpy()


def _triplet_part(X_aug, X):
    """The appended triplet columns (three legs joined by '*')."""
    return X_aug[[c for c in X_aug.columns if c not in X.columns and c.split("__", 1)[0].count("*") == 2]]


@pytest.mark.parametrize("p", [12, 130])  # 12: every triple is enumerated; 130: C(p, 3) is over the budget and the seed pool is ranked by baseline MI
@pytest.mark.parametrize("seed", [0, 1])
def test_the_triplets_do_not_depend_on_the_univariate_stage(p, seed):
    """Seeds, triplet scores and the appended triplet columns are the same with and without stage 1; without it no univariate winner is appended."""
    X, y = _frame(3000, p, seed)
    full_aug, _full_uni, full_trip = hybrid_orth_mi_triplet_fe(X, y, top_triplet_seed_k=4, top_triplet_count=2)
    skip_aug, skip_uni, skip_trip = hybrid_orth_mi_triplet_fe(X, y, top_triplet_seed_k=4, top_triplet_count=2, skip_univariate_stage=True)
    pd.testing.assert_frame_equal(full_trip.reset_index(drop=True), skip_trip.reset_index(drop=True))
    pd.testing.assert_frame_equal(_triplet_part(full_aug, X), _triplet_part(skip_aug, X))
    assert skip_uni.empty
    assert [c for c in skip_aug.columns if c not in X.columns] == list(_triplet_part(skip_aug, X).columns)
