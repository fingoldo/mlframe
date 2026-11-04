"""The member quality gate excludes a member that is NaN on some rows, and still catches an outlier when the peers agree exactly."""
import numpy as np

from mlframe.models.ensembling.quality_gate import compute_member_quality_gate


def test_member_that_is_nan_on_a_few_rows_is_excluded():
    """Row 0 was dropped for ALL members as soon as one was non-finite, so member 2 came out with finite statistics, was kept, and the blend was NaN on row 0."""
    rng = np.random.default_rng(0)
    base = rng.random(50)
    bad = base.copy()
    bad[0] = np.nan
    kept, excluded, _stats = compute_member_quality_gate([base, base + 0.01, bad])
    assert 2 not in kept and any(i == 2 for i, _reason in excluded)


def test_outlier_is_excluded_when_the_other_members_agree_exactly():
    """[a, a, a + 1000]: the median MAE is 0, which used to switch the relative threshold off."""
    a = np.random.default_rng(1).random(40)
    kept, excluded, _stats = compute_member_quality_gate([a, a.copy(), a + 1000.0])
    assert kept == [0, 1] and [i for i, _r in excluded] == [2]


def test_identical_members_are_all_kept():
    """No spread at all keeps everyone."""
    a = np.random.default_rng(2).random(40)
    kept, excluded, _stats = compute_member_quality_gate([a, a.copy(), a.copy()])
    assert kept == [0, 1, 2] and not excluded
