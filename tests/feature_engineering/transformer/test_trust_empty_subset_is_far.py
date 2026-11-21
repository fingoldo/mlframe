"""An empty correct-subset gives the far sentinel for both distances (audit 2026-11-20, finding 09-F12)."""
import numpy as np

from mlframe.feature_engineering.transformer.trust_score_oof import _FAR, _nn_dist_to_subset


def test_empty_subset_is_far_for_mean_and_nearest():
    """Nothing is close to an empty subset: both the mean and the nearest distance must be the far sentinel, not 1.0 and 0.0."""
    Xt = np.zeros((5, 2), dtype=np.float32)
    mask = np.zeros(5, dtype=bool)
    Xq = np.ones((3, 2), dtype=np.float32)
    mean, nearest = _nn_dist_to_subset(Xt, mask, Xq, k_neighbors=4)
    assert np.all(mean == _FAR) and np.all(nearest == _FAR)


def test_nonempty_subset_still_measures_distance():
    """A non-empty subset returns real distances."""
    Xt = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.array([True, True])
    Xq = np.array([[0.0, 0.0]], dtype=np.float32)
    mean, nearest = _nn_dist_to_subset(Xt, mask, Xq, k_neighbors=2)
    assert nearest[0] == 0.0
    assert mean[0] == 2.5
