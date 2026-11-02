"""time_aware target encoding uses earlier rows only and leaves no unassigned rows (tracker 2026-11-02 wave 15, LEAK15-1)."""

import numpy as np

from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder


def test_first_block_does_not_see_later_targets():
    """One category, a target that is 0 for the first half and 1 for the second: encodings of the first half must not carry the later 1s."""
    n = 100
    cats = np.array(["a"] * n)
    y = np.concatenate([np.zeros(50), np.ones(50)])
    enc = LeakageSafeEncoder(method="target_mean", cv=4, time_aware=True, smoothing=0.0)
    out = np.asarray(enc.fit_transform(cats, y)).ravel()
    assert np.isfinite(out).all()
    assert out[:40].max() <= 0.5 + 1e-9  # the first block gets the prior, the second the mean of the first (0)
    assert out[20:40].max() == 0.0 or out[20:40].max() < 0.5
