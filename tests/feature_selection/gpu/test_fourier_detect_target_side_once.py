"""The target's part of the Fourier detector's host split is done once per target object and shared by the columns."""

import numpy as np

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_detect_gpu_resident as mod


def test_columns_of_one_target_share_the_split_target(monkeypatch):
    """Five columns against one target: the target is split once, every column gets the same read-only target arrays and its own column arrays."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(5000)
    made = []
    real = mod._seeded_split_masks

    def spy(n):
        """Count the split-mask requests."""
        made.append(n)
        return real(n)

    monkeypatch.setattr(mod, "_seeded_split_masks", spy)
    outs = [mod._host_split_for_detect(rng.random(5000).astype(np.float32), y, [1.0, 2.0], min_rows=800, fourier_detect_max_n=2000) for _ in range(5)]
    assert all(o is not None for o in outs)
    assert len(made) == 1 + 5  # one for the target, one per column
    assert all(o[3] is outs[0][3] and o[4] is outs[0][4] for o in outs)
    assert not outs[0][3].flags.writeable and len(outs[0][1]) == len(outs[0][3]) == 2000
    assert not np.array_equal(outs[0][1], outs[1][1])


def test_unusable_target_or_column_gives_none():
    """A flat or non-finite target, a flat column and a short frame are refused; a new target object is examined afresh."""
    rng = np.random.default_rng(1)
    z = rng.random(3000).astype(np.float32)
    kw = dict(min_rows=800, fourier_detect_max_n=0)
    assert mod._host_split_for_detect(z, np.ones(3000), [1.0], **kw) is None
    bad = rng.standard_normal(3000)
    bad[4] = np.inf
    assert mod._host_split_for_detect(z, bad, [1.0], **kw) is None
    good = rng.standard_normal(3000)
    assert mod._host_split_for_detect(np.full(3000, 0.5, dtype=np.float32), good, [1.0], **kw) is None
    assert mod._host_split_for_detect(z[:500], good[:500], [1.0], **kw) is None
    assert mod._host_split_for_detect(z, good, [1.0], **kw) is not None
