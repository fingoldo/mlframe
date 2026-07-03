"""Regression (MRMR critique ST-1): the UAED elbow trim computed the elbow in COMBINED (raw+engineered) index space
but sliced the RAW-only support_ and set n_features_ = support_.size, dropping the engineered count while the
recipes still fired in transform -> transform emitted more columns than n_features_/mrmr_gains_ claimed. The trim now
retains raw support AND engineered recipes in lockstep so the retained count is exactly elbow+1 everywhere.

The invariant this pins (must hold for ANY uaed_auto_size fit, with or without engineered features):
  len(get_feature_names_out()) == n_features_ == transform(X).shape[1] == len(mrmr_gains_)
"""
import numpy as np
import pandas as pd
import pytest


def _signal_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    # 3 strong signal cols + several weak/noise cols so the greedy gain trace has a clear elbow
    x = rng.integers(0, 5, size=(n, 10)).astype(float)
    y = ((x[:, 0] + x[:, 1] + x[:, 2]) > 6).astype(int)
    cols = [f"f{i}" for i in range(10)]
    return pd.DataFrame(x, columns=cols), pd.Series(y)


@pytest.mark.parametrize("with_fe", [False, True])
def test_uaed_support_transform_gains_consistent(with_fe):
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _signal_frame()
    kw = dict(uaed_auto_size=True, full_npermutations=1, cv=2, run_additional_rfecv_minutes=False)
    if with_fe:
        # interactions produce k-way engineered candidates; the invariant below must hold whether or not the UAED
        # elbow trims into the engineered tail (the ST-1 fix trims raw support AND engineered recipes in lockstep).
        kw.update(interactions_max_order=2)
    m = MRMR(**kw)
    m.fit(X, y)

    n_out = len(m.get_feature_names_out())
    n_feat = int(m.n_features_)
    width = m.transform(X).shape[1]
    assert n_out == n_feat == width, (
        f"UAED support/output desync (with_fe={with_fe}): get_feature_names_out={n_out}, "
        f"n_features_={n_feat}, transform_width={width}"
    )
    if getattr(m, "mrmr_gains_", None) is not None:
        assert len(np.asarray(m.mrmr_gains_)) == n_feat, "mrmr_gains_ length != n_features_ after UAED trim"
