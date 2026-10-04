"""BorutaShap working-set memory: shadow buffer peak, caller-frame sharing, release of large working frames."""
from __future__ import annotations

import tracemalloc
import types

import numpy as np
import pandas as pd

from mlframe.feature_selection.boruta_shap import _shadow_stats as ss


def _state(X: pd.DataFrame) -> types.SimpleNamespace:
    """Minimal estimator stand-in carrying the attributes ``create_shadow_features`` reads."""
    return types.SimpleNamespace(X_=X, shadow_min_pad=0, _rng=np.random.default_rng(0))


def test_shadow_build_peak_is_two_frames_not_three():
    """Building [real | shadow] peaks at ~2x the frame (single buffer) where concat needed ~3x."""
    X = pd.DataFrame(np.random.default_rng(1).random((100_000, 20)))
    nb = int(X.memory_usage(index=False).sum())
    st = _state(X)
    tracemalloc.start()
    ss.create_shadow_features(st)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 2.4 * nb
    assert st.X_boruta_.shape == (100_000, 40)
    assert np.shares_memory(st.X_shadow_.to_numpy(), st.X_boruta_.to_numpy())


def test_shadow_values_are_permutations_of_real_columns():
    """Each shadow column holds the same multiset of values as its real column, in a different order."""
    X = pd.DataFrame(np.random.default_rng(2).random((500, 4)), columns=list("abcd"))
    st = _state(X)
    ss.create_shadow_features(st)
    assert list(st.X_boruta_.columns) == list("abcd") + ["shadow_" + c for c in "abcd"]
    for c in "abcd":
        assert np.array_equal(np.sort(st.X_boruta_["shadow_" + c].to_numpy()), np.sort(X[c].to_numpy()))
        assert not np.array_equal(st.X_boruta_["shadow_" + c].to_numpy(), X[c].to_numpy())
    assert np.array_equal(st.X_boruta_[list("abcd")].to_numpy(), X.to_numpy())


def test_fit_leaves_caller_frame_untouched_and_shares_buffers():
    """fit() on a frame with an object column encodes only the working copy; the caller's frame is bit-identical afterwards."""
    from mlframe.feature_selection.boruta_shap import BorutaShap
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(3)
    X = pd.DataFrame({"num": rng.random(300), "cat": rng.choice(["x", "y", "z"], 300), "noise": rng.random(300)})
    y = (X["num"] > 0.5).astype(int)
    snap = X.copy(deep=True)
    bs = BorutaShap(model=RandomForestClassifier(n_estimators=10, random_state=0), importance_measure="gini", classification=True, random_state=0, n_trials=3)
    bs.fit(X, y)
    pd.testing.assert_frame_equal(X, snap)
    assert X["cat"].dtype == object
    assert np.shares_memory(bs.X_["num"].to_numpy(), X["num"].to_numpy())


def test_large_working_frames_released_after_fit_small_ones_kept(monkeypatch):
    """Frames at or above the byte threshold are dropped at fit end; below it they stay inspectable."""
    X = pd.DataFrame(np.random.default_rng(4).random((200, 3)))
    st = _state(X)
    ss.create_shadow_features(st)
    st.X_boruta_train_ = st.X_boruta_.iloc[:100]
    st.X_boruta_test_ = st.X_boruta_.iloc[100:]
    ss.release_large_working_frames(st)
    assert st.X_boruta_ is not None
    monkeypatch.setattr(ss, "RELEASE_WORKING_FRAMES_MIN_BYTES", 1)
    ss.release_large_working_frames(st)
    assert st.X_boruta_ is None and st.X_shadow_ is None and st.X_boruta_train_ is None and st.X_boruta_test_ is None
