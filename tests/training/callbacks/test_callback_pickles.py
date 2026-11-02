"""The early-stopping callbacks attached to fitted boosters survive plain pickle (tracker 2026-11-02, PKL7-1)."""

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

from mlframe.training.callbacks._callbacks import CatBoostCallback, LightGBMCallback


def test_lightgbm_callback_pickles_with_a_stop_closure_inside():
    """A user stop function, a lambda, used to make the callback unpicklable; the restored callback never asks to stop."""
    cb = LightGBMCallback(stop_flag=lambda: True)
    clone = pickle.loads(pickle.dumps(cb))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert clone.stop_flag() is False
    assert clone._widget is None


def test_catboost_callback_pickles_by_default():
    """The default stop flag is a module-level function, so no lambda is left in the instance."""
    clone = pickle.loads(pickle.dumps(CatBoostCallback()))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert clone.stop_flag() is False
