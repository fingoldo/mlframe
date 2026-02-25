"""
Regression tests for training callback compatibility.

Covers two fixes applied for XGBoost >= 2.x compatibility:

Fix 1 (helpers.py): UniversalCallback.__init__ now calls super().__init__(), which
  chains to TrainingCallback.__init__() in the MRO for XGBoostCallback. XGBoost >= 2.x
  made TrainingCallback an ABC and added a strict isinstance check in
  CallbackContainer.__init__. Not calling __init__ can cause isinstance failures in
  certain XGBoost versions.

Fix 2 (trainer.py): _setup_early_stopping_callback now filters existing_callbacks to
  only keep proper TrainingCallback instances. Old-style XGBoost 1.x callbacks (set via
  xgb_kwargs) that do not inherit from TrainingCallback would otherwise cause:
      TypeError: callback must be an instance of `TrainingCallback`.
"""

import pytest

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    import xgboost
    HAS_XGBOOST = True
    XGBOOST_VERSION = tuple(int(x) for x in xgboost.__version__.split(".")[:2])
except ImportError:
    HAS_XGBOOST = False
    XGBOOST_VERSION = (0, 0)

try:
    import lightgbm  # noqa: F401
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost  # noqa: F401
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from mlframe.training.helpers import XGBoostCallback, LightGBMCallback, CatBoostCallback
    HAS_MLFRAME_HELPERS = True
except (ImportError, ModuleNotFoundError):
    HAS_MLFRAME_HELPERS = False

try:
    from mlframe.training.trainer import _setup_early_stopping_callback
    HAS_MLFRAME_TRAINER = True
except (ImportError, ModuleNotFoundError):
    HAS_MLFRAME_TRAINER = False

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_xgboost = pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
requires_lightgbm = pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm not installed")
requires_catboost = pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not installed")
requires_mlframe_helpers = pytest.mark.skipif(
    not HAS_MLFRAME_HELPERS, reason="mlframe.training.helpers could not be imported"
)
requires_mlframe_trainer = pytest.mark.skipif(
    not HAS_MLFRAME_TRAINER, reason="mlframe.training.trainer could not be imported"
)
# XGBoost >= 2.x changed TrainingCallback to ABC with strict isinstance check
requires_xgboost_2 = pytest.mark.skipif(
    not HAS_XGBOOST or XGBOOST_VERSION < (2, 0),
    reason="strict isinstance check only in XGBoost >= 2.x",
)


# =============================================================================
# Fix 1: UniversalCallback.super().__init__() — XGBoostCallback isinstance
# =============================================================================


class TestXGBoostCallbackIsInstance:
    """XGBoostCallback must be a proper TrainingCallback subclass.

    XGBoost >= 2.x: CallbackContainer.__init__ raises TypeError if any callback
    in the list is not an instance of xgboost.callback.TrainingCallback.
    """

    @requires_xgboost
    @requires_mlframe_helpers
    def test_isinstance_trainingcallback(self):
        """XGBoostCallback must pass isinstance(cb, TrainingCallback)."""
        from xgboost.callback import TrainingCallback

        cb = XGBoostCallback()
        assert isinstance(cb, TrainingCallback), (
            "XGBoostCallback must be a TrainingCallback subclass. "
            "XGBoost >= 2.x CallbackContainer raises TypeError otherwise."
        )

    @requires_xgboost
    @requires_mlframe_helpers
    def test_trainingcallback_init_called_via_mro(self):
        """TrainingCallback.__init__ must be invoked through the MRO chain.

        MRO for XGBoostCallback(UniversalCallback, TrainingCallback):
            XGBoostCallback -> UniversalCallback -> TrainingCallback -> object

        UniversalCallback.__init__ must call super().__init__() so that
        TrainingCallback.__init__() is executed. This ensures ABC registration
        is complete for all XGBoost versions.
        """
        from xgboost.callback import TrainingCallback

        init_calls = []
        original_init = TrainingCallback.__init__

        try:
            def patched_init(self):
                init_calls.append(self)
                original_init(self)

            TrainingCallback.__init__ = patched_init
            cb = XGBoostCallback()
            assert cb in init_calls, (
                "TrainingCallback.__init__ was not called during XGBoostCallback instantiation. "
                "UniversalCallback.__init__ must call super().__init__()."
            )
        finally:
            TrainingCallback.__init__ = original_init

    @requires_xgboost_2
    @requires_mlframe_helpers
    def test_fit_no_typeerror_with_early_stopping(self):
        """XGBoostCallback must not cause TypeError in XGBoost fit().

        Regression test for:
            TypeError: callback must be an instance of `TrainingCallback`.
        which was raised by XGBoost's CallbackContainer when XGBoostCallback was
        not properly recognised as a TrainingCallback subclass.
        """
        import numpy as np
        from xgboost import XGBClassifier

        cb = XGBoostCallback(patience=5, time_budget_mins=10)
        model = XGBClassifier(n_estimators=20, early_stopping_rounds=3, verbosity=0, callbacks=[cb])

        X = np.random.default_rng(0).random((100, 4))
        y = np.random.default_rng(0).integers(0, 2, 100)

        # Must not raise: TypeError: callback must be an instance of `TrainingCallback`
        model.fit(X, y, eval_set=[(X, y)], verbose=False)

    @requires_xgboost
    @requires_lightgbm
    @requires_mlframe_helpers
    def test_lightgbm_callback_not_xgb_trainingcallback(self):
        """LightGBMCallback does NOT inherit from xgboost TrainingCallback — intentionally.

        LightGBM uses a __call__ protocol, not the XGBoost ABC interface.
        This test documents expected behaviour and ensures the super().__init__()
        change in UniversalCallback does not accidentally make LightGBMCallback
        appear as an XGBoost TrainingCallback.
        """
        from xgboost.callback import TrainingCallback

        cb = LightGBMCallback()
        assert not isinstance(cb, TrainingCallback)

    @requires_lightgbm
    @requires_catboost
    @requires_mlframe_helpers
    def test_super_init_does_not_break_other_callbacks(self):
        """super().__init__() in UniversalCallback must not break LightGBM/CatBoost callbacks.

        For LightGBMCallback(UniversalCallback) and CatBoostCallback(UniversalCallback) the
        MRO ends at object after UniversalCallback, so super().__init__() calls
        object.__init__() which is always safe.
        """
        lgb_cb = LightGBMCallback(patience=10)
        assert lgb_cb.patience == 10

        cb_cb = CatBoostCallback(time_budget_mins=5)
        assert cb_cb.time_budget_mins == 5


# =============================================================================
# Fix 2: _setup_early_stopping_callback filters non-TrainingCallback objects
# =============================================================================


class TestSetupEarlyStoppingCallbackLegacyFilter:
    """Regression tests for the legacy-callback filter in _setup_early_stopping_callback.

    When a model is constructed with xgb_kwargs containing callbacks from XGBoost 1.x
    that do not inherit from TrainingCallback, they must be silently dropped before
    being passed to XGBoost's CallbackContainer, which raises TypeError otherwise.
    """

    @requires_xgboost_2
    @requires_mlframe_helpers
    @requires_mlframe_trainer
    def test_legacy_non_trainingcallback_filtered(self):
        """Non-TrainingCallback objects in existing_callbacks must be removed.

        Scenario: user passes xgb_kwargs={'callbacks': [old_callback]} where
        old_callback is an XGBoost 1.x style callback (no TrainingCallback base).
        After _setup_early_stopping_callback, old_callback must be gone.
        """
        from xgboost import XGBClassifier
        from xgboost.callback import TrainingCallback

        class LegacyCallback:
            """Old-style XGBoost 1.x callback — does not inherit TrainingCallback."""
            def after_iteration(self, model, epoch, evals_log):
                return False

        legacy_cb = LegacyCallback()
        assert not isinstance(legacy_cb, TrainingCallback), "pre-condition"

        model_obj = XGBClassifier(n_estimators=10, verbosity=0, callbacks=[legacy_cb])
        callback_params = {"time_budget_mins": 60, "patience": 10}

        _setup_early_stopping_callback("xgb", {}, callback_params, model_obj)

        callbacks = model_obj.get_params().get("callbacks", [])
        assert legacy_cb not in callbacks, "Legacy callback must be filtered out"
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], XGBoostCallback)
        assert all(isinstance(cb, TrainingCallback) for cb in callbacks)

    @requires_xgboost_2
    @requires_mlframe_helpers
    @requires_mlframe_trainer
    def test_fit_no_typeerror_after_legacy_callback_filter(self):
        """Integration: fit() must not raise TypeError when model had a legacy callback.

        Full end-to-end regression test for the TypeError seen after upgrading
        to XGBoost >= 2.x on a model constructed with old-style callbacks.
        """
        import numpy as np
        from xgboost import XGBClassifier

        class LegacyCallback:
            def after_iteration(self, model, epoch, evals_log):
                return False

        model_obj = XGBClassifier(
            n_estimators=20, early_stopping_rounds=3, verbosity=0,
            callbacks=[LegacyCallback()],
        )
        _setup_early_stopping_callback("xgb", {}, {"time_budget_mins": 60, "patience": 5}, model_obj)

        X = np.random.default_rng(42).random((120, 5))
        y = np.random.default_rng(42).integers(0, 2, 120)
        model_obj.fit(X, y, eval_set=[(X, y)], verbose=False)

    @requires_xgboost_2
    @requires_mlframe_helpers
    @requires_mlframe_trainer
    def test_valid_user_callbacks_preserved(self):
        """Valid user-provided TrainingCallback instances must NOT be filtered out.

        Only invalid (non-TrainingCallback) objects are removed; proper user
        callbacks that happen to not be XGBoostCallback must survive.
        """
        from xgboost import XGBClassifier
        from xgboost.callback import TrainingCallback

        class ValidUserCallback(TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                return False

        user_cb = ValidUserCallback()
        model_obj = XGBClassifier(n_estimators=10, verbosity=0, callbacks=[user_cb])
        _setup_early_stopping_callback("xgb", {}, {"time_budget_mins": 60, "patience": 10}, model_obj)

        callbacks = model_obj.get_params().get("callbacks", [])
        assert user_cb in callbacks, "Valid user callback must be preserved"
        assert len(callbacks) == 2
        xgb_cbs = [cb for cb in callbacks if isinstance(cb, XGBoostCallback)]
        assert len(xgb_cbs) == 1
