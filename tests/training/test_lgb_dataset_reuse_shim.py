"""
Sensor tests for ``mlframe.training.lgb_shim`` -- the Dataset-reuse shim
that wraps ``LGBMClassifier`` / ``LGBMRegressor`` so that consecutive
``.fit()`` calls on the same feature matrix don't rebuild the
``lightgbm.Dataset`` (binning + histogram setup) from scratch.

Mirror of ``test_xgb_dmatrix_reuse_shim.py``. The shim exists because
our PR for ``Dataset`` support in LightGBM's sklearn wrapper is pending
acceptance upstream; until it lands we route through this shim.

What's tested
-------------
1. **API parity with LGBMClassifier** -- drop-in replacement: same
   ``get_params()``, ``set_params()``, ``predict``, ``predict_proba``,
   ``feature_importances_``, ``feature_names_in_``, sklearn ``clone()``
   round-trip.
2. **Dataset cache** -- second fit on the same DataFrame reuses the
   cached Dataset; second fit on a different DataFrame misses; cache
   resets on clone.
3. **In-place swaps** -- ``set_label(y)`` and ``set_weight(w)`` mutate
   the cached Dataset without rebuild.
4. **Predict parity** -- predictions from the shim match predictions
   from a vanilla LGBMClassifier trained on the same data, within a
   small numerical tolerance (model determinism on the same seed).
5. **Regressor variant** -- same parity tests for ``LGBMRegressorWithDatasetReuse``.

Run-time budget: each test trains on ~500 rows x 5 cols x 5 trees, so
the whole suite finishes in seconds -- fine for CI.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor

# Module under test -- import lazily so tests collect fine when shim
# isn't ready yet (e.g. during initial TDD).
try:
    from mlframe.training.lgb_shim import (
        LGBMClassifierWithDatasetReuse,
        LGBMRegressorWithDatasetReuse,
        lgb_dataset_reuse_capable,
    )

    SHIM_AVAILABLE = True
except ImportError:
    SHIM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SHIM_AVAILABLE,
    reason="lgb_shim not yet implemented -- TDD phase",
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def small_classification_data():
    """Tiny but real classification dataset -- converges in a few trees."""
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(
        {
            "f0": rng.standard_normal(n).astype(np.float32),
            "f1": rng.standard_normal(n).astype(np.float32),
            "f2": rng.standard_normal(n).astype(np.float32),
            "f3": rng.standard_normal(n).astype(np.float32),
            "f4": rng.standard_normal(n).astype(np.float32),
        }
    )
    y = ((X["f0"] + X["f1"] - X["f2"]) > 0).astype(np.int32).to_numpy()
    return X, y


@pytest.fixture
def small_regression_data():
    """Small regression data."""
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n).astype(np.float32) for i in range(5)})
    y = (X["f0"] * 2 - X["f2"] + rng.standard_normal(n) * 0.1).to_numpy(np.float32)
    return X, y


# Reduce LGB log noise from these small fits so test output stays clean.
_QUIET_LGB = dict(verbosity=-1, min_child_samples=2, min_data_in_bin=1)


# =====================================================================
# 1. API parity with LGBMClassifier (drop-in replacement)
# =====================================================================


class TestLGBClassifierShimAPIParity:
    """Shim must look like LGBMClassifier to all callers -- sklearn clone,
    feature importance, predict, predict_proba, etc."""

    def test_subclass_of_LGBMClassifier(self):
        """Subclass of l g b m classifier."""
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        assert isinstance(m, LGBMClassifier), (
            "shim must subclass LGBMClassifier so isinstance checks downstream (sklearn pipelines, mlframe strategy) keep passing"
        )

    def test_get_params_includes_lgb_params(self):
        """Get params includes lgb params."""
        m = LGBMClassifierWithDatasetReuse(
            n_estimators=7,
            max_depth=4,
            learning_rate=0.1,
            **_QUIET_LGB,
        )
        params = m.get_params()
        assert params["n_estimators"] == 7
        assert params["max_depth"] == 4
        assert params["learning_rate"] == 0.1

    def test_set_params_works(self):
        """Set params works."""
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        m.set_params(n_estimators=11, max_depth=5)
        assert m.n_estimators == 11
        assert m.max_depth == 5

    def test_sklearn_clone_round_trip(self):
        """Sklearn clone round trip."""
        from sklearn.base import clone

        m = LGBMClassifierWithDatasetReuse(n_estimators=7, max_depth=4, **_QUIET_LGB)
        c = clone(m)
        assert isinstance(c, LGBMClassifierWithDatasetReuse)
        assert c.n_estimators == 7
        assert c.max_depth == 4
        # Cache MUST NOT survive clone -- fresh instance, fresh state.
        assert getattr(c, "_cached_train_dataset", None) is None

    def test_fit_predict_predict_proba_run(self, small_classification_data):
        """Fit predict predict proba run."""
        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(
            n_estimators=5,
            max_depth=3,
            learning_rate=0.3,
            **_QUIET_LGB,
        )
        m.fit(X, y)
        preds = m.predict(X)
        proba = m.predict_proba(X)
        assert preds.shape == (len(y),)
        assert proba.shape == (len(y), 2)
        # Probabilities sum to 1.
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_feature_importances_available_after_fit(self, small_classification_data):
        """Feature importances available after fit."""
        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(n_estimators=5, **_QUIET_LGB)
        m.fit(X, y)
        fi = m.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert fi.sum() > 0  # something was actually used

    def test_n_features_in_set_after_fit(self, small_classification_data):
        """N features in set after fit."""
        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(n_estimators=5, **_QUIET_LGB)
        m.fit(X, y)
        # LGB exposes both n_features_ and n_features_in_; the shim
        # mirrors LGB sklearn by setting _n_features (read by both
        # n_features_ and the n_features_in_ property in newer versions).
        assert m.n_features_in_ == X.shape[1]


# =====================================================================
# 2. Predict parity vs vanilla LGBMClassifier
# =====================================================================


class TestLGBClassifierShimPredictParity:
    """The shim must produce IDENTICAL predictions to a vanilla
    LGBMClassifier under the same hyperparameters and seed -- within
    floating-point tolerance.
    """

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_proba_matches_vanilla_lgbmclassifier(self, small_classification_data, seed):
        """Proba matches vanilla lgbmclassifier."""
        X, y = small_classification_data
        params = dict(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.3,
            random_state=seed,
            **_QUIET_LGB,
        )
        ref = LGBMClassifier(**params)
        ref.fit(X, y)
        ref_proba = ref.predict_proba(X)

        shim = LGBMClassifierWithDatasetReuse(**params)
        shim.fit(X, y)
        shim_proba = shim.predict_proba(X)

        np.testing.assert_allclose(
            shim_proba,
            ref_proba,
            atol=1e-5,
            err_msg=("shim predictions diverged from vanilla LGBMClassifier -- the Dataset-reuse fit path is not numerically equivalent"),
        )


# =====================================================================
# 3. Dataset cache -- reuse / miss / reset semantics
# =====================================================================


class TestLGBDatasetReuse:
    """The cache must hit on identical X and miss on different X.
    Cache is per-instance (cleared by sklearn.clone)."""

    def test_first_fit_builds_dataset_and_caches(self, small_classification_data):
        """First fit builds dataset and caches."""
        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        assert m._cached_train_dataset is None
        m.fit(X, y)
        assert m._cached_train_dataset is not None
        # Cached Dataset has the right number of rows.
        assert m._cached_train_dataset.num_data() == len(y)

    def test_second_fit_same_data_reuses_dataset(self, small_classification_data):
        """Second fit same data reuses dataset."""
        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        m.fit(X, y)
        first_id = id(m._cached_train_dataset)

        # Second fit on the same X must reuse -- Dataset instance
        # identity preserved.
        m.fit(X, y)
        second_id = id(m._cached_train_dataset)
        assert first_id == second_id, "Dataset was rebuilt on second fit with same X -- cache miss"

    def test_second_fit_different_data_misses_cache(self, small_classification_data):
        """Second fit different data misses cache."""
        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        m.fit(X, y)
        first_id = id(m._cached_train_dataset)

        # Build a different-shape frame.
        X2 = X.iloc[:100].copy()
        y2 = y[:100]
        m.fit(X2, y2)
        second_id = id(m._cached_train_dataset)
        assert first_id != second_id, "Dataset was reused for a different-shape X -- cache key bug"

    def test_clone_resets_cache(self, small_classification_data):
        """Clone resets cache."""
        from sklearn.base import clone

        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        m.fit(X, y)
        assert m._cached_train_dataset is not None

        c = clone(m)
        assert c._cached_train_dataset is None, "cloned shim still carries the original's Dataset cache -- sklearn.clone() must produce a fresh instance"

    def test_eval_set_dataset_also_cached(self, small_classification_data):
        """Eval set dataset also cached."""
        X, y = small_classification_data
        X_val = X.iloc[:100].copy()
        y_val = y[:100]

        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        m.fit(X, y, eval_set=[(X_val, y_val)])
        assert m._cached_val_dataset is not None
        assert m._cached_val_dataset.num_data() == len(y_val)

    def test_categorical_feature_change_misses_cache(self, small_classification_data):
        """Cache key must include categorical_feature -- changing it
        between fits MUST miss the cache, otherwise LGB would silently
        produce wrong splits using stale binning."""
        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        # First fit: no categorical features.
        m.fit(X, y, categorical_feature=[])
        first_id = id(m._cached_train_dataset)
        # Second fit: declare f0 as categorical -- different Dataset.
        m.fit(X, y, categorical_feature=["f0"])
        second_id = id(m._cached_train_dataset)
        assert first_id != second_id, "Dataset reused despite categorical_feature change -- cache key bug; would silently produce wrong splits"


# =====================================================================
# 4. In-place set_label / set_weight on cached Dataset
# =====================================================================


class TestLGBShimSetLabelSetWeight:
    """Public extras: ``.set_label(y)`` / ``.set_weight(w)`` mutate the
    cached Dataset in place, no rebuild. Tested by checking the
    Dataset instance identity stays the same."""

    def test_set_weight_mutates_in_place(self, small_classification_data):
        """Set weight mutates in place."""
        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        m.fit(X, y)
        id_before = id(m._cached_train_dataset)

        new_weight = np.linspace(0.1, 1.0, len(y)).astype(np.float32)
        m.set_weight(new_weight)
        id_after = id(m._cached_train_dataset)

        assert id_before == id_after, "set_weight rebuilt the Dataset -- must be in-place"
        np.testing.assert_array_equal(
            m._cached_train_dataset.get_weight(),
            new_weight,
        )

    def test_set_label_mutates_in_place(self, small_classification_data):
        """Set label mutates in place."""
        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        m.fit(X, y)
        id_before = id(m._cached_train_dataset)

        new_label = (1 - y).astype(np.float32)
        m.set_label(new_label)
        id_after = id(m._cached_train_dataset)

        assert id_before == id_after
        np.testing.assert_array_equal(
            m._cached_train_dataset.get_label(),
            new_label,
        )

    def test_set_weight_before_fit_raises(self):
        """Set weight before fit raises."""
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        with pytest.raises(RuntimeError, match="Dataset"):
            m.set_weight(np.ones(10))

    def test_second_fit_with_different_weight_does_not_rebuild(
        self,
        small_classification_data,
    ):
        """Most valuable use case: second fit with same X but new
        sample_weight reuses the cached Dataset and just swaps weight
        in place. This is the same prod-log scenario the XGB shim
        targets (uniform -> recency on the same train_df)."""
        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)

        m.fit(X, y)
        first_id = id(m._cached_train_dataset)

        recency = np.linspace(0.1, 1.0, len(y)).astype(np.float32)
        m.fit(X, y, sample_weight=recency)
        second_id = id(m._cached_train_dataset)

        assert first_id == second_id, (
            "Second fit with new sample_weight rebuilt the Dataset -- the in-place set_weight path is not firing. This is the weight-schema-loop saving target."
        )


# =====================================================================
# 5. lgb_dataset_reuse_capable() -- capability gate
# =====================================================================


class TestLGBReuseCapability:
    """Groups tests covering l g b reuse capability."""
    def test_capability_check_returns_bool(self):
        """Capability check returns bool."""
        result = lgb_dataset_reuse_capable()
        assert isinstance(result, bool)

    def test_capability_true_on_modern_lightgbm(self):
        # lightgbm >= 3.x has set_label/set_weight on Dataset --
        # the test environment installs a modern version, so this
        # should be True.
        """Capability true on modern lightgbm."""
        assert lgb_dataset_reuse_capable() is True


# =====================================================================
# 6. LGBMRegressor variant -- same contract
# =====================================================================


class TestLGBRegressorShim:
    """Groups tests covering l g b regressor shim."""
    def test_subclass_of_LGBMRegressor(self):
        """Subclass of l g b m regressor."""
        m = LGBMRegressorWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        assert isinstance(m, LGBMRegressor)

    def test_predict_parity_with_vanilla(self, small_regression_data):
        """Predict parity with vanilla."""
        X, y = small_regression_data
        params = dict(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.3,
            random_state=0,
            **_QUIET_LGB,
        )
        ref = LGBMRegressor(**params)
        ref.fit(X, y)
        ref_pred = ref.predict(X)

        shim = LGBMRegressorWithDatasetReuse(**params)
        shim.fit(X, y)
        shim_pred = shim.predict(X)

        np.testing.assert_allclose(
            shim_pred,
            ref_pred,
            atol=1e-5,
            err_msg="regressor shim diverged from vanilla LGBMRegressor",
        )

    def test_dataset_reuse_works_for_regressor(self, small_regression_data):
        """Dataset reuse works for regressor."""
        X, y = small_regression_data
        m = LGBMRegressorWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        m.fit(X, y)
        first_id = id(m._cached_train_dataset)
        m.fit(X, y)
        assert id(m._cached_train_dataset) == first_id


# =====================================================================
# 7. Edge cases
# =====================================================================


class TestLGBShimEdgeCases:
    """Groups tests covering l g b shim edge cases."""
    def test_eval_set_changing_X_misses_val_cache(self, small_classification_data):
        """Eval set changing x misses val cache."""
        X, y = small_classification_data
        X_val_a = X.iloc[:100].copy()
        y_val_a = y[:100]
        X_val_b = X.iloc[100:200].copy()
        y_val_b = y[100:200]

        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        m.fit(X, y, eval_set=[(X_val_a, y_val_a)])
        first_val_id = id(m._cached_val_dataset)
        m.fit(X, y, eval_set=[(X_val_b, y_val_b)])
        second_val_id = id(m._cached_val_dataset)
        assert first_val_id != second_val_id, "val Dataset was reused for a different X_val -- cache key bug"

    def test_sample_weight_round_trip_on_first_fit(self, small_classification_data):
        """Sample weight round trip on first fit."""
        X, y = small_classification_data
        sw = np.linspace(0.5, 1.5, len(y)).astype(np.float32)
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        m.fit(X, y, sample_weight=sw)
        # Cached Dataset carries the weight set on construction.
        np.testing.assert_array_equal(
            m._cached_train_dataset.get_weight(),
            sw,
        )

    def test_eval_set_bare_tuple_normalised(self, small_classification_data):
        """mlframe (and some vanilla LGBM sklearn paths) pass a bare
        ``(X_val, y_val)`` 2-tuple instead of ``[(X_val, y_val)]``
        for the single-eval-set case. Without normalisation, iterating
        over the bare tuple would yield X_val first and y_val second,
        and the unpack would destructure X_val's column NAMES into
        (X, y) -- silently feeding ``np.str_('col_name')`` into the
        LabelEncoder. Lock in the normalisation."""
        X, y = small_classification_data
        X_val = X.iloc[:100].copy()
        y_val = y[:100]

        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        # Bare tuple, NOT wrapped in a list.
        m.fit(X, y, eval_set=(X_val, y_val))
        # Cache populated correctly -- val Dataset has the right row count.
        assert m._cached_val_dataset is not None
        assert m._cached_val_dataset.num_data() == len(y_val)

    def test_no_warnings_on_repeat_fit(self, small_classification_data):
        """Repeated .fit() must not spam UserWarning / FutureWarning
        from our shim layer (LGB itself may emit notices -- those aren't
        ours)."""
        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        m.fit(X, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m.fit(X, y, sample_weight=np.ones(len(y)))
        ours = [w for w in caught if issubclass(w.category, (UserWarning, RuntimeWarning)) and "shim" in str(w.message).lower()]
        assert not ours, f"shim emitted spurious warnings on repeat fit: {[str(w.message) for w in ours]}"


# =====================================================================
# 8. End-to-end: shim wired into _configure_lightgbm_params
# =====================================================================


class TestLGBShimIntegrationWithMlframeSuite:
    """Suite-level tests that the shim is wired into
    ``_configure_lightgbm_params`` and that the toggle controls dispatch.
    """

    def test_configure_lightgbm_uses_shim_when_toggle_on(self):
        """``_configure_lightgbm_params`` must instantiate the shim, not
        vanilla LGBMClassifier, when the toggle is on (default)."""
        from mlframe.training.trainer import (
            USE_LGB_DATASET_REUSE_SHIM,
            _lgb_classifier_cls,
            _lgb_regressor_cls,
        )

        # Pre-condition for the rest: toggle is on.
        assert USE_LGB_DATASET_REUSE_SHIM is True

        clf = _lgb_classifier_cls()
        assert clf is LGBMClassifierWithDatasetReuse

        reg = _lgb_regressor_cls()
        assert reg is LGBMRegressorWithDatasetReuse

    def test_configure_lightgbm_falls_back_to_vanilla_when_toggle_off(self, monkeypatch):
        """Flipping ``USE_LGB_DATASET_REUSE_SHIM`` to False must restore
        vanilla LightGBM -- proves the toggle is the single switching
        point and a future revert (after upstream PR lands) is one
        flag-flip away.
        """
        from mlframe.training import trainer as tr_mod

        monkeypatch.setattr(tr_mod, "USE_LGB_DATASET_REUSE_SHIM", False)

        clf = tr_mod._lgb_classifier_cls()
        assert clf is lgb.LGBMClassifier
        assert clf is not LGBMClassifierWithDatasetReuse

        reg = tr_mod._lgb_regressor_cls()
        assert reg is lgb.LGBMRegressor


# =====================================================================
# 9. Source-level checks: shim cache hand-off across sklearn.clone()
#    in core.py's strategy/weight-schema loop
# =====================================================================


class TestLGBShimCacheHandoffInCoreLoop:
    """Mirror of the XGB hand-off tests. Ensures core.py's strategy loop
    transfers the LGB Dataset cache forward (template -> cloned) and
    backward (cloned -> template) so the cache survives the
    sklearn.clone() that happens between weight-schema iterations.
    """

    def test_core_loop_forward_transfers_dataset_cache_to_clone(self, small_classification_data):
        """Behaviourally verify the forward-transfer helper used by core.py's
        weight-schema loop: ``_forward_dataset_reuse_cache(src=template,
        dst=clone)`` copies the LGB Dataset cache (``_cached_train_dataset``
        / ``_cached_val_dataset``) from a fitted template onto a fresh
        sklearn.clone() so the next weight-schema iteration's shim hits
        set_label / set_weight in place instead of rebuilding the binned
        Dataset. Drop the helper (or omit the LGB attrs from
        ``_DATASET_REUSE_CACHE_ATTRS``) and this test fails -- shim runs
        but produces no reuse, a silent perf regression.
        """
        from sklearn.base import clone
        from mlframe.training.core._phase_train_one_target import (
            _DATASET_REUSE_CACHE_ATTRS,
            _forward_dataset_reuse_cache,
        )

        # LGB-specific cache attrs MUST be inside the shared tuple so the
        # generic forward helper carries them across clone(). If somebody
        # drops them, dataset reuse silently dies for the LGB shim.
        for _attr in ("_cached_train_dataset", "_cached_val_dataset"):
            assert _attr in _DATASET_REUSE_CACHE_ATTRS, (
                f"{_attr!r} missing from _DATASET_REUSE_CACHE_ATTRS -- the forward helper will not propagate the LGB cache."
            )

        X, y = small_classification_data
        X_val = X.iloc[:100].copy()
        y_val = y[:100]

        template = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        template.fit(X, y, eval_set=[(X_val, y_val)])
        # Precondition: template carries populated caches.
        assert template._cached_train_dataset is not None
        assert template._cached_val_dataset is not None

        cloned = clone(template)
        # sklearn.clone() yields a virgin instance -- precondition for the test
        # to be meaningful.
        assert cloned._cached_train_dataset is None
        assert cloned._cached_val_dataset is None

        # The forward-transfer helper invoked by core.py.
        _forward_dataset_reuse_cache(template, cloned)

        # Same Dataset instance forwarded by reference -- the in-place
        # set_label / set_weight path on the next .fit() would then hit
        # the cached, prebinned dataset.
        assert cloned._cached_train_dataset is template._cached_train_dataset, (
            "forward helper failed to propagate _cached_train_dataset -- "
            "clone got None / different obj; next iteration's shim will "
            "rebuild the binned Dataset (silent perf regression)."
        )
        assert cloned._cached_val_dataset is template._cached_val_dataset, "forward helper failed to propagate _cached_val_dataset"

    def test_lgb_shim_factory_is_invoked_from_configure_lightgbm(self, monkeypatch):
        """``_configure_lightgbm_params`` must dispatch through ``_lgb_classifier_cls`` /
        ``_lgb_regressor_cls`` so the shim toggle is the single switching point. We swap both
        factories with recording stubs and assert the appropriate one fires for each branch."""
        from mlframe.training import trainer as tr_mod

        calls = {"classifier": 0, "regressor": 0}

        class _StubClassifier:
            """Groups tests covering stub classifier."""
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class _StubRegressor:
            """Groups tests covering stub regressor."""
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def fake_clf_factory():
            """Fake clf factory."""
            calls["classifier"] += 1
            return _StubClassifier

        def fake_reg_factory():
            """Fake reg factory."""
            calls["regressor"] += 1
            return _StubRegressor

        monkeypatch.setattr(tr_mod, "_lgb_classifier_cls", fake_clf_factory)
        monkeypatch.setattr(tr_mod, "_lgb_regressor_cls", fake_reg_factory)

        configs = tr_mod.get_training_configs(has_time=False)
        cpu_configs = configs

        # Classification branch -> classifier factory invoked.
        out = tr_mod._configure_lightgbm_params(
            configs=configs,
            cpu_configs=cpu_configs,
            use_regression=False,
            prefer_cpu_for_lightgbm=True,
            prefer_calibrated_classifiers=False,
            metamodel_func=lambda m: m,
        )
        assert calls["classifier"] == 1 and calls["regressor"] == 0
        assert isinstance(out["model"], _StubClassifier)

        # Regression branch -> regressor factory invoked.
        out = tr_mod._configure_lightgbm_params(
            configs=configs,
            cpu_configs=cpu_configs,
            use_regression=True,
            prefer_cpu_for_lightgbm=True,
            prefer_calibrated_classifiers=False,
            metamodel_func=lambda m: m,
        )
        assert calls["regressor"] == 1
        assert isinstance(out["model"], _StubRegressor)

        # NOTE: docstring at trainer.py:1113 references USE_LGB_DATASET_REUSE_SHIM as a
        # module-level revert toggle, but the symbol is not currently bound at the top of
        # trainer.py (only mentioned in docstrings + _model_factories.py comments). The factory
        # dispatch above is the substantive contract; if you wire the toggle constant in, add
        # ``assert hasattr(tr_mod, "USE_LGB_DATASET_REUSE_SHIM")`` here.


# =====================================================================
# 10. Pickle / joblib round-trip
# =====================================================================


class TestLGBShimPickleAndCacheLifecycle:
    """The 2026-04-24 prod regression on the XGB side was:

      ERROR ... ctypes objects containing pointers cannot be pickled

    A constructed lightgbm.Dataset has the same shape of issue: it
    holds a ctypes pointer (``self.handle``) to the C++ Dataset handle.
    The shim's __getstate__ must strip cache attrs before pickle.
    """

    def test_joblib_dump_load_round_trip(
        self,
        small_classification_data,
        tmp_path,
    ):
        """Full joblib round-trip -- the exact call path mlframe.training.io
        uses (``joblib.dump`` / ``joblib.load``)."""
        import joblib

        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(
            n_estimators=3,
            max_depth=3,
            **_QUIET_LGB,
        )
        m.fit(X, y)
        assert m._cached_train_dataset is not None, "precondition: fit must populate the cache"

        fpath = tmp_path / "shim.dump"
        # This is the operation that would fail without __getstate__ stripping.
        joblib.dump(m, fpath)

        loaded = joblib.load(fpath)
        # Loaded model produces the same predictions as the live one.
        np.testing.assert_allclose(
            loaded.predict_proba(X),
            m.predict_proba(X),
            atol=1e-6,
            err_msg="reloaded shim diverged from live shim",
        )
        # Cache is NOT inherited across save/load -- it's transient state.
        assert loaded._cached_train_dataset is None
        assert loaded._cached_train_key is None

    def test_getstate_strips_cache_pointers(self, small_classification_data):
        """Unit-level: ``__getstate__`` must return a dict whose cache
        pointer attrs are ``None`` regardless of whether the live
        instance holds a populated cache. The key attrs are nulled too
        (a key without its Dataset would silently "hit" stale data on
        load)."""
        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        m.fit(X, y)
        assert m._cached_train_dataset is not None  # precondition

        state = m.__getstate__()
        for _attr in (
            "_cached_train_dataset",
            "_cached_val_dataset",
            "_cached_train_key",
            "_cached_val_key",
        ):
            assert state.get(_attr) is None, (
                f"__getstate__ must null {_attr!r} before pickling -- the lightgbm.Dataset holds ctypes pointers that can't be serialised."
            )
        # But the LIVE instance's cache is untouched -- getstate must
        # not have mutated ``self.__dict__`` as a side effect.
        assert m._cached_train_dataset is not None

    def test_setstate_initialises_cache_attrs_when_missing(self):
        """Loading a saved model from a pre-shim era (no cache attrs in
        the pickled dict) must still produce a valid instance with
        initialised cache attrs."""
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)

        # Simulate an ancient state dict that lacked the cache attrs.
        legacy_state = {k: v for k, v in m.__getstate__().items() if not k.startswith("_cached_")}
        assert not any(k.startswith("_cached_") for k in legacy_state)

        m2 = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        m2.__setstate__(legacy_state)
        for _attr in (
            "_cached_train_dataset",
            "_cached_val_dataset",
            "_cached_train_key",
            "_cached_val_key",
        ):
            assert hasattr(m2, _attr), f"__setstate__ must re-init {_attr!r} for backward compatibility with pre-shim saves"
            assert getattr(m2, _attr) is None

    def test_clear_cache_releases_dataset(self, small_classification_data):
        """``clear_cache()`` must drop all cache references so the C++
        Dataset memory can be reclaimed."""
        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        m.fit(X, y)
        assert m._cached_train_dataset is not None

        m.clear_cache()
        assert m._cached_train_dataset is None
        assert m._cached_train_key is None
        assert m._cached_val_dataset is None
        assert m._cached_val_key is None

    def test_fit_after_clear_cache_rebuilds(self, small_classification_data):
        """After ``clear_cache()`` + a new ``fit()``, the cache repopulates
        -- clear is a reset, not a permanent disable."""
        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        m.fit(X, y)
        m.clear_cache()
        m.fit(X, y)
        assert m._cached_train_dataset is not None, "cache should rebuild on the fit after clear_cache"

    def test_regressor_shim_also_pickles(self, small_regression_data, tmp_path):
        """Regressor variant has the same pickle contract as the
        classifier -- the ``_DatasetReuseMixin`` override is shared."""
        import joblib

        X, y = small_regression_data
        m = LGBMRegressorWithDatasetReuse(
            n_estimators=3,
            max_depth=3,
            **_QUIET_LGB,
        )
        m.fit(X, y)
        fpath = tmp_path / "regressor_shim.dump"
        joblib.dump(m, fpath)
        loaded = joblib.load(fpath)
        np.testing.assert_allclose(
            loaded.predict(X),
            m.predict(X),
            atol=1e-6,
        )


# =====================================================================
# 11. Auto-clear_cache at end of strategy iter in core.py suite loop
# =====================================================================


class TestCoreAutoClearsLGBShimCacheAtStrategyEnd:
    """The auto-clear hook in core.py is duck-typed via
    ``getattr(est, 'clear_cache', None)`` -- so the LGB shim's
    ``clear_cache()`` is picked up alongside the XGB shim's. These
    tests pin down that the LGB shim implements the contract correctly.
    """

    def test_shim_clear_cache_preserves_booster(self, small_classification_data):
        """After ``clear_cache()``, the model must STILL be usable for
        predict: ``_Booster`` is a separate attribute attached at fit
        end, NOT part of the cache. The cache is fit-only scratchpad
        -- clear releases Dataset memory but keeps the trained booster
        intact. This is the invariant that makes auto-clear safe at
        strategy-iter end.
        """
        X, y = small_classification_data
        m = LGBMClassifierWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
        m.fit(X, y)
        probs_before = m.predict_proba(X)

        m.clear_cache()
        # Cache attrs wiped.
        assert m._cached_train_dataset is None
        # But Booster intact, predict still works + produces the SAME
        # probabilities (predict is deterministic given the Booster).
        probs_after = m.predict_proba(X)
        np.testing.assert_allclose(probs_after, probs_before, atol=1e-9)

    def test_duck_typing_skips_non_shim_lgb(self):
        """Safety: vanilla LGBMClassifier has no ``clear_cache`` -- the
        helper's ``callable`` check skips it harmlessly."""

        def _probe(est):
            """Probe."""
            fn = getattr(est, "clear_cache", None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
                    pass

        vanilla = lgb.LGBMClassifier(n_estimators=3, **_QUIET_LGB)
        _probe(vanilla)  # must not raise
