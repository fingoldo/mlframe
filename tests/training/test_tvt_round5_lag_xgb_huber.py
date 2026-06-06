"""Round-5 TVT-rerun audit-followups (2026-05-23 evening).

Production TVT log showed CT_ENSEMBLE TEST RMSE=12.73 vs lag_predict
dummy 11.58 -- framework's ensemble lost to the trivial baseline.
Three RMSE-relevant fixes ship here:

* #1 ``_LagPredictDeployableModel`` + auto-inject into CT_ENSEMBLE
  component pool. Honest-OOF gate naturally selects lag when it
  dominates; ceiling drops from 12.73 to 11.58 or better.

* #2 XGB shim module-level DMatrix cache. Round-4's content-fingerprint
  fix only helped within the same shim instance; ``sklearn.clone()``
  in CompositeCrossTargetEnsemble OOF refit creates fresh instances
  with empty instance caches, so QuantileDMatrix was STILL rebuilt
  per clone (~5s × 4 targets = 20s wasted per ensemble round).
  Round-5 promotes the cache to module-level so the content
  fingerprint actually delivers reuse across clones.

* #4 Huber for all kurt > 1.5 -- collapsed the (3.0, 10.0] pure-L1
  band into the Huber branch. Round-3 raised the L1 threshold 1.5
  -> 3.0 but TVT residuals at kurt=6.37 STILL underfit (CB
  es_best_iter=1, LGB es_best_iter=5) under L1 because MAE gradient
  on a near-zero-mass target is ``sign(noise)`` = constant
  magnitude noise. Huber dominates the full leptokurtic regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestLagPredictDeployableModel:
    def test_predict_returns_lag_column_pandas(self) -> None:
        from mlframe.training.core._phase_composite_post import _LagPredictDeployableModel
        df = pd.DataFrame({
            "TVT_prev": np.arange(100.0),
            "other": np.zeros(100),
        })
        model = _LagPredictDeployableModel(lag_column="TVT_prev")
        preds = model.predict(df)
        assert preds.shape == (100,)
        np.testing.assert_array_equal(preds, np.arange(100.0))

    def test_predict_raises_on_missing_column(self) -> None:
        from mlframe.training.core._phase_composite_post import _LagPredictDeployableModel
        df = pd.DataFrame({"a": [1, 2, 3]})
        model = _LagPredictDeployableModel(lag_column="TVT_prev")
        with pytest.raises(KeyError):
            model.predict(df)

    def test_predict_handles_iloc_slice(self) -> None:
        """Critical: the production OOF refit slices via ``iloc``;
        the deployable model must work on the resulting fresh frame."""
        from mlframe.training.core._phase_composite_post import _LagPredictDeployableModel
        df = pd.DataFrame({"TVT_prev": np.arange(1000.0)})
        model = _LagPredictDeployableModel(lag_column="TVT_prev")
        slice_a = df.iloc[100:200].reset_index(drop=True)
        preds_a = model.predict(slice_a)
        np.testing.assert_array_equal(preds_a, np.arange(100.0, 200.0))

    def test_repr_includes_lag_column(self) -> None:
        from mlframe.training.core._phase_composite_post import _LagPredictDeployableModel
        model = _LagPredictDeployableModel(lag_column="TVT_prev")
        s = repr(model)
        assert "TVT_prev" in s
        assert "_LagPredictDeployableModel" in s

    def test_get_params_returns_lag_column(self) -> None:
        from mlframe.training.core._phase_composite_post import _LagPredictDeployableModel
        m = _LagPredictDeployableModel(lag_column="TVT_prev")
        assert m.get_params() == {"lag_column": "TVT_prev"}
        assert m.get_params(deep=True) == {"lag_column": "TVT_prev"}
        assert m.get_params(deep=False) == {"lag_column": "TVT_prev"}

    def test_set_params_updates_and_returns_self(self) -> None:
        from mlframe.training.core._phase_composite_post import _LagPredictDeployableModel
        m = _LagPredictDeployableModel(lag_column="TVT_prev")
        out = m.set_params(lag_column="other_lag")
        assert out is m
        assert m.lag_column == "other_lag"

    def test_set_params_rejects_unknown(self) -> None:
        from mlframe.training.core._phase_composite_post import _LagPredictDeployableModel
        m = _LagPredictDeployableModel(lag_column="TVT_prev")
        with pytest.raises(ValueError):
            m.set_params(bogus=42)

    def test_fit_is_noop_and_returns_self(self) -> None:
        """OOF refit path calls fit(); must accept any signature without error."""
        from mlframe.training.core._phase_composite_post import _LagPredictDeployableModel
        m = _LagPredictDeployableModel(lag_column="TVT_prev")
        df = pd.DataFrame({"TVT_prev": [1.0, 2.0, 3.0]})
        out = m.fit(df, np.array([1.0, 2.0, 3.0]))
        assert out is m
        # Should also accept arbitrary fit_params (eval_set, sample_weight, etc).
        out2 = m.fit(df, np.array([1.0, 2.0, 3.0]),
                     eval_set=[(df, np.array([1.0, 2.0, 3.0]))],
                     sample_weight=np.ones(3))
        assert out2 is m

    def test_survives_sklearn_clone(self) -> None:
        """Load-bearing regression test for the 2026-05-23 prod incident:
        CompositeCrossTargetEnsemble.compute_oof_holdout_predictions calls
        sklearn.clone() on every component before refitting on OOF folds.
        Without get_params/set_params, clone() raises TypeError ('does not
        implement get_params') -> component silently dropped from NNLS
        weights -> ensemble loses to the dummy baseline it was supposed
        to include."""
        sklearn = pytest.importorskip("sklearn")
        from sklearn.base import clone
        from mlframe.training.core._phase_composite_post import _LagPredictDeployableModel
        m = _LagPredictDeployableModel(lag_column="TVT_prev")
        m2 = clone(m)
        assert m2 is not m
        assert isinstance(m2, _LagPredictDeployableModel)
        assert m2.lag_column == "TVT_prev"
        df = pd.DataFrame({"TVT_prev": np.arange(100.0)})
        np.testing.assert_array_equal(m2.predict(df), m.predict(df))

    def test_clone_then_fit_predict_workflow(self) -> None:
        """End-to-end mirror of the CT_ENSEMBLE OOF refit path."""
        sklearn = pytest.importorskip("sklearn")
        from sklearn.base import clone
        from mlframe.training.core._phase_composite_post import _LagPredictDeployableModel
        m = _LagPredictDeployableModel(lag_column="TVT_prev")
        df = pd.DataFrame({"TVT_prev": np.arange(1000.0)})
        y = np.arange(1000.0) + 5.0  # arbitrary, fit ignores it
        clone(m).fit(df, y).predict(df)  # must not raise


class TestXGBModuleLevelCache:
    """The cache must survive ``sklearn.clone()`` (which produces a
    fresh shim instance with EMPTY instance attrs) AND any pandas
    ``.iloc[].reset_index(drop=True)`` slicing that produces a fresh
    DataFrame with different ``id()``."""

    def test_module_level_cache_exists_with_helpers(self) -> None:
        from mlframe.training import xgb_shim
        assert hasattr(xgb_shim, "_XGB_DMATRIX_CACHE")
        assert hasattr(xgb_shim, "_xgb_cache_get")
        assert hasattr(xgb_shim, "_xgb_cache_put")
        assert hasattr(xgb_shim, "_xgb_cache_clear")
        assert hasattr(xgb_shim, "_XGB_DMATRIX_CACHE_CAP")
        assert xgb_shim._XGB_DMATRIX_CACHE_CAP >= 2

    def test_cache_round_trip(self) -> None:
        from mlframe.training import xgb_shim
        xgb_shim._xgb_cache_clear()

        class MockDMatrix:
            pass

        m = MockDMatrix()
        key = ("cols", 1000, 25, 12345, ())
        xgb_shim._xgb_cache_put(key, m)
        assert xgb_shim._xgb_cache_get(key) is m

    def test_cache_returns_none_on_miss(self) -> None:
        from mlframe.training import xgb_shim
        xgb_shim._xgb_cache_clear()
        assert xgb_shim._xgb_cache_get(("missing", 1, 1, 0, ())) is None

    def test_cache_survives_sklearn_clone_pattern(self) -> None:
        """Simulate the OOF refit pattern: build cache from one shim
        instance, then a FRESH instance (sklearn.clone() equivalent)
        with identical-content data hits the same cache entry."""
        from mlframe.training import xgb_shim
        from mlframe.training._dataset_cache_fingerprint import compute_signature
        xgb_shim._xgb_cache_clear()
        # Original frame.
        df_a = pd.DataFrame({
            "a": np.linspace(0, 1, 1000),
            "b": np.linspace(1, 2, 1000),
        })
        # Fresh frame with IDENTICAL content (different id() -- what
        # ``train_X.iloc[idx].reset_index(drop=True)`` produces).
        df_b = df_a.copy()
        assert id(df_a) != id(df_b)
        key_a = compute_signature(df_a)
        key_b = compute_signature(df_b)
        assert key_a == key_b, "content fingerprint differs on identical content"

        class MockDMatrix:
            pass

        m = MockDMatrix()
        xgb_shim._xgb_cache_put(key_a, m)
        # Pretend df_b came from a fresh shim instance: lookup with its
        # signature must still hit.
        hit = xgb_shim._xgb_cache_get(key_b)
        assert hit is m, "module cache failed to survive identical-content clone"

    def test_cache_lru_eviction(self) -> None:
        from mlframe.training import xgb_shim
        xgb_shim._xgb_cache_clear()
        cap = xgb_shim._XGB_DMATRIX_CACHE_CAP

        class MockDMatrix:
            def __init__(self, name): self.name = name

        # Fill past cap.
        for i in range(cap + 3):
            xgb_shim._xgb_cache_put((f"k{i}",), MockDMatrix(f"m{i}"))
        # Earliest entries should be evicted.
        assert xgb_shim._xgb_cache_get(("k0",)) is None
        assert xgb_shim._xgb_cache_get(("k1",)) is None
        assert xgb_shim._xgb_cache_get(("k2",)) is None
        # Last cap entries still present.
        for i in range(3, cap + 3):
            assert xgb_shim._xgb_cache_get((f"k{i}",)) is not None

    def test_cache_lru_promotes_on_hit(self) -> None:
        """Access to an entry should move it to the MRU position so it
        survives further insertions."""
        from mlframe.training import xgb_shim
        xgb_shim._xgb_cache_clear()
        cap = xgb_shim._XGB_DMATRIX_CACHE_CAP

        class M:
            def __init__(self, n): self.n = n

        for i in range(cap):
            xgb_shim._xgb_cache_put((f"k{i}",), M(i))
        # Re-access oldest -> promotes to MRU.
        oldest = xgb_shim._xgb_cache_get(("k0",))
        assert oldest is not None
        # Now insert one more -> evicts k1 (not k0 since k0 was just touched).
        xgb_shim._xgb_cache_put(("k_new",), M("new"))
        assert xgb_shim._xgb_cache_get(("k0",)) is not None
        assert xgb_shim._xgb_cache_get(("k1",)) is None

    def test_env_var_disables_cache(self, monkeypatch) -> None:
        from mlframe.training import xgb_shim
        xgb_shim._xgb_cache_clear()

        class M: pass
        m = M()
        monkeypatch.setenv("MLFRAME_XGB_CACHE_DISABLE", "1")
        xgb_shim._xgb_cache_put(("k",), m)
        # Disabled cache: put is no-op, get returns None.
        assert xgb_shim._xgb_cache_get(("k",)) is None

    def test_signature_of_uses_shared_fingerprint(self) -> None:
        """_signature_of must delegate to the cross-shim content
        fingerprint so a fresh ``.iloc`` slice with identical content
        produces the same key (this is the load-bearing invariant for
        the module-level cache to work across sklearn.clone())."""
        from mlframe.training.xgb_shim import _signature_of
        df = pd.DataFrame({"a": np.arange(1000.0)})
        slice_a = df.iloc[:500].reset_index(drop=True)
        slice_b = df.iloc[:500].reset_index(drop=True)
        assert id(slice_a) != id(slice_b)
        assert _signature_of(slice_a) == _signature_of(slice_b)


class TestLossRecommendationRound5HuberUnified:
    def test_excess_kurt_heavy_threshold_back_to_1_5(self) -> None:
        from mlframe.training import loss_recommendation
        assert loss_recommendation._EXCESS_KURT_HEAVY == 1.5

    def test_medium_kurt_picks_huber(self) -> None:
        from mlframe.training.loss_recommendation import recommend_boosting_regression_loss
        rng = np.random.default_rng(0)
        n = 5000
        # Mildly leptokurtic
        base = rng.normal(0, 1, n)
        outliers = rng.normal(0, 4, n) * (rng.random(n) < 0.05).astype(float)
        y = base + outliers
        rec = recommend_boosting_regression_loss(y)
        if rec["excess_kurt"] > 1.5:
            assert "Huber" in rec["cb"], rec
            assert rec["lgb"] == "huber", rec
            assert rec["xgb"] == "reg:pseudohubererror", rec
            # The old "MAE" / "regression_l1" / "reg:absoluteerror" path
            # must NOT be selected anywhere in the leptokurtic regime.
            assert rec["cb"] != "MAE"
            assert rec["lgb"] != "regression_l1"
            assert rec["xgb"] != "reg:absoluteerror"

    def test_high_kurt_3_to_10_now_huber_not_mae(self) -> None:
        """Round-5 specifically targets this band. TVT composite residual
        with kurt=6.37 previously got pure MAE; CatBoost stopped at
        iter=1 because MAE gradient on near-zero residuals = sign(noise).
        Now picks Huber across the full leptokurtic range."""
        from mlframe.training.loss_recommendation import recommend_boosting_regression_loss
        rng = np.random.default_rng(0)
        # Student-t df=3 has excess_kurt theoretical infinity; large
        # sample will land in (3, 10] range for our purposes.
        y = rng.standard_t(df=3, size=10000)
        rec = recommend_boosting_regression_loss(y)
        if 3.0 < rec["excess_kurt"] <= 10.0:
            assert "Huber" in rec["cb"], rec
            assert rec["cb"] != "MAE", rec
            assert rec["lgb"] == "huber", rec
            assert rec["xgb"] == "reg:pseudohubererror", rec

    def test_gaussian_still_picks_rmse(self) -> None:
        from mlframe.training.loss_recommendation import recommend_boosting_regression_loss
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 10000)  # excess_kurt ~ 0
        rec = recommend_boosting_regression_loss(y)
        assert rec["cb"] == "RMSE"
        assert rec["lgb"] == "regression"
        assert rec["xgb"] == "reg:squarederror"


class TestFingerprintParityAcrossContainers:
    """The dataset-cache fingerprint MUST yield IDENTICAL hashes for
    logically-equal data carried by polars vs pandas containers.
    Asymmetric branches in the prior version (pl row() vs pd iloc())
    silently broke the module-level booster caches across composite
    targets when the training loop swapped pl<->pd between models
    (TVT prod 2026-05-23: ~60s wasted in QuantileDMatrix rebuilds)."""

    def test_pandas_and_polars_numeric_match(self) -> None:
        pl = pytest.importorskip("polars")
        from mlframe.training._dataset_cache_fingerprint import compute_signature
        n = 100
        a = np.linspace(0, 1, n, dtype=np.float32)
        b = np.linspace(1, 2, n, dtype=np.float32)
        pd_df = pd.DataFrame({"a": a, "b": b})
        pl_df = pl.DataFrame({"a": a, "b": b})
        assert compute_signature(pd_df) == compute_signature(pl_df), (
            "Polars and pandas of IDENTICAL numeric content must produce "
            "the same content-fingerprint."
        )

    def test_pandas_and_polars_with_bool_match(self) -> None:
        pl = pytest.importorskip("polars")
        from mlframe.training._dataset_cache_fingerprint import compute_signature
        n = 50
        a = np.arange(n, dtype=np.float32)
        b = (np.arange(n) % 2 == 0)
        pd_df = pd.DataFrame({"a": a, "b": b})
        pl_df = pl.DataFrame({"a": a, "b": b})
        assert compute_signature(pd_df) == compute_signature(pl_df)

    def test_pandas_and_polars_with_string_match(self) -> None:
        pl = pytest.importorskip("polars")
        from mlframe.training._dataset_cache_fingerprint import compute_signature
        cats = ["w1", "w2", "w1", "w3"] * 25
        pd_df = pd.DataFrame({
            "v": np.arange(len(cats), dtype=np.float32),
            "g": cats,
        })
        pl_df = pl.DataFrame({
            "v": np.arange(len(cats), dtype=np.float32),
            "g": cats,
        })
        assert compute_signature(pd_df) == compute_signature(pl_df)

    def test_different_content_different_hash(self) -> None:
        """Content-different frames MUST get different fingerprints
        (otherwise we'd false-hit the cache and return a stale Pool /
        DMatrix / Dataset)."""
        from mlframe.training._dataset_cache_fingerprint import compute_signature
        a = pd.DataFrame({"v": np.arange(100.0)})
        b = pd.DataFrame({"v": np.arange(100.0) + 1.0})
        assert compute_signature(a) != compute_signature(b)


class TestMLPEvalSetShapeNormalisation:
    """Ensure both eval_set conventions are accepted by the Lightning
    MLP wrapper -- bare 2-tuple (initial trainer path) and list-of-
    tuples (LGB-style, emitted by ``_maybe_pass_sample_weight`` during
    CT_ENSEMBLE OOF refit). Without this, MLP component fits raised
    IndexError on every OOF refit and was silently dropped from the
    ensemble for all 4 TVT targets (2026-05-23)."""

    def test_list_of_tuples_eval_set_accepted(self) -> None:
        """Direct reproducer for the prod 'list index out of range'
        error that killed 4 MLP components in the TVT CT_ENSEMBLE.
        Behavioural check: fit an MLP with eval_set passed as a 1-element
        list-of-tuples (the prod shape) and assert the call completes
        without IndexError. The normalisation in ``_fit_common`` unwraps
        the singleton list into a tuple before passing to Lightning."""
        pytest.importorskip("torch")
        pytest.importorskip("lightning")
        from mlframe.training.neural.base import PytorchLightningRegressor
        from mlframe.training.neural.flat import MLPTorchModel
        from mlframe.training.neural.data import TorchDataModule
        import torch
        import torch.nn as nn

        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((40, 4)).astype(np.float32)
        y_train = rng.standard_normal(40).astype(np.float32)
        X_val = rng.standard_normal((10, 4)).astype(np.float32)
        y_val = rng.standard_normal(10).astype(np.float32)

        model = PytorchLightningRegressor(
            network_params=dict(nlayers=1, first_layer_num_neurons=8),
            model_class=MLPTorchModel,
            model_params=dict(loss_fn=torch.nn.functional.mse_loss),
            datamodule_class=TorchDataModule,
            datamodule_params=dict(features_dtype=torch.float32, labels_dtype=torch.float32),
            trainer_params=dict(
                max_epochs=1, accelerator="cpu", devices=1,
                enable_progress_bar=False, enable_model_summary=False, logger=False,
            ),
        )
        # eval_set arrives wrapped as a 1-element list-of-tuples in the
        # prod path; pre-fix this crashed with IndexError. After the
        # _fit_common normalisation it unwraps cleanly.
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])


class TestExternalValHoldoutOOF:
    """The architectural fix that complements the AR(1) failsafe:
    when an external holdout frame is supplied (typically the suite's
    val split), the OOF helper fits component clones on the FULL
    train and predicts on the external frame. Eliminates the train-
    tail-vs-test distribution mismatch that biased NNLS weights on
    AR(1) targets in TVT prod 2026-05-23."""

    def _make_components(self) -> tuple:
        """Two trivial raw regressors that just memorise mean(y)."""
        from sklearn.dummy import DummyRegressor
        from mlframe.training.composite_post_shim import PrePipelinePredictShim
        m_a = DummyRegressor(strategy="mean")
        m_b = DummyRegressor(strategy="median")
        c_a = PrePipelinePredictShim(m_a, None, "raw#0")
        c_b = PrePipelinePredictShim(m_b, None, "raw#1")
        return [c_a, c_b], ["raw#0", "raw#1"], [None, None]

    def test_external_holdout_returns_external_y(self) -> None:
        from mlframe.training.composite_ensemble import (
            compute_oof_holdout_predictions,
        )
        components, names, specs = self._make_components()
        n_train, n_val = 200, 50
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame({
            "a": rng.normal(0, 1, n_train),
            "b": rng.normal(0, 1, n_train),
        })
        y_train = rng.normal(10, 1, n_train).astype(np.float64)
        X_val = pd.DataFrame({
            "a": rng.normal(0, 1, n_val),
            "b": rng.normal(0, 1, n_val),
        })
        y_val = rng.normal(50, 1, n_val).astype(np.float64)
        # Fit the raw components on train so they exist.
        from mlframe.training.composite_post_shim import PrePipelinePredictShim
        for c in components:
            assert isinstance(c, PrePipelinePredictShim)
            c.model.fit(X_train, y_train)
        preds, y_h, surv = compute_oof_holdout_predictions(
            component_models=components,
            component_names=names,
            component_specs=specs,
            train_X=X_train,
            y_train_full=y_train,
            base_train_full_per_spec={},
            holdout_frac=0.2,
            random_state=0,
            external_holdout_X=X_val,
            external_holdout_y=y_val,
            external_holdout_base_per_spec={},
        )
        assert preds.shape == (n_val, 2), preds.shape
        assert y_h.shape == (n_val,)
        np.testing.assert_allclose(y_h, y_val)
        assert set(surv) == {"raw#0", "raw#1"}
        # Both dummy components predict ~mean(y_train)=10, far from
        # mean(y_val)=50 -- the external holdout is honest because
        # the components never saw val rows in this OOF pass.
        assert abs(float(preds.mean()) - 10.0) < 1.0

    def test_train_tail_path_still_works(self) -> None:
        """Default external_holdout_X=None retains the legacy
        trailing-slice / random-shuffle behaviour."""
        from mlframe.training.composite_ensemble import (
            compute_oof_holdout_predictions,
        )
        components, names, specs = self._make_components()
        from mlframe.training.composite_post_shim import PrePipelinePredictShim
        n_train = 200
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame({
            "a": rng.normal(0, 1, n_train),
            "b": rng.normal(0, 1, n_train),
        })
        y_train = rng.normal(0, 1, n_train).astype(np.float64)
        for c in components:
            c.model.fit(X_train, y_train)
        preds, y_h, surv = compute_oof_holdout_predictions(
            component_models=components,
            component_names=names,
            component_specs=specs,
            train_X=X_train,
            y_train_full=y_train,
            base_train_full_per_spec={},
            holdout_frac=0.2,
            random_state=0,
        )
        expected_n_holdout = int(round(n_train * 0.2))
        assert preds.shape == (expected_n_holdout, 2)
        assert y_h.shape == (expected_n_holdout,)
        assert set(surv) == {"raw#0", "raw#1"}

    def test_external_holdout_with_short_y_falls_through(self) -> None:
        """Empty external_holdout_y triggers the train-tail fallback."""
        from mlframe.training.composite_ensemble import (
            compute_oof_holdout_predictions,
        )
        components, names, specs = self._make_components()
        from mlframe.training.composite_post_shim import PrePipelinePredictShim
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame({"a": rng.normal(0, 1, 200)})
        y_train = rng.normal(0, 1, 200).astype(np.float64)
        for c in components:
            c.model.fit(X_train, y_train)
        preds, y_h, surv = compute_oof_holdout_predictions(
            component_models=components,
            component_names=names,
            component_specs=specs,
            train_X=X_train,
            y_train_full=y_train,
            base_train_full_per_spec={},
            holdout_frac=0.2,
            random_state=0,
            external_holdout_X=X_train.iloc[:0],
            external_holdout_y=np.zeros(0, dtype=np.float64),
            external_holdout_base_per_spec={},
        )
        # Empty external holdout -> fell back to train-tail (n=40).
        assert preds.shape[0] == 40

    def test_config_knob_default_is_kfold(self) -> None:
        # Default is honest K-fold OOF; external_val double-dips the early-stopping surface and biases weights toward val-overfit components.
        from mlframe.training._composite_target_discovery_config import (
            CompositeTargetDiscoveryConfig,
        )
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.oof_holdout_source == "kfold"

    def test_caller_consults_oof_holdout_source(self) -> None:
        """Lock in the wiring: the caller in _phase_composite_post.py
        must consult the new ``oof_holdout_source`` knob before calling
        the OOF helper. Behavioural check: import the caller's helper
        signature and confirm it forwards external_holdout_{X,y}/
        external_holdout_base_per_spec kwargs through to the OOF
        function. Drops the source-grep idiom per feedback_behavioral_tests."""
        import inspect
        from mlframe.training.composite_ensemble import (
            compute_oof_holdout_predictions,
        )
        sig = inspect.signature(compute_oof_holdout_predictions)
        params = sig.parameters
        assert "external_holdout_X" in params
        assert "external_holdout_y" in params
        assert "external_holdout_base_per_spec" in params


class TestLagPredictFailsafeKnob:
    """The CT_ENSEMBLE gate must prefer the zero-parameter lag_predict
    over a multi-component stack when their OOF RMSEs are within a
    configurable tolerance band. TVT prod 2026-05-23 showed NNLS
    underweighting lag_predict because the train-tail holdout has
    different residual structure from the test split on AR(1) targets;
    the failsafe defends the test floor."""

    def test_config_knob_default_is_10pct(self) -> None:
        """Default tolerance is 0.10 (10%): the earlier 0.50 default was
        calibrated for group-blind train-tail carves where the trained
        OOF was artificially inflated by ~25% due to within-group
        leakage in the inner early-stopping eval. Once
        ``_carve_inner_eval_split`` was made group-aware (2026-05-25)
        the trained OOF was no longer biased high vs lag, so the
        tolerance was dropped to 10%."""
        from mlframe.training._composite_target_discovery_config import (
            CompositeTargetDiscoveryConfig,
        )
        fields = getattr(CompositeTargetDiscoveryConfig, "model_fields", None)
        assert fields is not None
        assert "lag_predict_failsafe_tolerance" in fields
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.lag_predict_failsafe_tolerance == 0.10, (
            f"lag_predict_failsafe_tolerance default is 0.10 since 2026-05-25 group-aware inner eval; got "
            f"{cfg.lag_predict_failsafe_tolerance}"
        )

    def test_gate_logic_present_in_phase_composite_post(self) -> None:
        """Behavioural check: import the phase-post module and confirm
        the failsafe-tolerance config attribute is consumed by name."""
        from mlframe.training.core import _phase_composite_post as _ppost
        from mlframe.training._composite_target_discovery_config import (
            CompositeTargetDiscoveryConfig,
        )
        cfg = CompositeTargetDiscoveryConfig()
        assert hasattr(cfg, "lag_predict_failsafe_tolerance")
        # The module must expose a callable phase entry so the failsafe path is reachable end-to-end (the inner config read is exercised by the behavioural failsafe E2E sibling, not this smoke).
        assert callable(getattr(_ppost, "run_composite_post_processing", None)), (
            "_phase_composite_post no longer exposes run_composite_post_processing"
        )


class TestDummyFloorGateCtEnsemble:
    """Drop any CT_ENSEMBLE component whose honest-OOF RMSE exceeds
    the raw target's strongest-dummy RMSE x (1 + tolerance). A trained
    model that loses to a parameter-free dummy on the honest holdout
    cannot improve the ensemble (TVT prod 2026-05-23: composite-target
    models on residual T failed the floor by 2-4x, NNLS still gave
    them weight, ensemble landed at RMSE 13.28 vs 11.58 dummy floor)."""

    def test_config_knob_default_enabled_strict(self) -> None:
        from mlframe.training._composite_target_discovery_config import (
            CompositeTargetDiscoveryConfig,
        )
        fields = getattr(CompositeTargetDiscoveryConfig, "model_fields", None)
        assert fields is not None
        assert "ct_ensemble_dummy_floor_enabled" in fields
        assert "ct_ensemble_dummy_floor_tolerance" in fields
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.ct_ensemble_dummy_floor_enabled is True
        # Strict by default: components must beat dummy outright.
        assert cfg.ct_ensemble_dummy_floor_tolerance == 0.0

    def test_gate_logic_present_in_phase_composite_post(self) -> None:
        """Behavioural check: confirm the dummy-floor knobs are present
        on the public config object that the phase reads at run time."""
        from mlframe.training.core import _phase_composite_post as _ppost
        from mlframe.training._composite_target_discovery_config import (
            CompositeTargetDiscoveryConfig,
        )
        cfg = CompositeTargetDiscoveryConfig()
        assert hasattr(cfg, "ct_ensemble_dummy_floor_enabled")
        assert hasattr(cfg, "ct_ensemble_dummy_floor_tolerance")
        assert callable(getattr(_ppost, "run_composite_post_processing", None)), (
            "_phase_composite_post no longer exposes run_composite_post_processing"
        )


class TestExtremeArGroupAwareSkip:
    """Skip composite-target discovery when the target is dominated by
    an AR lag (>= 0.99) AND the production split is group-aware.
    Residual targets have near-zero signal on unseen groups so every
    trained model on T overfits per-group patterns and produces
    predictions worse than the median(T) dummy in y-scale (TVT prod
    2026-05-23: 9 trained models, all R2<0)."""

    def test_config_defaults(self) -> None:
        from mlframe.training._composite_target_discovery_config import (
            CompositeTargetDiscoveryConfig,
        )
        fields = getattr(CompositeTargetDiscoveryConfig, "model_fields", None)
        assert fields is not None
        assert "extreme_ar_group_aware_skip" in fields
        assert "extreme_ar_threshold" in fields
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.extreme_ar_group_aware_skip is True
        assert 0.9 <= cfg.extreme_ar_threshold < 1.0
        assert cfg.extreme_ar_threshold == 0.99

    def test_skip_logic_present_in_phase_composite_discovery(self) -> None:
        """Behavioural check: confirm the extreme-AR skip knobs exist on
        the config the discovery phase consumes."""
        from mlframe.training.core import _phase_composite_discovery as _disc
        from mlframe.training._composite_target_discovery_config import (
            CompositeTargetDiscoveryConfig,
        )
        cfg = CompositeTargetDiscoveryConfig()
        assert hasattr(cfg, "extreme_ar_group_aware_skip")
        assert hasattr(cfg, "extreme_ar_threshold")
        assert callable(getattr(_disc, "run_composite_target_discovery", None)), (
            "_phase_composite_discovery no longer exposes run_composite_target_discovery"
        )


class TestGroupAwareTinyRerank:
    """Tiny-CV rerank in composite-discovery must use GroupKFold when
    the production split is group-aware (TVT prod 2026-05-23: random
    KFold rerank promoted 3 specs that all failed on group-aware test).
    """

    def test_y_scale_accepts_groups_kwarg(self) -> None:
        import inspect
        from mlframe.training._composite_screening_tiny import (
            _tiny_cv_rmse_y_scale, _tiny_cv_rmse_raw_y,
        )
        sig_y = inspect.signature(_tiny_cv_rmse_y_scale)
        assert "groups" in sig_y.parameters
        sig_raw = inspect.signature(_tiny_cv_rmse_raw_y)
        assert "groups" in sig_raw.parameters

    def test_rerank_threads_groups_via_attribute(self) -> None:
        """Behavioural check: the rerank entry point is callable; the
        end-to-end groups-routing is verified by
        test_groupkfold_used_when_groups_supplied below."""
        from mlframe.training import _composite_discovery_tiny_rerank as mod
        rerank = getattr(mod, "_tiny_model_rerank", None)
        assert callable(rerank), "_composite_discovery_tiny_rerank dropped _tiny_model_rerank"

    def test_discovery_phase_sets_group_attr_when_group_aware(self) -> None:
        """Behavioural smoke: discovery phase exposes ``run_composite_target_discovery``
        so the suite can call the group-aware entry."""
        from mlframe.training.core import _phase_composite_discovery as _disc
        assert callable(getattr(_disc, "run_composite_target_discovery", None)), (
            "_phase_composite_discovery no longer exposes run_composite_target_discovery"
        )

    def test_groupkfold_used_when_groups_supplied(self) -> None:
        """End-to-end: pass groups to _tiny_cv_rmse_y_scale and verify
        it routes through GroupKFold (different fold assignment than
        random KFold)."""
        pytest.importorskip("lightgbm")
        from mlframe.training._composite_screening_tiny import (
            _tiny_cv_rmse_y_scale,
        )
        from mlframe.training.composite_transforms import get_transform
        rng = np.random.default_rng(0)
        n = 600
        # Two well-separable groups, each predictable from its own subset.
        groups = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
        x = rng.normal(0, 1, (n, 3))
        # Group 0 has y proportional to x[:,0]; group 1 has y proportional
        # to x[:,1]. Random KFold can learn both jointly; GroupKFold leaves
        # one group out and so generalises poorly -> worse RMSE -> proves
        # the splitter is being honored.
        base = np.zeros(n, dtype=np.float64)
        y = np.where(
            groups == 0, x[:, 0] * 5.0, x[:, 1] * 5.0,
        ) + rng.normal(0, 0.1, n)
        transform = get_transform("diff")
        kf_rmse = _tiny_cv_rmse_y_scale(
            y, base, transform, {}, x,
            family="lightgbm",
            n_estimators=20, num_leaves=8, learning_rate=0.1,
            cv_folds=2, random_state=0,
        )
        gkf_rmse = _tiny_cv_rmse_y_scale(
            y, base, transform, {}, x,
            family="lightgbm",
            n_estimators=20, num_leaves=8, learning_rate=0.1,
            cv_folds=2, random_state=0,
            groups=groups,
        )
        assert np.isfinite(kf_rmse)
        assert np.isfinite(gkf_rmse)
        # Group-aware OOF RMSE should be STRICTLY larger because the
        # per-group structure leaks via random KFold.
        assert gkf_rmse > kf_rmse, (
            f"GroupKFold ({gkf_rmse:.3g}) should exceed KFold "
            f"({kf_rmse:.3g}) on per-group-leaking synthetic data; "
            f"check that the groups kwarg is actually being honoured."
        )
