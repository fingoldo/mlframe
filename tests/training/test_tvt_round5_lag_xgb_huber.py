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
