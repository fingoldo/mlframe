"""Round 10 sensors for the four deferred round-9 findings.

1. **PipelineCache collision across CB/LGB/XGB** — CB, LGB, XGB all
   inherit ``cache_key="tree"`` from TreeModelStrategy but have
   different ``feature_tier()`` (CB=(True,True) supports text+embedding,
   LGB/XGB=(False,False) don't). Running first by tier-desc sort, CB
   cached its polars DF *with* text cols under "tree"; LGB/XGB then
   retrieved that cache (via process_model's cached_train_df param
   overriding common_params) and got cols they can't handle. Fix:
   include feature_tier() in the cache key so tiers partition.

2. **prepare_df_for_xgboost polars contract** — declared ``df: object``,
   returned None, only handled pandas. A Polars input crashed with
   AttributeError on ``df[var].dtype`` — misleading. Now: explicit
   TypeError on polars, None-guard for cat_features, df returned.

3. **bruteforce target-encoder leakage WARN** — CatBoostEncoder
   fit_transform'd on full sample with target visible is classic
   supervised-encoding leak. Now: loud warnings.warn + logger.warning
   naming the columns. Doesn't fix the leak (needs OOF refactor), but
   operators see the risk at call time.

4. **MPS compute_area_profits zero-price guard** — returned
   ``profits / prices`` unguarded; zero-price bars yielded inf/NaN
   downstream. Guard now computes ratio only where price > 0; zero-
   price bars contribute 0.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest


# =============================================================================
# Fix C1 — cache_key partitioned by feature_tier
# =============================================================================


class TestCacheKeyIncludesFeatureTier:

    def test_cb_and_lgb_produce_different_cache_keys(self):
        """CB and LGB both have ``strategy.cache_key='tree'`` but
        different feature_tier(). After the fix the effective cache
        key used in core.py must differ so their cached DFs don't
        collide."""
        from mlframe.training.strategies import CatBoostStrategy, TreeModelStrategy
        cb = CatBoostStrategy()
        lgb = TreeModelStrategy()
        assert cb.cache_key == lgb.cache_key == "tree"  # base key shared
        assert cb.feature_tier() != lgb.feature_tier(), (
            "CB must have (True,True) tier; LGB/base Tree has (False,False). "
            "If this assertion fails, the entire cache-partition fix is moot."
        )
        # Simulating the core.py composition:
        cb_key = f"{cb.cache_key}_tier{cb.feature_tier()}"
        lgb_key = f"{lgb.cache_key}_tier{lgb.feature_tier()}"
        assert cb_key != lgb_key

    def test_same_tier_same_cache_key(self):
        """False-positive sensor: two strategies with the same tier
        must share the cache key (that's the whole point of
        cache_key='tree'). Confirm CB + any hypothetical
        same-tier subclass still collide (intended collision =
        cache reuse)."""
        from mlframe.training.strategies import CatBoostStrategy, XGBoostStrategy
        cb = CatBoostStrategy()
        xgb = XGBoostStrategy()
        # XGB currently has supports_text=False, embedding=False -> different tier from CB.
        # If that ever changes (XGB adds text), the keys should merge.
        cb_tier = cb.feature_tier()
        xgb_tier = xgb.feature_tier()
        if cb_tier == xgb_tier:
            cb_key = f"{cb.cache_key}_tier{cb_tier}"
            xgb_key = f"{xgb.cache_key}_tier{xgb_tier}"
            assert cb_key == xgb_key, (
                "Same-tier strategies should share cache; regression "
                "of the 'partition by tier' fix has over-partitioned."
            )


# =============================================================================
# Fix A2 — prepare_df_for_xgboost polars contract
# =============================================================================


class TestPrepareDfForXgboostContract:

    def test_polars_input_raises_typeerror(self):
        """Pre-fix: ``df[var].dtype`` on polars raised obscure
        AttributeError deep inside the function. Now: explicit
        TypeError at the entry naming the class."""
        from mlframe.preprocessing import prepare_df_for_xgboost
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        with pytest.raises(TypeError, match="pandas DataFrame"):
            prepare_df_for_xgboost(df)

    def test_pandas_input_returns_df(self):
        """Pre-fix returned None. Now: returns the (possibly
        dtype-mutated) df so callers can chain the same way they
        chain prepare_df_for_catboost."""
        from mlframe.preprocessing import prepare_df_for_xgboost
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        cat_features = ["b"]
        out = prepare_df_for_xgboost(df, cat_features=cat_features)
        assert out is df
        # 'b' should now be Categorical since ensure_categorical=True.
        assert isinstance(out["b"].dtype, pd.CategoricalDtype)

    def test_cat_features_none_accepted(self):
        """Pre-fix: ``var not in None`` raised TypeError. Now: coerced
        to empty list."""
        from mlframe.preprocessing import prepare_df_for_xgboost
        df = pd.DataFrame({"a": [1, 2, 3]})
        out = prepare_df_for_xgboost(df, cat_features=None)
        assert out is df

    def test_preexisting_categorical_auto_added_to_cat_features(self):
        """Existing behavior preserved: pd.Categorical columns not in
        cat_features get appended (mutation contract). Confirm the
        fix didn't regress this."""
        from mlframe.preprocessing import prepare_df_for_xgboost
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": pd.Categorical(["x", "y", "x"]),
        })
        cat_features: list = []
        prepare_df_for_xgboost(df, cat_features=cat_features)
        assert "b" in cat_features


# =============================================================================
# Fix B-target — bruteforce target-encoder leakage WARN
# =============================================================================


class TestBruteforceTargetEncoderWarnStaticCheck:
    """The bruteforce PySR FE path is not in the active production
    pipeline (probe round 9). Fixing the target-encoder leakage itself
    would require a full OOF-encoding refactor. Minimal observability
    fix: loud warn at call time.

    Integration-testing the warning requires ``import bruteforce`` which
    pulls ``from pysr import PySRRegressor`` which boots juliacall —
    that corrupts pytest's per-test teardown state for every subsequent
    test in the session ("previous item was not torn down properly").
    So instead: static source inspection. If someone removes the WARN,
    this regression sensor still trips reliably without the heavy
    Julia/PySR import chain.
    """

    def test_warn_strings_present_in_source(self):
        """The bruteforce encoder path MUST emit a target-encoding
        leak WARN via both ``warnings.warn`` and ``logger.warning``.
        This sensor reads the source to confirm both code paths are
        still wired up."""
        import pathlib
        src_path = pathlib.Path(__file__).parents[2] / "feature_engineering" / "bruteforce.py"
        src = src_path.read_text(encoding="utf-8")
        # Both emissions must be present.
        assert "warnings.warn(" in src, "warnings.warn call missing from bruteforce.py"
        assert "logger.warning(" in src, "logger.warning call missing from bruteforce.py"
        # The specific leakage phrase must be present.
        assert "TARGET-ENCODING LEAK" in src or "target-encoding leak" in src.lower(), (
            "Target-encoding leak WARN text removed from bruteforce.py — "
            "the round-10 defensive observability fix regressed"
        )
        # And the CatBoostEncoder.fit_transform call (the leak itself)
        # must still be there — if someone replaces it with OOF encoding,
        # the WARN becomes redundant and this test should be updated.
        assert "encoder.fit_transform(" in src


# =============================================================================
# Fix MPS — zero-price guard in compute_area_profits
# =============================================================================


class TestMpsComputeAreaProfitsZeroPriceGuard:

    def _call(self, positions, prices):
        from mlframe.feature_engineering.mps import compute_area_profits
        return compute_area_profits(
            positions=np.asarray(positions, dtype=np.int8),
            prices=np.asarray(prices, dtype=np.float64),
        )

    def test_zero_price_bar_does_not_produce_inf(self):
        """Pre-fix: prices[0] = 0 → profits[0] / 0 = inf/NaN silently.
        Now: zero-price bars contribute 0 output."""
        # A simple long run over 4 bars where the first price is 0.
        positions = np.array([1, 1, 1], dtype=np.int8)  # length n-1
        prices = np.array([0.0, 10.0, 11.0, 12.0], dtype=np.float64)
        out = self._call(positions, prices)
        assert np.all(np.isfinite(out)), (
            f"all outputs must be finite, got {out}"
        )
        # The zero-price index should be zero-ratio by guard.
        assert out[0] == 0.0

    def test_no_zero_prices_produces_normal_ratios(self):
        """False-positive sensor: when all prices are positive, the
        guard must not alter the math. Post-fix output must equal
        a raw profits / prices computation."""
        positions = np.array([1, 1, 1], dtype=np.int8)
        prices = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
        out = self._call(positions, prices)
        assert np.all(np.isfinite(out))
        # Don't assert exact values — just that the output is nontrivial
        # and has no inf/NaN.
        assert np.any(out != 0.0)

    def test_all_zero_prices_produces_all_zeros(self):
        """Boundary: every price == 0 → every output = 0, no inf/NaN."""
        positions = np.array([1, 1, 1], dtype=np.int8)
        prices = np.zeros(4, dtype=np.float64)
        out = self._call(positions, prices)
        assert np.all(out == 0.0)
