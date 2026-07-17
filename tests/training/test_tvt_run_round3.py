"""Regression tests for the 2026-05-23 TVT-rerun (round 3) audit followups.

Production TVT run with tree boosters added to the zoo produced
**CT_ENSEMBLE TEST RMSE=12.82** -- worse than the trivial lag_predict
dummy (11.58) AND worse than Ridge raw (11.63). Four agents audit:

* Agent A: Booster under-convergence. ``def_regr_metric="MAE"`` is used
  as eval-metric for ES even when objective is RMSE -> mismatch causes
  premature ES (CB iter=147 / LGB iter=76 on 5000-iter cap).
  Auto-loss ``kurt > 1.5`` switches objective to MAE/L1 on composite
  residuals near zero -> MAE gradient = sign(noise) constant magnitude
  -> CB stops at iter=1 on TVT-addres-TVT_prev.

* Agent B: CT_ENSEMBLE silently drops 4/20 components (every LGBM clone)
  because OOF refit doesn't pass ``eval_set`` -- early_stopping callback
  on LGBM clones raises "at least one dataset and eval metric is
  required for evaluation". Ensemble fitted on the weaker surviving 16
  components -> RMSE 12.82.

* Agent D: XGB ``_signature_of`` keys on ``id(X)`` -- after
  ``sklearn.clone()`` the shim cache is empty AND ``train_X.iloc[idx]``
  produces fresh frames -> DMatrix rebuilt every OOF round.

These tests cement the four code fixes landed in this commit.
"""

from __future__ import annotations

import inspect

import numpy as np


class TestOOFRefitEvalSetPassthrough:
    def test_maybe_pass_sample_weight_accepts_eval_set(self) -> None:
        from mlframe.training.composite.ensemble import _maybe_pass_sample_weight

        sig = inspect.signature(_maybe_pass_sample_weight)
        assert "eval_set" in sig.parameters, (
            "OOF refit helper must accept eval_set so LightGBM clones with "
            "early_stopping_rounds callback don't crash on missing eval data "
            "(production TVT 2026-05-23: 4/20 components silently dropped)"
        )

    def test_carve_inner_eval_split_returns_tail(self) -> None:
        import pandas as pd
        from mlframe.training.composite.ensemble import _carve_inner_eval_split

        X = pd.DataFrame({"x": np.arange(2000, dtype=np.float64)})
        y = np.arange(2000, dtype=np.float64)
        X_fit, _y_fit, X_eval, y_eval = _carve_inner_eval_split(X, y, frac=0.1)
        assert X_eval is not None
        assert y_eval is not None
        assert len(X_fit) == 1800
        assert len(X_eval) == 200
        # Tail split: eval rows are the LAST 200.
        assert int(y_eval[0]) == 1800
        assert int(y_eval[-1]) == 1999

    def test_carve_inner_eval_split_skips_below_threshold(self) -> None:
        import pandas as pd
        from mlframe.training.composite.ensemble import _carve_inner_eval_split

        X = pd.DataFrame({"x": np.arange(500, dtype=np.float64)})
        y = np.arange(500, dtype=np.float64)
        X_fit, _y_fit, X_eval, y_eval = _carve_inner_eval_split(X, y)
        assert X_eval is None
        assert y_eval is None
        assert len(X_fit) == 500

    def test_passes_eval_set_to_lightgbm_like_estimator(self) -> None:
        """A mock estimator whose fit signature accepts ``eval_set``
        should receive it; verifies the inspect-based dispatch path."""
        from mlframe.training.composite.ensemble import _maybe_pass_sample_weight

        calls = []

        class Fake:
            def fit(self, X, y, sample_weight=None, eval_set=None):
                calls.append({"sw": sample_weight, "es": eval_set})

        m = Fake()
        X = np.ones((10, 3))
        y = np.zeros(10)
        sw = np.ones(10)
        _maybe_pass_sample_weight(
            m,
            X,
            y,
            sw,
            eval_set=(X[5:], y[5:]),
        )
        assert calls, "fit was not invoked"
        assert calls[0]["sw"] is sw
        assert calls[0]["es"] is not None
        assert len(calls[0]["es"]) == 1  # normalised to list-of-tuples


class TestLossRecommendationHuberBand:
    def test_kurt_threshold_back_to_1_5_with_huber_unified_band(self) -> None:
        """Round-5 (2026-05-23 evening): threshold lowered back 3.0 -> 1.5
        AND the pure-L1 (3.0, 10.0] band was collapsed into the Huber
        branch. Production TVT residuals at kurt=6.37 STILL underfit
        under pure L1 (CB es_best_iter=1) DESPITE the round-3 threshold
        raise. Huber covers the entire leptokurtic regime now."""
        from mlframe.training import loss_recommendation

        assert loss_recommendation._EXCESS_KURT_HEAVY == 1.5

    def test_huber_for_all_leptokurtic(self) -> None:
        """Both medium (~2.5) and high (~6) kurt should pick Huber, not
        the old MAE path."""
        from mlframe.training.loss_recommendation import recommend_boosting_regression_loss

        rng = np.random.default_rng(0)
        # Medium-kurt Laplace-like
        n = 5000
        base = rng.normal(0, 1, n)
        outliers = rng.normal(0, 4, n) * (rng.random(n) < 0.05).astype(float)
        y = base + outliers
        rec = recommend_boosting_regression_loss(y)
        if rec["excess_kurt"] > 1.5:
            assert "Huber" in rec["cb"], rec
            assert rec["lgb"] == "huber", rec
            assert rec["xgb"] == "reg:pseudohubererror", rec

    def test_high_kurt_now_picks_huber_too(self) -> None:
        """High kurt (~6+) previously picked MAE; round-5 collapses
        into Huber for the same gradient-on-near-zero-residuals reason
        that bit CB es_best_iter=1 on production TVT composite residual.

        ``standard_t(df=3)`` was the original choice but it empirically
        lands at sample kurt > 20 on n=5000 -- which trips the
        ``_EXCESS_KURT_HUBER_FAILS`` threshold and reverts the
        recommendation to RMSE (the Huber gradient collapses at
        excessive kurt; production check added later). df=6 puts the
        sample reliably in the [1.5, 20] heavy-but-not-extreme band
        where the Huber recommendation still holds.
        """
        from mlframe.training.loss_recommendation import recommend_boosting_regression_loss

        rng = np.random.default_rng(0)
        n = 5000
        y = rng.standard_t(df=6, size=n)  # kurt ~ 3 (within [1.5, 20])
        rec = recommend_boosting_regression_loss(y)
        if rec["excess_kurt"] > 1.5:
            assert "Huber" in rec["cb"], rec
            assert rec["lgb"] == "huber", rec
            assert rec["xgb"] == "reg:pseudohubererror", rec


class TestXGBShimContentFingerprint:
    """``_signature_of`` must NOT include ``id(X)`` -- pandas
    ``.iloc[idx]`` produces fresh frames with different ids but
    identical logical content. ``sklearn.clone()`` on the wrapper
    instantiates fresh shim instances with empty caches; a content
    fingerprint lets two ``.iloc`` views of the same source DataFrame
    hit the cache."""

    def test_signature_does_not_depend_on_id(self) -> None:
        from mlframe.training.xgb_shim import _signature_of
        import pandas as pd

        df = pd.DataFrame(
            {
                "a": np.linspace(0, 1, 1000),
                "b": np.linspace(1, 2, 1000),
            }
        )
        # Two views of the SAME logical data via .iloc -- different ids.
        df_a = df.iloc[:500].reset_index(drop=True)
        df_b = df.iloc[:500].reset_index(drop=True)
        assert id(df_a) != id(df_b)
        sig_a = _signature_of(df_a)
        sig_b = _signature_of(df_b)
        assert sig_a == sig_b, f"content fingerprint mismatch on identical data: {sig_a} vs {sig_b}"

    def test_signature_distinguishes_different_content(self) -> None:
        from mlframe.training.xgb_shim import _signature_of
        import pandas as pd

        df1 = pd.DataFrame({"a": np.linspace(0, 1, 1000)})
        df2 = pd.DataFrame({"a": np.linspace(0, 2, 1000)})  # different content
        sig1 = _signature_of(df1)
        sig2 = _signature_of(df2)
        assert sig1 != sig2, f"content fingerprint failed to distinguish different content: {sig1} vs {sig2}"


class TestLGBLearningRatePlumbing:
    def test_lgb_general_params_contains_learning_rate(self) -> None:
        """LGB_GENERAL_PARAMS must include ``learning_rate`` so the
        suite's ``learning_rate=`` kwarg actually overrides LightGBM's
        library default. Pre-fix the key was missing entirely -- silent
        override-surface gap."""
        from mlframe.training.helpers import get_training_configs

        configs = get_training_configs(learning_rate=0.05)
        # ``get_training_configs`` returns a SimpleNamespace; the LGB
        # params live on the ``LGB_GENERAL_PARAMS`` attribute. Use
        # ``getattr`` so this stays portable if the return type is
        # tightened to a Pydantic model later.
        lgb_params = getattr(configs, "LGB_GENERAL_PARAMS", None)
        assert lgb_params is not None, f"LGB_GENERAL_PARAMS attribute missing on return of get_training_configs; got attrs: {dir(configs)}"
        assert "learning_rate" in lgb_params, f"LGB_GENERAL_PARAMS missing learning_rate: keys={list(lgb_params.keys())}"
        assert lgb_params["learning_rate"] == 0.05
